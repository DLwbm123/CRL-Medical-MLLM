"""Label-free frozen real/masked conditional answer scores, two fixed orders."""
import hashlib
import json
import os
import signal
import time
from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from continual import Deadline, load_input
from state import atomic_json
from p23_reward import SCHEME, validate, messages, decision


def main():
    folder = Path(os.environ["P0_ROOT"]) / "outputs" / os.environ["P2_CAMPAIGN"]
    raw = (folder / "manifest.json").read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    if digest != (folder / "manifest.sha256").read_text().strip(): raise ValueError("Manifest changed")
    manifest = json.loads(raw); validate(manifest)
    calibration_mode = os.environ["P2_MODE"] == "p23_calibration_read"
    reference = folder / "reference" / ("calibration" if calibration_mode else "v")
    destination = reference / ("readouts.json" if calibration_mode else "targets.json")
    if destination.exists(): raise ValueError("Frozen readout cannot replace an existing seal")
    calibration = None
    if not calibration_mode:
        calibration = json.loads((folder / "calibration.json").read_text())
        if not calibration.get("sealed") or not calibration["admitted"]: raise ValueError("Retired calibration must freeze first")
    if torch.cuda.device_count() != 1 or os.environ["CUDA_VISIBLE_DEVICES"] != os.environ["P2_GPU_UUID"]: raise ValueError("Single authorized GPU required")
    deadline = Deadline(time.monotonic() + float(os.environ["P2_MAX_JOB_SECONDS"]))
    for sig in [signal.SIGINT, signal.SIGTERM]: signal.signal(sig, lambda *_: setattr(deadline, "requested", True))
    torch.set_num_threads(1); torch.manual_seed(96); torch.cuda.manual_seed_all(96); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    spec = manifest["teacher"]; path = Path(spec["model_path"])
    receipt = json.loads((path / "download_receipt.json").read_text())
    if receipt["revision"] != spec["revision"] or receipt["repo"] != spec["repo"]: raise ValueError("Teacher source differs")
    processor = AutoProcessor.from_pretrained(path, local_files_only=True, use_fast=False, trust_remote_code=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True, trust_remote_code=False).to("cuda:0").eval().requires_grad_(False)
    versions = {n: p._version for n, p in model.named_parameters()}
    if {p.dtype for p in model.parameters() if p.is_floating_point()} != {torch.bfloat16}: raise ValueError("Actual BF16 teacher required")
    output = {"reward_scheme": SCHEME, "teacher_revision": spec["revision"], "manifest_sha256": digest, "labels_read": False,
              "stream": [], "probe": [], "masked_rgb": [128, 128, 128], "actual_dtype": "bfloat16", "calibration": calibration}
    sections = [("stream", manifest["calibration_entries"])] if calibration_mode else [(key, manifest[key]) for key in ["stream", "probe"]]
    with torch.inference_mode():
        for section, entries in sections:
            for index, entry in enumerate(entries):
                deadline.check(); row = load_input(folder, entry)
                with Image.open(row["image_paths"][0]) as image: real = image.convert("RGB")
                masked = Image.new("RGB", real.size, (128, 128, 128))
                record = {"id": row["id"], "option_keys": ["A", "B"], "orders": []}
                for reverse in [False, True]:
                    prompt, mapping = messages(row, reverse)
                    text = processor.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                    if text.count("<|image_pad|>") != 1: raise ValueError("Single image placeholder required")
                    order = {"reverse": reverse, "real": {}, "masked": {}, "answer_token_counts": {}}
                    for condition, image in [("real", real), ("masked", masked)]:
                        encoded = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")
                        if len(encoded["image_grid_thw"]) != 1: raise ValueError("Single image grid required")
                        length = encoded["input_ids"].shape[-1]
                        for alias, canonical in mapping.items():
                            deadline.check()
                            tokens = processor.tokenizer.encode(alias, add_special_tokens=False)
                            if not 1 <= len(tokens) <= 8: raise ValueError("Fixed answer-token length invalid")
                            ids = torch.tensor([tokens], device="cuda:0", dtype=torch.long)
                            inputs = dict(encoded, input_ids=torch.cat([encoded["input_ids"], ids], dim=1),
                                          attention_mask=torch.cat([encoded["attention_mask"], torch.ones_like(ids)], dim=1))
                            logits = model(**inputs, use_cache=False).logits[0, length - 1:length + len(tokens) - 1].float()
                            logp = logits.log_softmax(-1).gather(-1, ids[0, :, None]).mean().item()
                            if not torch.isfinite(torch.tensor(logp)): raise ValueError("Nonfinite answer score")
                            order[condition][canonical] = logp; order["answer_token_counts"][canonical] = len(tokens)
                    record["orders"].append(order)
                if calibration is not None:
                    record["target"], record["confidence"], record["same_canonical_choice"] = decision(record["orders"], calibration)
                output[section].append(record)
                atomic_json(reference / "progress.json", output)
                print(json.dumps({"stage": "calibration_scores" if calibration_mode else "fresh_visual_scores", "section": section, "cursor": index + 1}), flush=True)
    if versions != {n: p._version for n, p in model.named_parameters()} or any(p.grad is not None or p.requires_grad for p in model.parameters()): raise ValueError("Frozen teacher changed")
    output.update(sealed=True, weights_unchanged=True, parameter_version_check=True, code_commit=os.environ["P2_CODE_COMMIT"], full_weight_byte_comparison=False)
    atomic_json(destination, output)


if __name__ == "__main__": main()
