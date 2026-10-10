"""One frozen 7B visual teacher, two fixed option orders, no optimizer."""
import hashlib
import json
import os
import random
import signal
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, GenerationConfig, Qwen2_5_VLForConditionalGeneration
from continual import Deadline, load_input
from state import atomic_json
from verified_reward import agreed_target, judge_messages, parse_judgment


def main():
    root = Path(os.environ["P0_ROOT"])
    folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
    raw = (folder / "manifest.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != (folder / "manifest.sha256").read_text().strip():
        raise ValueError("Locked manifest changed")
    manifest = json.loads(raw)
    spec = manifest["teacher"]
    if torch.cuda.device_count() != 1 or os.environ["CUDA_VISIBLE_DEVICES"] != os.environ["P2_GPU_UUID"]:
        raise ValueError("Verifier must see exactly the authorized UUID")
    deadline = Deadline(time.monotonic() + float(os.environ["P2_MAX_JOB_SECONDS"]))
    for signum in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(signum, lambda *_: setattr(deadline, "requested", True))
    seed = manifest["rollout_seeds"][0]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    path = Path(spec["model_path"])
    if json.loads((path / "download_receipt.json").read_text())["revision"] != spec["revision"]:
        raise ValueError("Teacher download revision differs")
    processor = AutoProcessor.from_pretrained(path, local_files_only=True, use_fast=False, trust_remote_code=False)
    actor = Qwen2_5_VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", local_files_only=True, trust_remote_code=False).to("cuda:0").eval().requires_grad_(False)
    versions = {n: p._version for n, p in actor.named_parameters()}
    generation = GenerationConfig(bos_token_id=actor.config.bos_token_id, pad_token_id=actor.generation_config.pad_token_id,
        eos_token_id=actor.generation_config.eos_token_id, do_sample=False, max_new_tokens=spec["max_new_tokens"], use_cache=True)
    output = {"teacher_revision": spec["revision"], "manifest_sha256": digest, "labels_read": False, "stream": [], "probe": []}
    destination = folder / "reference/targets.json"
    if destination.exists():
        raise RuntimeError("Frozen target worker cannot replace an existing signal")
    with torch.inference_mode():
        for section in ["stream", "probe"]:
            for index, entry in enumerate(manifest[section]):
                row = load_input(folder, entry)
                record = {"id": row["id"], "option_keys": sorted(row["options"]), "judgments": [], "readouts": []}
                for reverse in [False, True]:
                    deadline.check()
                    messages, mapping = judge_messages(row, reverse)
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    if text.count("<|image_pad|>") != len(row["images"]):
                        raise ValueError("Verifier image placeholder count differs")
                    images = []
                    for p in row["image_paths"]:
                        with Image.open(p) as image: images.append(image.convert("RGB"))
                    encoded = processor(text=[text], images=images, return_tensors="pt").to("cuda:0")
                    if len(encoded["image_grid_thw"]) != len(images):
                        raise ValueError("Verifier image-grid count differs")
                    tokens = actor.generate(**encoded, generation_config=generation, stopping_criteria=[deadline])[0, encoded["input_ids"].shape[-1]:]
                    deadline.check()
                    answer = processor.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    record["judgments"].append(parse_judgment(answer, mapping))
                    record["readouts"].append({"reverse": reverse, "text": answer, "tokens": len(tokens), "hit_length_cap": len(tokens) == spec["max_new_tokens"]})
                record["target"] = agreed_target(record["judgments"])
                output[section].append(record)
                atomic_json(folder / "reference/progress.json", output)
                print(json.dumps({"stage": "verifier_case", "section": section, "cursor": index + 1, "accepted": record["target"] is not None}), flush=True)
    if versions != {n: p._version for n, p in actor.named_parameters()} or any(p.requires_grad or p.grad is not None for p in actor.parameters()):
        raise ValueError("Frozen teacher parameter versions or gradient state changed")
    output.update(weights_unchanged=True, parameter_version_check=True, full_weight_byte_comparison=False, sealed=True,
                  code_commit=os.environ["P2_CODE_COMMIT"], model_family_independence=False, distinct_size_checkpoint=True)
    atomic_json(destination, output)
    atomic_json(folder / "reference/completion.json", {"sealed": True, "stream": len(output["stream"]), "probe": len(output["probe"]), "labels_read": False})


if __name__ == "__main__":
    main()
