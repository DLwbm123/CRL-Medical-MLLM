"""Fixed original-base option likelihood signal; no labels or optimizer."""
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from continual import Deadline, load_input
from p3_readout import messages, option_tokens, option_log_probabilities, ASSISTANT_PREFIX, INSTRUCTION
from state import atomic_json, inference_without_state_change


def main():
    started = time.monotonic()
    root = Path(os.environ["P0_ROOT"])
    folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
    raw = (folder / "manifest.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != (folder / "manifest.sha256").read_text().strip():
        raise ValueError("Manifest changed before reference scoring")
    manifest = json.loads(raw)
    cfg = json.loads((root / "configs/p2_a.json").read_text())
    if os.environ.get("CUDA_VISIBLE_DEVICES") != os.environ["P2_GPU_UUID"] or torch.cuda.device_count() != 1:
        raise RuntimeError("Reference scoring requires the single authorized GPU UUID")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    remaining = (datetime.fromisoformat(json.loads((root / "metadata/campaign_budget.json").read_text())["gpu_stop_utc"]) - datetime.now(timezone.utc)).total_seconds()
    deadline = Deadline(time.monotonic() + min(remaining, float(os.environ.get("P2_MAX_JOB_SECONDS", "900"))))
    model_path = root / cfg["model_path"]
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, use_fast=False, trust_remote_code=False)
    actor = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True, trust_remote_code=False).to("cuda:0").eval().requires_grad_(False)
    versions = [p._version for p in actor.parameters()]
    records = {"stream": [], "probe": []}
    with inference_without_state_change(actor), torch.no_grad():
        for section in records:
            for entry in manifest[section]:
                deadline.check()
                row = load_input(folder, entry)
                text = processor.apply_chat_template(messages(row), tokenize=False, add_generation_prompt=True) + ASSISTANT_PREFIX
                candidates = option_tokens(processor.tokenizer, text, row["options"])
                images = []
                for path in row["image_paths"]:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                if text.count("<|image_pad|>") != len(images):
                    raise ValueError("Reference image placeholders differ")
                encoded = processor(text=[text], images=images, return_tensors="pt").to("cuda:0")
                scores = option_log_probabilities(actor, encoded, candidates)
                winner = min(scores, key=lambda k: (-scores[k], k))
                records[section].append({"id": row["id"], "winner": winner, "scores": scores,
                                         "option_token_ids": candidates})
                print(json.dumps({"stage": "reference_group_complete", "section": section, "completed": len(records[section])}), flush=True)
                del encoded
    if versions != [p._version for p in actor.parameters()]:
        raise RuntimeError("Reference parameters changed")
    provenance = {"manifest_sha256": digest, "model_revision": cfg["model_revision"],
                  "code_commit": os.environ["P2_CODE_COMMIT"], "labels_read": False,
                  "optimizer_created": False, "parameters_unchanged": True, "rng_and_modes_preserved": True,
                  "protocol": {"instruction": INSTRUCTION, "assistant_prefix": ASSISTANT_PREFIX,
                               "score": "complete identifier token conditional log-probability sum",
                               "eos_scored": False, "length_normalized": False,
                               "tie_rule": "lexicographically first maximum",
                               "same_backbone_as_actor": True, "independent_external_verifier": False}}
    destination = folder / "reference"
    destination.mkdir(exist_ok=True)
    for section, rows in records.items():
        path = destination / (section + "_targets.json")
        if path.exists():
            raise ValueError("Refusing to replace closed reference predictions")
        atomic_json(path, {**provenance, "predictions": rows})
    atomic_json(destination / "completion.json", {**provenance, "closed_utc": datetime.now(timezone.utc).isoformat(),
                "coverage": {k: len(v) for k, v in records.items()}, "elapsed_seconds": time.monotonic() - started,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "cpu_peak_rss_kib": next(int(line.split()[1]) for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("VmHWM:"))})
    print(json.dumps({"reference_signal_closed": True, "labels_read": False, "stream": 16, "probe": 16}), flush=True)


if __name__ == "__main__":
    main()
