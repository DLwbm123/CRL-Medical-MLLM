"""Complete paired precision diagnostics; no reward, labels, optimizer or update."""
import hashlib
import json
import os
import random
import signal
import time
from pathlib import Path
import numpy as np
import torch
from continual import Deadline, Engine, load_input
from state import atomic_json


def validate_manifest(manifest):
    expected = {f"numeric-{arm}{seed}" for seed in [81, 82, 83] for arm in ["b", "f"]}
    if manifest.get("p18_version") != 1 or set(manifest["configurations"]) != expected or manifest["rollout_seeds"] != [81, 82, 83]:
        raise ValueError("The complete frozen precision matrix differs")
    if len(manifest["stream"]) != 32 or len(manifest["probe"]) != 32:
        raise ValueError("The complete frozen input scope differs")
    for seed in [81, 82, 83]:
        b = manifest["configurations"][f"numeric-b{seed}"]; f = manifest["configurations"][f"numeric-f{seed}"]
        if b["model_dtype"] != "bfloat16" or f["model_dtype"] != "float32" or {k: v for k, v in b.items() if k != "model_dtype"} != {k: v for k, v in f.items() if k != "model_dtype"}:
            raise ValueError("Paired configurations change more than precision")
        if b["method"] != "Frozen SC-8" or b["seed"] != seed or b["evaluation_labels_allowed"] or b["probability_check_bf16_atol"] != .1:
            raise ValueError("The diagnostic would change the frozen check or permit training")
        if (b["temperature"], b["top_p"], b["top_k"], b["repetition_penalty"]) != (1, 1, 0, 1):
            raise ValueError("Diagnostic generation must remain unwarped")


def main():
    root = Path(os.environ["P0_ROOT"]); folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
    raw = (folder / "manifest.json").read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    if digest != (folder / "manifest.sha256").read_text().strip(): raise ValueError("Frozen manifest changed")
    manifest = json.loads(raw); validate_manifest(manifest)
    cfg = manifest["configurations"][os.environ["P2_RUN"]]
    if torch.cuda.device_count() != 1 or os.environ["CUDA_VISIBLE_DEVICES"] != os.environ["P2_GPU_UUID"]:
        raise ValueError("Unauthorized GPU mapping")
    if json.loads((root / "metadata/model_download.json").read_text())["revision"] != cfg["model_revision"]:
        raise ValueError("Actor revision differs")
    out = folder / "runs" / os.environ["P2_RUN"]; out.mkdir(exist_ok=False)
    deadline = Deadline(time.monotonic() + float(os.environ["P2_MAX_JOB_SECONDS"]))
    for sig in [signal.SIGTERM, signal.SIGINT]: signal.signal(sig, lambda *_: setattr(deadline, "requested", True))
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    engine = Engine(root, out, cfg, deadline)
    try:
        if engine.training or engine.state is not None or any(p.requires_grad for p in engine.actor.parameters()):
            raise ValueError("Only frozen inference is authorized")
        expected_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[cfg["model_dtype"]]
        if any(p.is_floating_point() and p.dtype != expected_dtype for p in engine.actor.parameters()):
            raise ValueError("Actual frozen model precision differs")
        versions = {name: p._version for name, p in engine.actor.named_parameters()}
        rows = []
        for index, entry in enumerate(manifest["stream"] + manifest["probe"]):
            # Both precisions receive the same per-case RNG state; no seed outcome selects scope.
            seed = cfg["seed"] + index * 1000
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            metric = engine.probability_measurement(load_input(folder, entry))
            rows.append({"section": "calibration" if index < 32 else "verification", "anonymous_index": index % 32,
                         "seed": cfg["seed"], "precision": cfg["model_dtype"], **metric})
            atomic_json(out / "progress.json", {"cursor": len(rows), "planned": 64})
            engine.event("numeric_case", cursor=len(rows), passed=metric["within_tolerance"])
        unchanged = all(p._version == versions[name] for name, p in engine.actor.named_parameters())
        if not unchanged: raise RuntimeError("Frozen actor weights changed")
        atomic_json(out / "result.json", {"status": "completed", "sealed": True, "cursor": len(rows), "measurements": rows,
            "configuration": cfg, "manifest_sha256": digest, "code_commit": os.environ["P2_CODE_COMMIT"],
            "weights_unchanged": True, "optimizer_updates": 0, "labels_read": False})
    finally:
        engine.close()


if __name__ == "__main__": main()
