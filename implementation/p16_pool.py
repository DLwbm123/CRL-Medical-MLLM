"""Fresh frozen candidate pools; never initializes an optimizer or reads labels."""
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


def main(engine_class=Engine):
    root = Path(os.environ["P0_ROOT"]); folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
    raw = (folder / "manifest.json").read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    if digest != (folder / "manifest.sha256").read_text().strip(): raise ValueError("Manifest changed")
    manifest = json.loads(raw); name = os.environ["P2_RUN"]; cfg = manifest["configurations"][name]
    seeds = manifest["rollout_seeds"]
    if len(seeds) != 3 or len(set(seeds)) != 3 or cfg["method"] != "Frozen SC-8" or cfg["evaluation_labels_allowed"] or cfg["seed"] not in seeds:
        raise ValueError("Only frozen label-free candidate collection is authorized")
    if torch.cuda.device_count() != 1 or os.environ["CUDA_VISIBLE_DEVICES"] != os.environ["P2_GPU_UUID"]:
        raise ValueError("Unauthorized GPU mapping")
    receipt = json.loads((root / "models/Qwen2.5-VL-3B-Instruct/download_receipt.json").read_text()) if (root / "models/Qwen2.5-VL-3B-Instruct/download_receipt.json").exists() else json.loads((Path(os.environ["P10_SOURCE_ROOT"]) / "metadata/model_download.json").read_text())
    if receipt["revision"] != cfg["model_revision"]: raise ValueError("Frozen actor revision differs")
    out = folder / "runs" / name; out.mkdir(exist_ok=False); (out / "cases").mkdir()
    seed = cfg["seed"]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    deadline = Deadline(time.monotonic() + float(os.environ["P2_MAX_JOB_SECONDS"]))
    for signum in [signal.SIGTERM, signal.SIGINT]: signal.signal(signum, lambda *_: setattr(deadline, "requested", True))
    engine = engine_class(root, out, cfg, deadline)
    try:
        if engine.training or engine.state is not None or any(p.requires_grad for p in engine.actor.parameters()):
            raise ValueError("Frozen pool unexpectedly has optimization state")
        versions = {n: p._version for n, p in engine.actor.named_parameters()}
        entries = manifest["stream"] + manifest["probe"]
        atomic_json(out / "probability_check.json", engine.probability_check(load_input(folder, entries[0])))
        for index, entry in enumerate(entries):
            row = engine.case(load_input(folder, entry), index, False)
            if row["optimizer_updates_after"] or row["optimizer_updates_before"]: raise ValueError("Unexpected update")
            atomic_json(out / "cases" / f"i{index:06d}.json", row)
            atomic_json(out / "progress.json", {"cursor": index + 1, "planned": 64, "optimizer_updates": 0})
            engine.event("case_complete", cursor=index + 1, optimizer_updates=0)
        if versions != {n: p._version for n, p in engine.actor.named_parameters()} or any(p.grad is not None for p in engine.actor.parameters()):
            raise ValueError("Frozen actor parameter versions or gradients changed")
        atomic_json(out / "result.json", {"status": "completed", "sealed": True, "cursor": 64,
            "manifest_sha256": digest, "code_commit": os.environ["P2_CODE_COMMIT"], "configuration": cfg,
            "optimizer_updates": 0, "labels_read": False, "weights_unchanged": True, "pool_only_not_training": True})
    finally: engine.close()


if __name__ == "__main__": main()
