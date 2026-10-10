"""Same generated tokens, paired complete-prefix and incremental-cache recomputation."""
import hashlib
import json
import math
import os
import random
import signal
import time
from pathlib import Path
import numpy as np
import torch
from continual import Deadline, Engine, forward_response, load_input
from core import token_statistics
from state import atomic_json, inference_without_state_change


def cached_response(model, encoded, response):
    """Use the installed model's generation preparation and explicit cache positions."""
    prefix = encoded["input_ids"]; mask = encoded["attention_mask"]
    vision = {k: v for k, v in encoded.items() if k not in {"input_ids", "attention_mask"}}
    cache = None; logits = []
    for index, token in enumerate(response):
        positions = torch.arange(prefix.shape[-1], device=prefix.device) if index == 0 else torch.tensor([prefix.shape[-1] - 1], device=prefix.device)
        inputs = model.prepare_inputs_for_generation(prefix, past_key_values=cache, attention_mask=mask,
            cache_position=positions, use_cache=True, **vision)
        output = model(**inputs, return_dict=True)
        logits.append(output.logits[0, -1]); cache = output.past_key_values
        if cache is None: raise RuntimeError("Incremental model did not return its cache")
        prefix = torch.cat([prefix, token.reshape(1, 1)], -1)
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], -1)
    return torch.stack(logits)


def measure(engine, row):
    engine.deadline.check()
    with inference_without_state_change(engine.actor):
        encoded = engine.encode(row)
        output = engine.actor.generate(**encoded, generation_config=engine.generation_config(True, True),
            return_dict_in_generate=True, output_scores=True, stopping_criteria=[engine.deadline])
        response = output.sequences[0, encoded["input_ids"].shape[-1]:]
        generated = torch.stack([score[0].float().log_softmax(-1)[token] for score, token in zip(output.scores, response)])
        rows = []
        for name, logits in [("uncached", forward_response(engine.actor, encoded, response)), ("cached", cached_response(engine.actor, encoded, response))]:
            engine.deadline.check()
            logp, _, _ = token_statistics(logits, response, chunk_tokens=engine.cfg["statistics_chunk_tokens"])
            errors = (generated - logp).abs().tolist()
            if not errors or any(not math.isfinite(x) for x in errors): raise RuntimeError("Non-finite or empty numerical comparison")
            rows.append({"path": name, "tokens": len(response), "max_logp_difference": max(errors),
                "first_token_abs_difference": errors[0], "per_token_abs_difference": errors,
                "within_tolerance": max(errors) <= .1, "absolute_tolerance": .1, "same_sampled_tokens": True})
        return rows


def validate_manifest(manifest):
    if manifest.get("p19_version") != 1 or manifest["rollout_seeds"] != [84, 85, 86] or set(manifest["configurations"]) != {f"numeric-k{s}" for s in [84, 85, 86]}:
        raise ValueError("Complete same-token matrix differs")
    if len(manifest["stream"]) != 32 or len(manifest["probe"]) != 32: raise ValueError("Incomplete input scope")
    for seed in [84, 85, 86]:
        cfg = manifest["configurations"][f"numeric-k{seed}"]
        if cfg["seed"] != seed or cfg["method"] != "Frozen SC-8" or cfg["model_dtype"] != "bfloat16" or cfg["evaluation_labels_allowed"] or cfg["probability_check_bf16_atol"] != .1 or (cfg["temperature"], cfg["top_p"], cfg["top_k"], cfg["repetition_penalty"]) != (1, 1, 0, 1):
            raise ValueError("Fixed precision, unwarped sampler or original guard differs")


def main():
    root = Path(os.environ["P0_ROOT"]); folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
    raw = (folder / "manifest.json").read_bytes(); digest = hashlib.sha256(raw).hexdigest()
    if digest != (folder / "manifest.sha256").read_text().strip(): raise ValueError("Frozen scope changed")
    manifest = json.loads(raw); validate_manifest(manifest); cfg = manifest["configurations"][os.environ["P2_RUN"]]
    if torch.cuda.device_count() != 1 or os.environ["CUDA_VISIBLE_DEVICES"] != os.environ["P2_GPU_UUID"]: raise ValueError("Unauthorized GPU mapping")
    if json.loads((root / "metadata/model_download.json").read_text())["revision"] != cfg["model_revision"]: raise ValueError("Actor revision differs")
    out = folder / "runs" / os.environ["P2_RUN"]; out.mkdir(exist_ok=False)
    deadline = Deadline(time.monotonic() + float(os.environ["P2_MAX_JOB_SECONDS"]))
    for sig in [signal.SIGTERM, signal.SIGINT]: signal.signal(sig, lambda *_: setattr(deadline, "requested", True))
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    engine = Engine(root, out, cfg, deadline)
    try:
        if engine.training or engine.state is not None or any(p.requires_grad or p.dtype != torch.bfloat16 for p in engine.actor.parameters()): raise ValueError("Only unchanged BF16 frozen inference is allowed")
        versions = {name: p._version for name, p in engine.actor.named_parameters()}; rows = []
        for index, entry in enumerate(manifest["stream"] + manifest["probe"]):
            seed = cfg["seed"] + index * 1000
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            for metric in measure(engine, load_input(folder, entry)):
                rows.append({"section": "calibration" if index < 32 else "verification", "anonymous_index": index % 32, "seed": cfg["seed"], **metric})
            atomic_json(out / "progress.json", {"cursor": index + 1, "planned": 64})
            engine.event("numeric_case", cursor=index + 1)
        if any(p._version != versions[name] for name, p in engine.actor.named_parameters()): raise RuntimeError("Frozen weights changed")
        atomic_json(out / "result.json", {"status": "completed", "sealed": True, "cursor": 64, "measurements": rows,
            "configuration": cfg, "manifest_sha256": digest, "code_commit": os.environ["P2_CODE_COMMIT"],
            "weights_unchanged": True, "optimizer_updates": 0, "labels_read": False})
    finally: engine.close()


if __name__ == "__main__": main()
