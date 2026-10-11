"""Actual unchanged training loss, gradients, updates and checkpoint recovery."""
import gc
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
from p22_contract import SEEDS, ARMS
from state import atomic_json, inference_without_state_change, save_checkpoint, load_checkpoint, rng_state, restore_rng


def validate_update(value, cfg):
    norms = value["gradient_norms"]
    if any(not math.isfinite(norms[k]) or norms[k] <= 0 for k in ["vision", "language"]) or not math.isfinite(value["max_behavior_recompute_error"]) or value["max_behavior_recompute_error"] > cfg["behavior_recompute_atol"] or not value["masters_persisted"] or not value["moments_persisted"]:
        raise ValueError("Actual finite vision/language gradients or native behavior guard failed")
    if any(v["changed_elements"] <= 0 for v in value["tracked_actor_updates"].values()):
        raise ValueError("Actual BF16 vision/language actor weights did not change")


def full_response_check(engine, encoded):
    with inference_without_state_change(engine.actor):
        cfg = engine.generation_config(True, True); cfg.max_new_tokens = engine.cfg["max_new_tokens"]
        generated = engine.actor.generate(**encoded, generation_config=cfg, return_dict_in_generate=True,
            output_scores=True, stopping_criteria=[engine.deadline])
        engine.deadline.check()
        response = generated.sequences[0, encoded["input_ids"].shape[-1]:]
        raw = torch.stack([score[0].float().log_softmax(-1)[token] for score, token in zip(generated.scores, response)])
        logits = forward_response(engine.actor, encoded, response)
        recomputed, _, _ = token_statistics(logits, response, chunk_tokens=engine.cfg["statistics_chunk_tokens"])
        error = float((raw - recomputed).abs().max())
        if not math.isfinite(error) or error > engine.cfg["probability_check_bf16_atol"]:
            raise ValueError("Full-response native training recomputation exceeds unchanged0.1 guard")
    return {"tokens": len(response), "max_logp_difference": error, "atol": engine.cfg["probability_check_bf16_atol"],
            "full_response_tested": True, "cached_frozen_check_substituted": False}


def recovery(engine, encoded, responses, advantages, folder, digest):
    payload = {**engine.state.pack(), "cursor": 1, "processed_ids": ["diagnostic-step"], "configuration": engine.cfg,
        "code_commit": os.environ["P2_CODE_COMMIT"], "model_revision": engine.cfg["model_revision"], "manifest_sha256": digest,
        "method": engine.method, "run_id": "accept-t", "rng": rng_state()}
    checkpoint = folder / "checkpoints"
    save_checkpoint(checkpoint, payload); del payload
    second = engine.update(encoded, responses, advantages, False); validate_update(second, engine.cfg)
    expected = {**engine.state.pack(), "cursor": 2, "processed_ids": ["diagnostic-step", "diagnostic-step2"], "configuration": engine.cfg,
        "code_commit": os.environ["P2_CODE_COMMIT"], "model_revision": engine.cfg["model_revision"], "manifest_sha256": digest,
        "method": engine.method, "run_id": "accept-t", "rng": rng_state()}
    save_checkpoint(checkpoint, expected); del expected
    latest = json.loads((checkpoint / "latest.json").read_text())
    first = torch.load(checkpoint / "c000001-u000001/state.pt", map_location="cpu", weights_only=False, mmap=True)
    engine.state.load(first); restore_rng(first["rng"]); del first
    resumed = engine.update(encoded, responses, advantages, False); validate_update(resumed, engine.cfg)
    expected = load_checkpoint(checkpoint, engine.cfg, digest, engine.method)
    current = engine.state.pack()
    if engine.state.updates != 2 or latest["optimizer_updates"] != 2: raise ValueError("Actual restored Adam counters differ")
    for name, tensor in current["master_weights"].items():
        if not torch.equal(tensor, expected["master_weights"][name]): raise ValueError("Actual resumed FP32 masters differ")
    for key, item in current["optimizer"]["state"].items():
        for name, tensor in item.items():
            if not torch.equal(tensor, expected["optimizer"]["state"][key][name]): raise ValueError("Actual resumed Adam moments differ")
    return {"passed": True, "same_process_two_vs_one_plus_one": True, "fresh_process_actual_recovery_tested_this_round": False,
            "checkpoint_source_config_checked": True, "synthetic_diagnostic_advantages_only": True}


def main():
    root = Path(os.environ["P0_ROOT"]); folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
    raw = (folder / "manifest.json").read_bytes(); manifest = json.loads(raw); digest = hashlib.sha256(raw).hexdigest()
    if manifest.get("p22_version") != 1 or digest != (folder / "manifest.sha256").read_text().strip() or manifest["rollout_seeds"] != SEEDS:
        raise ValueError("Actual acceptance frozen scope differs")
    gate = json.loads((folder / "scores/qualification.json").read_text()); lock = json.loads((folder / "main_budget_lock.json").read_text())
    if not gate["go"] or gate["primary_teacher"] != "q" or not lock["admitted"] or lock["complete_seeds"] != SEEDS or lock["complete_arms"] != ARMS:
        raise RuntimeError("Reward and entire-scope budget admission must precede GPU acceptance")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != os.environ["P2_GPU_UUID"] or torch.cuda.device_count() != 1: raise ValueError("Unauthorized CUDA visibility")
    deadline = Deadline(time.monotonic() + float(os.environ["P2_MAX_JOB_SECONDS"]))
    for signum in [signal.SIGTERM, signal.SIGINT]: signal.signal(signum, lambda *_: setattr(deadline, "requested", True))
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    result = {"passed": False, "labels_read": False, "diagnostic_weights_discarded_before_main": True, "arms": {}}
    try:
        for arm in ARMS:
            seed = SEEDS[0]; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            cfg = manifest["configurations"][arm][str(seed)]
            out = folder / "runs" / ("accept-" + arm); out.mkdir(exist_ok=False)
            engine = None
            try:
                engine = Engine(root, out, cfg, deadline)
                row = load_input(folder, manifest["stream"][0]); encoded = engine.encode(row)
                four = engine.probability_check(row)
                full = full_response_check(engine, encoded)
                responses = [engine.generate(encoded, True)[0] for _ in range(cfg["rollouts"])]
                # A fixed positive engineering signal avoids cancellation when sampled responses coincide.
                advantages = torch.ones(cfg["rollouts"])
                update = engine.update(encoded, responses, advantages, False); validate_update(update, cfg)
                restored = recovery(engine, encoded, responses, advantages, out, digest) if arm == "t" else None
                engine.reference_unchanged()
                result["arms"][arm] = {"passed": True, "four_token_probability": four, "full_response_probability": full,
                    "actual_update": update, "checkpoint_recovery": restored, "fixed_reference_unchanged": True,
                    "real_scientific_reward_used_for_engineering_step": False}
            finally:
                if engine: engine.close()
                del engine; gc.collect(); torch.cuda.empty_cache()
        result["passed"] = set(result["arms"]) == set(ARMS)
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        raise
    finally:
        atomic_json(folder / "acceptance.json", result)


if __name__ == "__main__": main()
