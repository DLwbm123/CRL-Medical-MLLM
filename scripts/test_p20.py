"""Cached-check eligibility, unchanged abort guard and complete missing exports."""
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from p20_pool import CachedCheckEngine, validate_manifest
from report_p20 import export
from test_p19 import Fake


def main():
    cfg = {"method": "Frozen SC-8", "evaluation_labels_allowed": False, "probability_check_bf16_atol": .1,
        "statistics_chunk_tokens": 4, "model_dtype": "bfloat16"}
    engine = CachedCheckEngine.__new__(CachedCheckEngine); engine.training = False; engine.state = None; engine.method = "Frozen SC-8"
    engine.actor = Fake(); engine.cfg = cfg; engine.deadline = SimpleNamespace(check=lambda: None)
    engine.encode = lambda _: {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long), "pixel_values": "synthetic"}
    engine.generation_config = lambda *_: None
    result = engine.probability_check({})
    assert result["max_logp_difference"] == 0 and result["within_tolerance"] and engine.actor.draws == 1 and not result["training_loss_validated"]
    engine.training = True
    try: engine.probability_check({})
    except ValueError: pass
    else: raise AssertionError("Cached diagnostic admitted a training engine")
    engine.training = False
    failed = {"max_logp_difference": .12, "within_tolerance": False}
    with patch("p20_pool.measure", return_value=[failed, failed]):
        try: engine.probability_check({})
        except RuntimeError: pass
        else: raise AssertionError("Original0.1 failure was ignored")
    configs = {f"pool-{arm}{seed}": {**cfg, "seed": seed, "temperature": temperature, "top_p": top_p}
        for seed in [87, 88, 89] for arm, temperature, top_p in [("s", .7, .95), ("t", 1., 1.)]}
    manifest = {"p20_version": 1, "rollout_seeds": [87, 88, 89], "frozen_pool_probability_recompute_path": "incremental_cache",
        "training_authorized_in_this_round": False, "stream": [{}] * 32, "probe": [{}] * 32, "configurations": configs,
        "gate": {"minimum_accepted": 16, "minimum_correct": 12, "minimum_precision": .75, "wrong_positive_drop_pp": 10}}
    validate_manifest(manifest); wrong = copy.deepcopy(manifest); wrong["configurations"]["pool-s87"]["probability_check_bf16_atol"] = .2
    try: validate_manifest(wrong)
    except ValueError: pass
    else: raise AssertionError("Relaxed check admitted")
    with tempfile.TemporaryDirectory() as name:
        folder = Path(name); (folder / "FINAL.json").write_text(json.dumps({"finished_utc": "2026-10-10T00:00:00+00:00", "status": "failed", "code_commit": "synthetic",
            "gpu_process_seconds_used": 0, "cumulative_gpu_process_seconds": 84021.01047639032, "remaining_gpu_process_seconds": 175178.9895236097}))
        assert export(folder, folder / "public") == {"teacher_rows": 64, "pool_rows": 6, "case_rows": 384, "check_rows": 24, "probability_check_rows": 6}
        assert '"training_loss_validated": false' in (folder / "public/p20_results.json").read_text()
    print(json.dumps({"cached_frozen_only_original_abort_complete_scope_and_NA_checks_passed": True}))


if __name__ == "__main__": main()
