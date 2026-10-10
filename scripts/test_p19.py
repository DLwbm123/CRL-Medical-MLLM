"""Synthetic cache positions/vision handling, shared-token and complete closure checks."""
import copy
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from p19_cache import measure, validate_manifest
from p19_pipeline import aggregate
from report_p19 import export
from prepare_campaign import write_manifest


class Fake(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.positions = []; self.vision = []; self.draws = 0

    def prepare_inputs_for_generation(self, prefix, past_key_values, attention_mask, cache_position, use_cache, pixel_values, **_):
        assert use_cache and attention_mask.shape[-1] == prefix.shape[-1]
        self.positions.append(cache_position.tolist()); self.vision.append(pixel_values if cache_position[0] == 0 else None)
        return {"input_ids": prefix[:, cache_position], "past_key_values": past_key_values, "use_cache": use_cache}

    def forward(self, input_ids, past_key_values=None, use_cache=False, **_):
        logits = torch.nn.functional.one_hot(input_ids + 1, 10).float()
        return SimpleNamespace(logits=logits, past_key_values=(past_key_values or 0) + input_ids.shape[-1] if use_cache else None)

    def generate(self, input_ids, **_):
        self.draws += 1
        response = torch.tensor([[4, 5]])
        scores = [torch.nn.functional.one_hot(torch.tensor([i]), 10).float() for i in [4, 5]]
        return SimpleNamespace(sequences=torch.cat([input_ids, response], -1), scores=scores)


def main():
    actor = Fake(); encoded = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3, dtype=torch.long), "pixel_values": "synthetic"}
    engine = SimpleNamespace(actor=actor, cfg={"statistics_chunk_tokens": 4}, deadline=SimpleNamespace(check=lambda: None),
        encode=lambda _: encoded, generation_config=lambda *_: None)
    rows = measure(engine, {})
    assert actor.draws == 1 and [r["path"] for r in rows] == ["uncached", "cached"] and all(r["same_sampled_tokens"] and r["within_tolerance"] for r in rows)
    assert actor.positions == [[0, 1, 2], [3]] and actor.vision == ["synthetic", None]
    cfg = {"method": "Frozen SC-8", "model_dtype": "bfloat16", "evaluation_labels_allowed": False, "probability_check_bf16_atol": .1,
        "temperature": 1, "top_p": 1, "top_k": 0, "repetition_penalty": 1}
    configs = {f"numeric-k{s}": {**cfg, "seed": s} for s in [84, 85, 86]}
    manifest = {"p19_version": 1, "rollout_seeds": [84, 85, 86], "configurations": configs, "stream": [{}] * 32, "probe": [{}] * 32}
    validate_manifest(manifest)
    broken = copy.deepcopy(manifest); broken["configurations"]["numeric-k84"]["probability_check_bf16_atol"] = .2
    try: validate_manifest(broken)
    except ValueError: pass
    else: raise AssertionError("Relaxed diagnostic threshold accepted")
    with tempfile.TemporaryDirectory() as name:
        folder = Path(name); (folder / "controllers").mkdir(); (folder / "scores").mkdir()
        digest = write_manifest(folder / "manifest.json", manifest)
        (folder / "PIPELINE_STARTED.json").write_text(json.dumps({"code_commit": "synthetic"}))
        jobs = []
        for name, conf in configs.items():
            run = folder / "runs" / name; run.mkdir(parents=True)
            values = [{"section": section, "anonymous_index": i, "seed": conf["seed"], "path": path, "tokens": 2, "max_logp_difference": .14 if path == "uncached" else 0,
                "first_token_abs_difference": .14 if path == "uncached" else 0, "per_token_abs_difference": [.14, .14] if path == "uncached" else [0, 0],
                "within_tolerance": path == "cached", "same_sampled_tokens": True, "absolute_tolerance": .1} for section in ["calibration", "verification"] for i in range(32) for path in ["uncached", "cached"]]
            (run / "result.json").write_text(json.dumps({"status": "completed", "sealed": True, "cursor": 64, "measurements": values,
                "configuration": conf, "code_commit": "synthetic", "manifest_sha256": digest, "labels_read": False, "weights_unchanged": True, "optimizer_updates": 0}))
            jobs.append({"label": name, "return_code": 0, "wall_seconds": 1})
        (folder / "controllers/main-plan.json").write_text(json.dumps({"status": "all_planned_jobs_completed", "jobs": jobs}))
        closure = folder / "owned_cleanup_before_score.json"; closure.write_text(json.dumps({"owned_main_processes_ended": False}))
        try: aggregate(folder)
        except ValueError: pass
        else: raise AssertionError("Aggregation accepted live workers")
        closure.write_text(json.dumps({"owned_main_processes_ended": True})); summary = aggregate(folder)
        assert summary["diagnostic_passed"] and summary["cached_checks_passed"] == 192 and summary["measurements"] == 384 and not summary["training_performed"]
        (folder / "FINAL.json").write_text(json.dumps({"finished_utc": "2026-10-10T00:00:00+00:00", "status": "numerical_checks_passed", "code_commit": "synthetic",
            "gpu_process_seconds_used": 3, "cumulative_gpu_process_seconds": 83818.88884718131, "remaining_gpu_process_seconds": 175381.1111528187}))
        assert export(folder, folder / "public") == {"path_rows": 6, "measurement_rows": 384}
        (folder / "controllers/main-plan.json").write_text(json.dumps({"status": "stopped_after_job_failure", "jobs": []}))
        for run in (folder / "runs").iterdir(): (run / "result.json").unlink()
        (folder / "scores/numerical_summary.json").unlink()
        final = json.loads((folder / "FINAL.json").read_text()); final.update(status="failed", gpu_process_seconds_used=0); (folder / "FINAL.json").write_text(json.dumps(final))
        assert export(folder, folder / "public") == {"path_rows": 6, "measurement_rows": 384}
        assert "not_started" in (folder / "public/p19_path_results.csv").read_text()
    print(json.dumps({"same_draw_cache_position_vision_closure_and_full_NA_checks_passed": True}))


if __name__ == "__main__": main()
