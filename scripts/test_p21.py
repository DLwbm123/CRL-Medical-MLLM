"""Paired source provenance, missing-control label barrier and fixed-primary tests."""
import copy
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from p21_reward import SOURCES, SEEDS, validate, main
from prepare_campaign import write_manifest
from continual import group_rewards
from report_p21 import export


def main_test():
    previous = dict(os.environ)
    try:
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); folder = root / "outputs/fixture"
            for part in ["inputs", "reference/q", "reference/m", "scores/q", "scores/m", "controllers"]: (folder / part).mkdir(parents=True)
            teachers = {k: {"repo": repo, "revision": revision, "model_path": "/synthetic", "evidence_first": True, "json_suffix": True,
                "max_new_tokens": 1024, "option_orders": ["sorted", "reversed"], "gpu_index": 1} for k, (repo, revision) in SOURCES.items()}
            entries = []
            for i in range(64):
                row = {"id": str(i), "source_id": i, "dataset": "fixture", "split": "train", "question": "Synthetic?",
                    "options": {"A": "yes", "B": "no"}, "images": [str(i)], "image_paths": ["/synthetic"], "language": "en"}
                rel = f"inputs/i{i:03d}.json"; (folder / rel).write_text(json.dumps(row)); entries.append({"id": str(i), "input": rel})
            configs = {f"pool-{arm}{seed}": {"seed": seed, "method": "Frozen SC-8", "evaluation_labels_allowed": False,
                "model_dtype": "bfloat16", "probability_check_bf16_atol": .1, "temperature": temperature, "top_p": top_p}
                for seed in SEEDS for arm, temperature, top_p in [("s", .7, .95), ("t", 1., 1.)]}
            manifest = {"p21_version": 1, "rollout_seeds": SEEDS, "stream": entries[:32], "probe": entries[32:], "teachers": teachers,
                "teacher": teachers["q"], "primary_teacher": "q", "configurations": configs, "training_authorized_in_this_round": False,
                "gate": {"minimum_accepted": 16, "minimum_correct": 12, "minimum_precision": .75, "wrong_positive_drop_pp": 10}}
            validate(manifest)
            for mutate in [lambda m: m.update(primary_teacher="m"), lambda m: m["teachers"]["m"].update(revision="other"),
                           lambda m: m["teachers"]["q"].update(max_new_tokens=512), lambda m: m["gate"].update(minimum_precision=.5)]:
                bad = copy.deepcopy(manifest); mutate(bad)
                try: validate(bad)
                except ValueError: pass
                else: raise AssertionError("Changed factor or relaxed gate admitted")
            digest = write_manifest(folder / "manifest.json", manifest)
            records = [{"id": str(i), "target": "B", "option_keys": ["A", "B"], "judgments": [{"choice": "B", "visual_support": "supported", "clinical_consistency": "consistent"}] * 2,
                "readouts": [{"tokens": 5, "hit_length_cap": False}] * 2} for i in range(64)]
            for key in SOURCES:
                values = copy.deepcopy(records)
                if key == "q":
                    for r in values:
                        r["target"] = None
                        for j in r["judgments"]: j["visual_support"] = "uncertain"
                signal = {"stream": values[:32], "probe": values[32:], "labels_read": False, "weights_unchanged": True,
                    "sealed": True, "manifest_sha256": digest, "teacher_revision": teachers[key]["revision"], "code_commit": "synthetic"}
                (folder / "reference" / key / "targets.json").write_text(json.dumps(signal))
            for part in ["qualifier_owned_ended.json", "owned_cleanup_before_score.json"]: (folder / part).write_text(json.dumps({"owned_main_processes_ended": True}))
            ledger = {"status": "all_planned_jobs_completed", "jobs": [{"label": n, "return_code": 0, "wall_seconds": 0} for n in configs]}
            (folder / "controllers/main-plan.json").write_text(json.dumps(ledger))
            completions = ["Final answer: A"] * 6 + ["Final answer: B"] * 2
            rewards, _, vote = group_rewards(completions, {"A": "yes", "B": "no"})
            for run_name, cfg in configs.items():
                run = folder / "runs" / run_name; (run / "cases").mkdir(parents=True)
                (run / "result.json").write_text(json.dumps({"status": "completed", "sealed": True, "cursor": 64, "manifest_sha256": digest,
                    "configuration": cfg, "code_commit": "synthetic", "optimizer_updates": 0, "labels_read": False, "weights_unchanged": True}))
                (run / "probability_check.json").write_text(json.dumps({"max_logp_difference": 0, "temperature": 1, "top_p": 1,
                    "within_tolerance": True, "same_sampled_tokens": True, "paired_uncached_control": {"max_logp_difference": .02}}))
                for i in range(64):
                    (run / "cases" / f"i{i:06d}.json").write_text(json.dumps({"id": str(i), "completions": completions, "vote": vote, "rewards": rewards.tolist(),
                        "optimizer_updates_before": 0, "optimizer_updates_after": 0, "hit_length_cap": [False] * 8}))
            os.environ.update(P0_ROOT=str(root), P10_SOURCE_ROOT=str(root), P2_CAMPAIGN="fixture", P2_CODE_COMMIT="synthetic", P2_MODE="p21_score")
            control = folder / "reference/m/targets.json"; saved = control.read_text(); control.unlink()
            with patch("p16_score.main", side_effect=AssertionError("Labels were accessed before the control sealed")):
                try: main()
                except FileNotFoundError: pass
                else: raise AssertionError("Missing control did not stop scoring")
            control.write_text(saved)
            labels = root / "views/evaluation_labels/SLAKE/train.jsonl"; labels.parent.mkdir(parents=True)
            labels.write_text("".join(json.dumps({"id": str(i), "answer": "no"}) + "\n" for i in range(64)))
            main(); result = json.loads((folder / "scores/qualification.json").read_text())
            assert result["go"] is False and result["teachers"]["m"]["go"] is True and not result["primary_selected_after_scoring"]
            (folder / "FINAL.json").write_text(json.dumps({"finished_utc": "2026-10-10T00:00:00+00:00", "status": "reward_qualification_negative", "code_commit": "synthetic",
                "gpu_process_seconds_used": 0, "cumulative_gpu_process_seconds": 96825.18736666531, "remaining_gpu_process_seconds": 162374.81263333472}))
            counts = export(folder, root / "public")
            assert (counts["teacher_rows"], counts["paired_case_rows"], counts["check_rows"]) == (128, 768, 48)
            assert json.loads((root / "public/p21_results.json").read_text())["candidate_qualified"] is False
            for key in SOURCES:
                for part in ["qualification.json", "anonymous_teacher_cases.json", "anonymous_pool_cases.json"]: (folder / "scores" / key / part).unlink()
            export(folder, root / "missing")
            assert json.loads((root / "missing/p21_results.json").read_text())["candidate_qualified"] is None
    finally:
        os.environ.clear(); os.environ.update(previous)
    print(json.dumps({"paired_source_provenance_label_barrier_fixed_primary_full_NA_passed": True}))


if __name__ == "__main__": main_test()
