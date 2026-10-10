"""Small boundary checks for new selection, gates and complete failure exports."""
import json
import os
import tempfile
from pathlib import Path
from prepare_p16 import select_rows, heldout_image_references, binary_domain_metadata, validate_binary_domain
from p16_score import block_gate, main as score
from continual import group_rewards
from prepare_campaign import write_manifest
from report_p16 import export


def main():
    rows = [{"qid": i, "img_name": str(i // 2), "q_lang": "en", "answer_type": "CLOSED", "question": "Is this visible?", "modality": "CT", "content_type": "Organ"} for i in range(160)]
    first, count = select_rows(rows, {"0"})
    assert count == 79 and len({r["img_name"] for r in first}) == 79 and all(r["img_name"] != "0" for r in first)
    assert first == select_rows(list(reversed(rows)), {"0"})[0]
    annotations = [{**r, "answer": " YES " if r["qid"] % 2 else "neither"} for r in rows]
    domain = binary_domain_metadata(annotations)
    assert len(domain) == 80 and all("answer" not in r for r in domain)
    assert domain == binary_domain_metadata([{**r, "answer": "no" if r["answer"] == " YES " else r["answer"]} for r in annotations])
    try: select_rows([{**r, "answer": "secret"} for r in rows], set())
    except ValueError: pass
    else: raise AssertionError("Selector accepted labels")
    with tempfile.TemporaryDirectory() as name:
        root = Path(name); refs = root / "identities.jsonl"
        refs.write_text(json.dumps({"images": ["overlap"], "question": "not returned", "answer": "not returned"}) + "\n")
        assert heldout_image_references(refs) == {"overlap"}
        refs.write_text(json.dumps({"id": "chosen", "answer": "yes"}) + "\n")
        assert validate_binary_domain(refs, {"chosen"})["groups"] == 1
        refs.write_text(json.dumps({"id": "chosen", "answer": "neither"}) + "\n")
        try: validate_binary_domain(refs, {"chosen"})
        except ValueError: pass
        else: raise AssertionError("Domain admission accepted a nonbinary scoring view")
        m = {"correct_negative": {"percent": 30}, "wrong_positive": {"percent": 40}}
        v = {"correct_negative": {"percent": 20}, "wrong_positive": {"percent": 30}, "counts": {"minority_rescued": 1}}
        assert block_gate(m, v, 12, 16)["go"]
        assert not block_gate(m, v, 11, 16)["go"] and not block_gate(m, v, 12, 17)["go"]
        v["correct_negative"]["percent"] = None
        assert not block_gate(m, v, 12, 16)["go"]
        folder = root / "closed"; folder.mkdir()
        (folder / "FINAL.json").write_text(json.dumps({"finished_utc": "2026-10-10T00:00:00+00:00", "status": "failed", "code_commit": "synthetic",
            "gpu_process_seconds_used": 0, "cumulative_gpu_process_seconds": 68746.33746506431, "remaining_gpu_process_seconds": 190453.6625349357}))
        counts = export(folder, root / "public")
        assert counts == {"teacher_rows": 64, "pool_rows": 6, "case_rows": 384, "check_rows": 24}
        assert not any(word in p.read_text() for p in (root / "public").iterdir() for word in ["secret", "image_paths", "SLAKE/train/"])
        (folder / "offline_failure_audit.json").write_text(json.dumps({"failure_type": "input_label_contract_failure", "nonbinary_label_count": 1, "selected_groups": 64}))
        export(folder, root / "public")
        failure = json.loads((root / "public/p16_results.json").read_text())
        assert failure["failure_type"] == "input_label_contract_failure" and failure["qualification"] is None
        export(folder, root / "new_public", "p17", (78, 79, 80))
        import csv
        with (root / "new_public/p17_all_pool_cases.csv").open() as stream:
            assert {int(r["seed"]) for r in csv.DictReader(stream)} == {78, 79, 80}
        (folder / "offline_failure_audit.json").unlink()
        (folder / "engineering_failure_audit.json").write_text(json.dumps({"failure_type": "probability_check_failure", "teacher_readouts": 128,
            "max_logp_difference": .13839125633239746, "absolute_tolerance": .1}))
        export(folder, root / "new_public", "p17", (78, 79, 80))
        engineering = json.loads((root / "new_public/p17_results.json").read_text())
        assert engineering["failure_type"] == "probability_check_failure" and engineering["qualification"] is None
        assert "Five remaining pools never started" in (root / "new_public/p17_report.md").read_text()
        folder = root / "outputs" / "fixture"; folder.mkdir(parents=True)
        for part in ["inputs", "reference", "scores", "controllers"]: (folder / part).mkdir()
        entries = []
        for i in range(64):
            row = {"id": str(i), "source_id": i, "dataset": "fixture", "split": "train", "question": "Synthetic?",
                   "options": {"A": "yes", "B": "no"}, "images": [str(i)], "image_paths": ["/synthetic"], "language": "en"}
            relative = f"inputs/i{i:03d}.json"; (folder / relative).write_text(json.dumps(row))
            entries.append({"id": str(i), "input": relative})
        configs = {f"pool-{arm}{seed}": {"seed": seed, "probability_check_bf16_atol": .1} for seed in [75, 76, 77] for arm in ["s", "t"]}
        manifest = {"stream": entries[:32], "probe": entries[32:], "teacher": {"revision": "synthetic"}, "rollout_seeds": [75, 76, 77], "configurations": configs}
        digest = write_manifest(folder / "manifest.json", manifest)
        teacher = [{"id": str(i), "target": "B", "option_keys": ["A", "B"],
            "judgments": [{"choice": "B", "visual_support": "supported", "clinical_consistency": "consistent"}] * 2,
            "readouts": [{"tokens": 5, "hit_length_cap": False}] * 2} for i in range(64)]
        (folder / "reference/targets.json").write_text(json.dumps({"stream": teacher[:32], "probe": teacher[32:], "labels_read": False,
            "weights_unchanged": True, "sealed": True, "manifest_sha256": digest, "teacher_revision": "synthetic", "code_commit": "synthetic"}))
        for name in ["qualifier_owned_ended.json", "owned_cleanup_before_score.json"]:
            (folder / name).write_text(json.dumps({"owned_main_processes_ended": True}))
        (folder / "controllers/main-plan.json").write_text(json.dumps({"status": "all_planned_jobs_completed", "jobs": [{"return_code": 0}] * 6}))
        completions = ["Final answer: A"] * 6 + ["Final answer: B"] * 2
        rewards, _, vote = group_rewards(completions, {"A": "yes", "B": "no"})
        for name, cfg in configs.items():
            run = folder / "runs" / name; (run / "cases").mkdir(parents=True)
            (run / "result.json").write_text(json.dumps({"status": "completed", "sealed": True, "cursor": 64, "manifest_sha256": digest,
                "configuration": cfg, "code_commit": "synthetic", "optimizer_updates": 0, "labels_read": False, "weights_unchanged": True}))
            (run / "probability_check.json").write_text(json.dumps({"max_logp_difference": 0, "temperature": 1, "top_p": 1}))
            for i in range(64):
                (run / "cases" / f"i{i:06d}.json").write_text(json.dumps({"id": str(i), "completions": completions,
                    "vote": vote, "rewards": rewards.tolist(), "optimizer_updates_before": 0, "optimizer_updates_after": 0, "hit_length_cap": [False] * 8}))
        previous = dict(os.environ)
        try:
            os.environ.update(P0_ROOT=str(root), P10_SOURCE_ROOT=str(root), P2_CAMPAIGN="fixture", P2_CODE_COMMIT="synthetic")
            # Closed-worker guard must reject before even looking for the absent label file.
            closure = folder / "owned_cleanup_before_score.json"; closure.write_text(json.dumps({"owned_main_processes_ended": False}))
            try: score()
            except ValueError: pass
            else: raise AssertionError("Scorer ignored a live worker")
            closure.write_text(json.dumps({"owned_main_processes_ended": True}))
            labels = root / "views/evaluation_labels/SLAKE/train.jsonl"; labels.parent.mkdir(parents=True)
            labels.write_text("".join(json.dumps({"id": str(i), "answer": "neither" if i == 0 else "no"}) + "\n" for i in range(64)))
            try: score()
            except ValueError as exc: assert "nonbinary label" in str(exc)
            else: raise AssertionError("Scorer accepted a nonbinary label")
            assert not (folder / "scores/qualification.json").exists()
            labels.write_text("".join(json.dumps({"id": str(i), "answer": "no"}) + "\n" for i in range(64)))
            score()
            decision = json.loads((folder / "scores/qualification.json").read_text())
            assert decision["go"] and len(decision["blocks"]) == 4 and not decision["training_performed"]
            assert len(json.loads((folder / "scores/anonymous_pool_cases.json").read_text())) == 384
        finally:
            os.environ.clear(); os.environ.update(previous)
    print(json.dumps({"selection_gate_and_full_NA_report_checks_passed": True}))


if __name__ == "__main__": main()
