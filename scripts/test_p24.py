"""Complete conditional binding, budget stop, label barrier and NA coverage."""
import copy
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
from p23_reward import REPO, REVISION, SCHEME
SOURCES = {"v": (REPO, REVISION), "c": (REPO, REVISION)}
from p24_contract import admission, configurations, FULL_ESTIMATE, SEEDS, CHECKS
from prepare_campaign import write_manifest
from prepare_p24 import prepare
from report_p24 import export
from state import atomic_json


def main():
    environment = dict(os.environ); code_root = Path(os.environ["P0_ROOT"])
    try:
        assert FULL_ESTIMATE == 194400.
        assert admission(120691.1088064673 + 155520, 66*3600)["admitted"]
        assert admission(0, 66*3600)["admitted"]
        assert not admission(0, 233279)["admitted"]
        assert not admission(400000, 66*3600, 7200, True)["admitted"]
        for number in [float("nan"), -1., float("inf"), 518401.]:
            try: admission(number, 66*3600)
            except ValueError: pass
            else: raise AssertionError("Invalid cost admitted")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "new"; source = Path(temporary) / "old"
            folder = root / "outputs/fixture"; old = source / "workspaces/p23-fixture/outputs/p23-fixture"
            legacy = source / "workspaces/p15-20261010T095303/outputs/p15-20261010T095303"
            base = json.loads((code_root / "configs/p2_c.json").read_text()); atomic_json(source / "configs/p2_c.json", base)
            teachers = {k: {"repo": repo, "revision": rev, "model_path": "/synthetic", "evidence_first": True,
                "json_suffix": True, "max_new_tokens": 1024, "option_orders": ["sorted", "reversed"], "gpu_index": 1} for k, (repo, rev) in SOURCES.items()}
            entries = []
            for i in range(64):
                row = {"id": f"SLAKE/train/{i}", "source_id": i, "dataset": "SLAKE", "split": "train", "question": "Synthetic?",
                    "options": {"A": "yes", "B": "no"}, "images": [str(i)], "image_paths": ["/synthetic"], "language": "en"}
                rel = f"inputs/i{i:03d}.json"; atomic_json(old / rel, row); entries.append({"id": row["id"], "input": rel, "split": "train", "images": row["images"]})
            probes = []
            for i in range(16):
                row = {"id": f"MedXpertQA-MM/train/{i}", "source_id": i, "dataset": "MedXpertQA-MM", "split": "train", "question": "Synthetic?",
                    "options": {"A": "yes", "B": "no"}, "images": ["old"+str(i)], "image_paths": ["/synthetic"], "language": "en"}
                rel = f"inputs/p{i:03d}.json"; atomic_json(legacy / rel, row); probes.append({"id": row["id"], "input": rel, "split": "train", "images": row["images"]})
            atomic_json(legacy / "manifest.json", {"kind": "development", "probe": probes, "stream": [], "p15_version": 1})
            atomic_json(source / "outputs/p7-20261008T044250/runs/main-m54/probe/c000000.json", {"predictions": [{"id": e["id"], "answer": "A"} for e in probes]})
            configs = {f"pool-{arm}{seed}": {**base, "seed": seed, "method": "Frozen SC-8", "temperature": temperature, "top_p": top_p}
                for seed in [96,97,98] for arm, temperature, top_p in [("s", .7, .95), ("t", 1., 1.)]}
            manifest = {"p23_version": 1, "stream": entries[:32], "probe": entries[32:], "rollout_seeds": [96,97,98],
                "primary_teacher": "v", "teacher": teachers["v"], "teachers": teachers, "configurations": configs, "training_authorized_in_this_round": False,
                "calibration_entries": entries, "calibration_source": "p21-retired", "visual_reward": {"scheme": SCHEME, "real_coefficient": 2, "masked_coefficient": -1, "masked_rgb": [128,128,128], "temperature_bounds": [.25,8], "minimum_confidence": .75, "retired_minimum_accepted":32, "retired_minimum_correct":24},
                "gate": {"minimum_accepted": 16, "minimum_correct": 12, "minimum_precision": .75, "wrong_positive_drop_pp": 10}}
            digest = write_manifest(old / "manifest.json", manifest)
            calibration = {"admitted": True, "temperature": 1., "threshold": .75, "sealed": True, "retired_groups":64, "new_labels_used":False, "manifest_sha256":digest}
            orders = [{"reverse": reverse, "real": {"A": -1., "B": -3.}, "masked": {"A": -1., "B": -1.}} for reverse in [False, True]]
            from p23_reward import decision
            target, confidence, agree = decision(orders, calibration)
            records = [{"id":e["id"],"option_keys":["A","B"],"orders":orders,"target":target,"confidence":confidence,"same_canonical_choice":agree} for e in entries]
            atomic_json(old / "reference/v/targets.json", {"reward_scheme":SCHEME, "calibration":calibration, "stream": records[:32], "probe": records[32:], "teacher_revision": teachers["v"]["revision"], "code_commit": "old-source",
                "manifest_sha256": digest, "labels_read": False, "weights_unchanged": True, "sealed": True})
            final = {"status": "reward_qualified", "code_commit": "old-source", "cumulative_gpu_process_seconds": 400000., "remaining_gpu_process_seconds": 118400.,
                "whole_gpu_worker_lifetimes_counted": True, "budget_reset": False}
            atomic_json(old / "FINAL.json", final)
            gate = {"go": True, "primary_teacher": "v", "primary_selected_after_scoring": False, "teachers": {"v": {"go": True,
                "blocks": {b: {"checks": {name: True for name in CHECKS}} for b in ["calibration_s","calibration_t","verification_s","verification_t"]}}, "c": {"go": True}}}
            atomic_json(old / "scores/qualification.json", gate)
            atomic_json(root / "metadata/p23_delivery_receipt.json", {"public_delivery_complete": True, "remote_sha_verified": True, "anonymous_access_verified": True, "code_commit": "published-report"})
            for stage, count in [("calibration", 1), ("qualification", 2), ("main", 6)]:
                atomic_json(old / "controllers" / (stage+"-plan.json"), {"status": "all_planned_jobs_completed", "jobs": [{"return_code": 0} for _ in range(count)]})
            for name in ["calibration_owned_ended.json", "qualifier_owned_ended.json", "owned_cleanup_before_score.json"]: atomic_json(old / name, {"owned_main_processes_ended": True})
            os.environ.update(P0_ROOT=str(root), P10_SOURCE_ROOT=str(source), P2_CAMPAIGN="fixture", P24_PREDECESSOR="p23-fixture", P2_CODE_COMMIT="new-source")
            failed_q = copy.deepcopy(gate); failed_q["teachers"]["v"]["blocks"]["verification_s"]["checks"]["correct_minority_rescued"] = False
            for bad_gate in [dict(gate, go=False), dict(gate, primary_teacher="c"), dict(gate, primary_selected_after_scoring=True), failed_q]:
                atomic_json(old / "scores/qualification.json", bad_gate)
                try: prepare()
                except ValueError: pass
                else: raise AssertionError("Failed primary or post-score replacement admitted")
                assert not folder.exists()
            atomic_json(old / "scores/qualification.json", gate)
            for path,key in [(root/"metadata/p23_delivery_receipt.json","anonymous_access_verified"),(old/"owned_cleanup_before_score.json","owned_main_processes_ended")]:
                saved=path.read_text();bad=json.loads(saved);bad[key]=False;atomic_json(path,bad)
                try:prepare()
                except ValueError:pass
                else:raise AssertionError("Unclosed or undelivered predecessor admitted")
                assert not folder.exists();path.write_text(saved)
            prepare()
            locked = json.loads((folder / "manifest.json").read_text())
            assert (len(locked["stream"]), len(locked["probe"]), locked["rollout_seeds"]) == (64,16,SEEDS)
            assert locked["configurations"] == configurations(base)
            assert len(json.loads((folder/"reference/targets.json").read_text())["stream"]) == 64
            assert not json.loads((folder/"budget_admission.json").read_text())["admitted"]
            with patch("p10_pipeline.supervise", side_effect=AssertionError("Budget-blocked GPU started")), patch("p10_pipeline.invoke", side_effect=AssertionError("Budget-blocked worker started")):
                from p24_pipeline import main as pipeline
                with patch("p24_pipeline.supervise", side_effect=AssertionError("GPU started")), patch("p24_pipeline.invoke", side_effect=AssertionError("Worker started")):
                    pipeline()
            final = json.loads((folder/"FINAL.json").read_text())
            assert final["status"] == "full_matrix_budget_admission_blocked" and final["gpu_process_seconds_used"] == 0
            counts = export(folder, root/"public"); assert counts == {"matrix_rows":9,"case_rows":576,"probe_rows":144}
            for path in (root/"public").glob("*"):
                text = path.read_text(); assert "Synthetic?" not in text and "SLAKE/train/" not in text and "/synthetic" not in text
            from p24_score import selected_labels, original_probes
            atomic_json(root/"views/evaluation_labels/MedXpertQA-MM/train.jsonl.fixture", {})
            labels = root/"views/evaluation_labels/MedXpertQA-MM/train.jsonl"
            labels.write_text("".join(json.dumps({"id":e["id"],"label":"A" if i<3 else "B"})+"\n" for i,e in enumerate(probes)))
            labels = root/"views/evaluation_labels/SLAKE/train.jsonl"; labels.parent.mkdir(parents=True)
            labels.write_text("".join(json.dumps({"id":e["id"],"answer":"yes"})+"\n" for e in entries))
            values = selected_labels(root, locked); assert len(values)==80
            initial = [{"id": e["id"], "answer": "A"} for e in probes]; original_probes(folder, initial, values)
            changed = copy.deepcopy(initial); changed[0]["answer"]="B"; changed[3]["answer"]="B"
            try: original_probes(folder, changed, values)
            except ValueError: pass
            else: raise AssertionError("Different three initial correct anchors admitted")
            from p10_score import main as score
            with patch("p10_score.read_labels", side_effect=AssertionError("Labels read before complete matrix")):
                try: score(lambda *_: (_ for _ in ()).throw(AssertionError("Label join reached")))
                except FileNotFoundError: pass
                else: raise AssertionError("Unstarted matrix scored")
            from continual import group_rewards
            signal = json.loads((folder/"reference/targets.json").read_text())
            new_digest = (folder/"manifest.sha256").read_text().strip()
            predictions = [{**p,"text":"Final answer: A","tokens":4} for p in initial]
            for arm, values_by_seed in locked["configurations"].items():
                for seed,cfg in values_by_seed.items():
                    run = folder/"runs"/("main-a" if arm=="a" else f"main-{arm}{seed}")
                    atomic_json(run/"configuration.json",cfg)
                    atomic_json(run/"result.json",{"status":"completed","cursor":64,"method":cfg["method"],"manifest_sha256":new_digest,"reference_unchanged":True,"code_commit":"new-source"})
                    if arm=="v":atomic_json(run/"reference_signal_snapshot.json",signal)
                    for cursor in [0,64]:atomic_json(run/"probe"/f"c{cursor:06d}.json",{"cursor":cursor,"predictions":predictions,"rng_and_modes_restored":True})
                    for i,e in enumerate(locked["stream"]):
                        row={"id":e["id"],"index":i,"before":{"text":"Final answer: A","answer":"A","tokens":4},"after":{"text":"Final answer: A","answer":"A","tokens":4},"update_applied":False,"skip_reason":"synthetic_test","elapsed_seconds":0.,"phase_seconds":{}}
                        if arm!="a":
                            completions=["Final answer: A"]*4+["Final answer: B"]*4
                            rewards,_,vote=group_rewards(completions,{"A":"yes","B":"no"})
                            row.update(completions=completions,rewards=rewards.tolist(),vote=vote,response_lengths=[4]*8,hit_length_cap=[False]*8)
                        atomic_json(run/"cases"/f"i{i:06d}.json",row)
            atomic_json(folder/"controllers/main-plan.json",{"status":"all_planned_jobs_completed","jobs":[{"return_code":0}]*10})
            atomic_json(folder/"owned_cleanup_before_score.json",{"owned_main_processes_ended":True})
            atomic_json(folder/"acceptance.json",{"passed":True})
            with patch("p10_score.resource_summary",return_value={}):score(selected_labels,original_probes)
            matrix=json.loads((folder/"scores/public_summary.json").read_text())
            assert matrix["n"]==64 and matrix["frozen_correct"]==64 and len(matrix["probe_rows"])==144 and not matrix["stable_positive_development_result"]
            # The new actual checkpoint path is also exercised with a small real CPU optimizer.
            import torch
            from types import SimpleNamespace
            from state import PersistentAdam
            from p24_accept import recovery,validate_update
            actor=torch.nn.Module();actor.visual=torch.nn.Linear(2,2,bias=False);actor.language=torch.nn.Linear(2,2,bias=False)
            state=PersistentAdam(actor,base)
            def update(*_):
                before={n:p.detach().clone() for n,p in actor.named_parameters()}
                state.zero_grad();actor.language(actor.visual(torch.ones(1,2))).sum().backward()
                value=state.step()
                return {**value,"max_behavior_recompute_error":0.,"tracked_actor_updates":{n:{"changed_elements":int((p!=before[n]).sum())} for n,p in actor.named_parameters()}}
            validate_update(update(),base)
            engine=SimpleNamespace(cfg=base,state=state,method="TTRL",update=update)
            assert recovery(engine,None,None,None,root/"tiny_actual_schema",new_digest)["passed"]
            for mutate in [lambda v:v["gradient_norms"].update(vision=0),lambda v:v.update(max_behavior_recompute_error=float("nan"))]:
                bad=copy.deepcopy(update());mutate(bad)
                try:validate_update(bad,base)
                except ValueError:pass
                else:raise AssertionError("Invalid actual gradient diagnostic admitted")
    finally:
        os.environ.clear(); os.environ.update(environment)
    print(json.dumps({"passed":True,"qualified_primary_delivery_full_budget_no_GPU_label_barrier_anchor_and_full_NA_checked":True}))


if __name__ == "__main__": main()
