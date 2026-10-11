"""Real main-entry regression plus fail-closed hardware-reuse binding."""
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch


def main():
    from test_p24 import main as contract
    contract()
    from continual import main as run, validate_main_budget_lock
    from p24_contract import configurations, physical_gpu
    from state import atomic_json
    base = json.loads((Path(__file__).resolve().parents[1] / "configs/p2_c.json").read_text())
    configs = configurations(base)
    manifest = {"kind": "development", "p10_version": 1, "p24_version": 1,
                "rollout_seeds": [102], "configurations": configs}
    lock = {"admitted": True, "complete_seeds": [102], "complete_arms": ["s", "t", "v"]}
    validate_main_budget_lock(manifest, {"go": True}, lock)
    for bad in [dict(lock, admitted=False), dict(lock, complete_seeds=[]), dict(lock, complete_seeds=[102,103]), dict(lock, complete_arms=["s","v"])]:
        try: validate_main_budget_lock(manifest, {"go":True}, bad)
        except RuntimeError: pass
        else: raise AssertionError("Incomplete matrix admitted")
    try: validate_main_budget_lock(manifest, {"go":False}, lock)
    except RuntimeError: pass
    else: raise AssertionError("Failed reward admitted")
    legacy = {"p10_version":1,"rollout_seeds":[1,2,3]}
    validate_main_budget_lock(legacy, {"go":True}, {"complete_seeds":[1,2,3],"complete_arms":["s","t","v"]})
    try: validate_main_budget_lock({"p10_version":1,"rollout_seeds":[102]}, {"go":True}, lock)
    except RuntimeError: pass
    else: raise AssertionError("Legacy three-seed protection removed")
    try: physical_gpu(7)
    except ValueError: pass
    else: raise AssertionError("Revoked GPU7 admitted")
    class ModelEntryReached(Exception): pass
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary); folder=root/"outputs/fixture"; folder.mkdir(parents=True)
        raw=json.dumps(manifest).encode(); (folder/"manifest.json").write_bytes(raw)
        (folder/"manifest.sha256").write_text(hashlib.sha256(raw).hexdigest())
        atomic_json(folder/"acceptance.json", {"passed":True})
        atomic_json(folder/"scores/qualification.json", {"go":True})
        atomic_json(folder/"main_budget_lock.json", lock)
        atomic_json(root/"config.json", configs["a"]["42"])
        atomic_json(root/"metadata/campaign_budget.json", {"gpu_stop_utc":"2027-01-01T00:00:00+00:00"})
        environment={"P0_ROOT":str(root),"P2_CAMPAIGN":"fixture","P2_CONFIG":"config.json", "P2_RUN":"main-a", "P2_CODE_COMMIT":"fixture-source","CUDA_VISIBLE_DEVICES":"synthetic", "P2_GPU_UUID":"synthetic"}
        with patch.dict(os.environ,environment), patch("torch.cuda.device_count",return_value=1), patch("torch.cuda.reset_peak_memory_stats"), patch("torch.cuda.manual_seed_all"), patch("continual.Engine",side_effect=ModelEntryReached) as loader, patch("continual.traceback.print_exc"):
            try:run()
            except ModelEntryReached:pass
            else:raise AssertionError("Main entry did not reach the model boundary")
            assert loader.call_count==1
            atomic_json(folder/"main_budget_lock.json",dict(lock,admitted=False))
            try:run()
            except RuntimeError:pass
            else:raise AssertionError("Unadmitted main entry loaded a model")
            assert loader.call_count==1
        from prepare_p27 import require_reuse
        probability={"within_tolerance":True,"atol":.1,"max_logp_difference":0.}
        update={"gradient_norms":{"vision":1.,"language":1.},"max_behavior_recompute_error":0.,"masters_persisted":True,"moments_persisted":True,"tracked_actor_updates":{"vision":{"changed_elements":1},"language":{"changed_elements":1}}}
        arm={"passed":True,"fixed_reference_unchanged":True,"four_token_probability":probability,"full_response_probability":probability,"actual_update":update,"checkpoint_recovery":{"passed":True}}
        acceptance={"passed":True,"labels_read":False,"arms":{a:copy.deepcopy(arm) for a in ["s","t","v"]}}
        reuse={"sealed":True,"unchanged_training_execution":True,"manifest_sha256":hashlib.sha256(raw).hexdigest(),"executed_acceptance_source":"090539a9d15154526c36446d55e0da87f769224d","predecessor_delivery":"da48f6c89e8509ad73d9f8e95f3785837efbe08d"}
        atomic_json(folder/"acceptance.json",acceptance); atomic_json(folder/"acceptance_reuse.json",reuse);require_reuse(folder)
        for bad in [dict(reuse,sealed=False),dict(reuse,manifest_sha256="changed"),dict(reuse,executed_acceptance_source="different")]:
            atomic_json(folder/"acceptance_reuse.json",bad)
            try:require_reuse(folder)
            except ValueError:pass
            else:raise AssertionError("Changed acceptance binding admitted")
        atomic_json(folder/"acceptance_reuse.json",reuse)
        acceptance["arms"]["t"]["checkpoint_recovery"]["passed"]=False
        atomic_json(folder/"acceptance.json",acceptance)
        try:require_reuse(folder)
        except ValueError:pass
        else:raise AssertionError("Incomplete actual recovery admitted")
    print(json.dumps({"real_main_single_seed_admission_reaches_model":True,"legacy_three_seed_guard_preserved":True,"incomplete_scope_and_reuse_rejected":True,"GPU7_rejected":True}))


if __name__ == "__main__": main()
