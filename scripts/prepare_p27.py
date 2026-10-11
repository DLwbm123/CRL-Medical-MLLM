"""Bind unchanged, closed P26 hardware acceptance to the repaired main gate."""
import ast
import hashlib
import json
import os
from pathlib import Path
from prepare_p24 import prepare
from p24_accept import validate_update, require_probability
from state import atomic_json


def read(path):
    return json.loads(path.read_text())


def require_reuse(folder):
    manifest = read(folder / "manifest.json")
    reuse = read(folder / "acceptance_reuse.json")
    acceptance = read(folder / "acceptance.json")
    if (reuse.get("sealed") is not True or reuse.get("unchanged_training_execution") is not True
            or reuse.get("manifest_sha256") != hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest()
            or reuse.get("executed_acceptance_source") != "090539a9d15154526c36446d55e0da87f769224d"
            or reuse.get("predecessor_delivery") != "da48f6c89e8509ad73d9f8e95f3785837efbe08d"
            or not acceptance.get("passed") or acceptance.get("labels_read") is not False
            or set(acceptance["arms"]) != {"s", "t", "v"}):
        raise ValueError("Closed actual acceptance reuse binding differs")
    for arm, value in acceptance["arms"].items():
        if value.get("passed") is not True or value.get("fixed_reference_unchanged") is not True:
            raise ValueError("An actual arm is incomplete")
        require_probability(value["four_token_probability"])
        require_probability(value["full_response_probability"])
        validate_update(value["actual_update"], manifest["configurations"][arm]["102"])
    if acceptance["arms"]["t"]["checkpoint_recovery"].get("passed") is not True:
        raise ValueError("Actual checkpoint recovery is incomplete")


def main():
    root = Path(os.environ["P0_ROOT"])
    source = Path(os.environ["P10_SOURCE_ROOT"])
    old = source / "workspaces/p26-20261011T142055"
    previous = old / "outputs/p26-20261011T142055"
    final, audit = read(previous / "FINAL.json"), read(previous / "completion_audit.json")
    delivery = read(root / "metadata/p26_delivery_receipt.json")
    if (final.get("status") != "failed" or final.get("code_commit") != "090539a9d15154526c36446d55e0da87f769224d"
            or audit.get("failure_stage") != "legacy_main_admission_scope_validation"
            or audit.get("owned_processes_ended") is not True or audit.get("baseline_model_loaded") is not False
            or audit.get("scientific_training_started") is not False
            or delivery.get("code_commit") != "da48f6c89e8509ad73d9f8e95f3785837efbe08d"
            or any(delivery.get(k) is not True for k in ["public_delivery_complete", "remote_sha_verified", "anonymous_access_verified"])):
        raise ValueError("Failed predecessor is unclosed or its acceptance is unavailable")
    for name in ["generation_replay.py", "p19_cache.py", "p24_accept.py", "state.py", "core.py", "verified_reward.py"]:
        if (root / "implementation" / name).read_bytes() != (old / "implementation" / name).read_bytes():
            raise ValueError("The accepted training execution changed")
    def training_ast(workspace):
        tree = ast.parse((workspace / "implementation/continual.py").read_text())
        return [ast.dump(n, include_attributes=False) for n in tree.body if getattr(n, "name", None) in {"Engine", "forward_response", "group_rewards", "load_input"}]
    if training_ast(root) != training_ast(old):
        raise ValueError("The accepted Engine changed")
    prepare()
    folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
    manifest, prior = read(folder / "manifest.json"), read(previous / "manifest.json")
    if any(manifest[k] != prior[k] for k in ["stream", "probe", "configurations", "reward_binding", "rollout_seeds", "execution_physical_gpu"]):
        raise ValueError("Accepted inputs/configuration/reward binding changed")
    atomic_json(folder / "acceptance.json", read(previous / "acceptance.json"))
    atomic_json(folder / "acceptance_reuse.json", {"sealed": True, "unchanged_training_execution": True,
        "manifest_sha256": hashlib.sha256((folder / "manifest.json").read_bytes()).hexdigest(),
        "executed_acceptance_source": final["code_commit"], "predecessor_delivery": delivery["code_commit"],
        "acceptance_reused_not_rerun": True, "previous_GPU_cost_not_recounted": True,
        "diagnostic_weights_reused": False, "new_actual_fresh_process_recovery_claimed": False})
    require_reuse(folder)


if __name__ == "__main__": main()
