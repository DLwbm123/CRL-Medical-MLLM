"""Bind only the closed P27 OOM group's existing private tensors for engineering."""
import json
import os
from pathlib import Path
from prepare_p24 import prepare
from state import atomic_json


def main():
    root=Path(os.environ["P0_ROOT"]); source=Path(os.environ["P10_SOURCE_ROOT"])
    prior=source/"workspaces/p27-20261011T154615/outputs/p27-20261011T154615"
    read=lambda p:json.loads(p.read_text())
    audit=read(prior/"completion_audit.json"); final=read(prior/"FINAL.json")
    delivery=read(root/"metadata/p27_delivery_receipt.json")
    if (audit.get("owned_processes_ended") is not True or audit.get("root_error_type") != "OutOfMemoryError"
            or audit.get("failed_case_anonymous_index") != 5 or final.get("status") != "failed"
            or final.get("code_commit") != "02dc68f08a0e4fcd128f69d2222dba4d7f84b8c0"
            or delivery.get("code_commit") != "2cecf9b3dbcad2bffc57d9030535c5979698a881"
            or any(delivery.get(k) is not True for k in ["public_delivery_complete","remote_sha_verified","anonymous_access_verified"])):
        raise ValueError("Unclosed or different memory failure source")
    active=read(prior/"runs/main-s102/active_case.json")
    lengths=[58,97,74,4,221,69,228,4]; tokens=active["response_token_ids"]
    if (active["index"] != 5 or active["input_tokens"] != 1423 or active["response_lengths"] != lengths
            or len(tokens) != 8 or [len(t) for t in tokens] != lengths
            or any(type(t) is not int or not 0 <= t < 151936 for row in tokens for t in row)):
        raise ValueError("Complete prior OOM response tensors differ")
    shapes=[{"anonymous_index":r["index"],"input_tokens":r["input_tokens"]}
        for r in [read(p) for p in sorted((prior/"runs/main-a/cases").glob("i*.json"))]]
    if len(shapes) != 64 or max(shapes,key=lambda x:x["input_tokens"]) != {"anonymous_index":37,"input_tokens":1431}:
        raise ValueError("Complete sealed baseline prefix scope differs")
    prepare()
    folder=root/"outputs"/os.environ["P2_CAMPAIGN"]; manifest=read(folder/"manifest.json")
    old=read(prior/"manifest.json")
    if any(manifest[k] != old[k] for k in ["stream","probe","configurations","reward_binding","rollout_seeds"]):
        raise ValueError("Frozen scientific inputs or settings changed")
    atomic_json(folder/"reference/memory_challenge.json",{"index":5,"input_tokens":1423,
        "response_lengths":lengths,"response_token_ids":tokens,"engineering_only":True,"labels_read":False})
    atomic_json(folder/"memory_challenge.json",{"index":5,"input_tokens":1423,"response_lengths":lengths,
        "source_campaign":"p27-20261011T154615","executed_source":final["code_commit"],
        "final_public_delivery":delivery["code_commit"],"engineering_only":True,"labels_read":False,
        "old_probabilities_or_rewards_reused":False,"no_new_data":True,
        "largest_prefix_index":37,"largest_input_tokens":1431,"complete_prefix_scope_count":64})


if __name__ == "__main__": main()
