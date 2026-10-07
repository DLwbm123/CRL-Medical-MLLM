"""Prepare the authorized P2-data fallback without opening correctness labels."""
import hashlib
import json
import os
import shutil
from pathlib import Path

from prepare_campaign import write_manifest
from state import atomic_json
from continual import load_input, validate_development_configuration


def main():
    root = Path(os.environ["P0_ROOT"])
    source = Path(os.environ["P3_P2_FOLDER"])
    folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "inputs").mkdir()
    raw = (source / "manifest.json").read_bytes()
    old_digest = hashlib.sha256(raw).hexdigest()
    if old_digest != (source / "manifest.sha256").read_text().strip():
        raise ValueError("P2 manifest checksum differs")
    old = json.loads(raw)
    candidate = json.loads((source / "manifest_candidate.json").read_text())
    if old["selected_n"] != 16 or len(old["probe"]) != 16 or old["stream"] != candidate["stream"][:16]:
        raise ValueError("Unexpected P2 data boundary")
    groups = [entry["group"] for section in ["stream", "probe", "engineering"] for entry in old[section]]
    if len(groups) != len(set(groups)):
        raise ValueError("P2 stream, probe or engineering image groups overlap")
    for section in ["stream", "probe", "engineering"]:
        for entry in old[section]:
            load_input(source, entry)
            shutil.copyfile(source / entry["input"], folder / entry["input"])
    manifest = {
        **old, "p3_version": 1, "rollout_seeds": [43, 44], "selected_n": 16,
        "data_selection_seed": old["seed"], "source_p2_manifest_sha256": old_digest,
        "data_status": "replication on previously scored development stream and observed probe",
        "unused_candidates_retired_for_p3": False,
        "data_decision": "No separate retirement record for unused candidate groups; use explicit P3 fallback",
        "probe_cursors": [0, 16], "drift_cursors": [1, 8, 16],
        "selection_of_n": "authorized-data fallback fixed before P3 scoring; one common order and two rollout seeds",
    }
    digest = write_manifest(folder / "manifest.json", manifest)
    engineering = {**manifest, "kind": "engineering", "stream": [old["engineering"][1]],
                   "probe": [], "selected_n": 1, "probe_cursors": [], "drift_cursors": [1]}
    write_manifest(folder / "manifest_throughput.json", engineering)
    acceptance = json.loads((source / "acceptance.json").read_text())
    if not acceptance["passed"]:
        raise ValueError("P2 engineering acceptance did not pass")
    acceptance.update(inherited_for_p3=True, inheritance_scope="same training arithmetic/state/RNG routines; declared seeds only",
                      p2_delivery_commit="68c1733025c7be4135aba4b0f9db57c0a6ccaf09")
    for key in "bcd":
        for seed in [43, 44]:
            cfg = json.loads((root / "configs" / f"p3_{key}{seed}.json").read_text())
            validate_development_configuration(cfg, acceptance, manifest)
    atomic_json(folder / "acceptance.json", acceptance)
    atomic_json(folder / "data_scope.json", {
        "source_manifest_sha256": old_digest, "manifest_sha256": digest,
        "stream_groups": [entry["group"] for entry in old["stream"]],
        "probe_groups": [entry["group"] for entry in old["probe"]],
        "unused_candidate_count": len(candidate["stream"]) - 16,
        "unused_candidates_used": 0, "new_test_retirements": 0,
        "stream_previously_scored": True, "probe_scores_previously_observed": True,
        "labels_read": False, "patient_disjointness_verified": False,
    })
    # Only predictions are copied. P2's full states and logs remain untouched.
    cached = folder / "runs/main-a"
    (cached / "cases").mkdir(parents=True)
    (cached / "probe").mkdir()
    original = source / "runs/main-a"
    if json.loads((original / "configuration.json").read_text()) != json.loads((root / "configs/p2_a.json").read_text()):
        raise ValueError("Frozen cache decoding/model configuration differs")
    for i, entry in enumerate(manifest["stream"]):
        path = original / "cases" / f"i{i:06d}.json"
        row = json.loads(path.read_text())
        if row["id"] != entry["id"] or row["index"] != i:
            raise ValueError("Cached baseline order differs")
        shutil.copyfile(path, cached / "cases" / path.name)
    for cursor in manifest["probe_cursors"]:
        name = f"c{cursor:06d}.json"
        shutil.copyfile(original / "probe" / name, cached / "probe" / name)
    shutil.copyfile(original / "configuration.json", cached / "configuration.json")
    record = json.loads((original / "result.json").read_text())
    record.update(manifest_sha256=digest, cached=True,
                  cache_provenance={"source_manifest_sha256": old_digest, "source_code_commit": record["code_commit"],
                                    "model_input_prompt_parser_decode_unchanged": True, "new_generation_seconds": 0})
    atomic_json(cached / "result.json", record)
    print(json.dumps({"prepared": True, "stream": 16, "probe": 16, "seeds": [43, 44],
                      "manifest_sha256": digest, "previously_scored_data": True, "labels_read": False}))


if __name__ == "__main__":
    main()
