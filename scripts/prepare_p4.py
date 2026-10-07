"""Prepare a new bounded development round while preserving earlier campaigns."""
import json
import hashlib
import os
import shutil
from pathlib import Path
from continual import load_input, validate_development_configuration
from prepare_campaign import write_manifest
from state import atomic_json


def main():
    root = Path(os.environ["P0_ROOT"])
    source = Path(os.environ["P4_P3_FOLDER"])
    folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "inputs").mkdir()
    raw = (source / "manifest.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != (source / "manifest.sha256").read_text().strip():
        raise ValueError("Source manifest changed")
    old = json.loads(raw)
    if len(old["stream"]) != 16 or len(old["probe"]) != 16:
        raise ValueError("Unexpected authorized development boundary")
    groups = [row["group"] for name in ["stream", "probe", "engineering"] for row in old[name]]
    if len(groups) != len(set(groups)):
        raise ValueError("Image groups overlap")
    for section in ["stream", "probe", "engineering"]:
        for entry in old[section]:
            load_input(source, entry)
            shutil.copyfile(source / entry["input"], folder / entry["input"])
    manifest = {**old, "p4_version": 1, "rollout_seeds": [45, 46, 47],
                "reward_sources": ["majority", "frozen_legal"],
                "data_status": "reused observed development groups; new rollout seeds",
                "probe_cursors": [0, 16], "selected_n": 16,
                "data_decision": "No new test retirement; prior observed groups only"}
    digest = write_manifest(folder / "manifest.json", manifest)
    acceptance = json.loads((source / "acceptance.json").read_text())
    if not acceptance["passed"]:
        raise ValueError("Inherited engineering acceptance did not pass")
    acceptance.update(inherited_for_p4=True,
                      inheritance_scope="same binary-reward normalization, objective, optimizer, state and RNG; added fixed pseudo-target source tested separately")
    for kind in ["m", "v"]:
        for seed in [45, 46, 47]:
            cfg = json.loads((root / "configs" / f"p4_{kind}{seed}.json").read_text())
            validate_development_configuration(cfg, acceptance, manifest)
    atomic_json(folder / "acceptance.json", acceptance)
    original = source / "runs/main-a"
    if json.loads((original / "configuration.json").read_text()) != json.loads((root / "configs/p2_a.json").read_text()):
        raise ValueError("Frozen cache configuration differs")
    cached = folder / "runs/main-a"
    (cached / "cases").mkdir(parents=True)
    (cached / "probe").mkdir()
    for index, entry in enumerate(manifest["stream"]):
        path = original / "cases" / f"i{index:06d}.json"
        saved = json.loads(path.read_text())
        if saved["index"] != index or saved["id"] != entry["id"]:
            raise ValueError("Frozen cache order differs")
        shutil.copyfile(path, cached / "cases" / path.name)
    for cursor in manifest["probe_cursors"]:
        name = f"c{cursor:06d}.json"
        shutil.copyfile(original / "probe" / name, cached / "probe" / name)
    shutil.copyfile(original / "configuration.json", cached / "configuration.json")
    result = json.loads((original / "result.json").read_text())
    result.update(manifest_sha256=digest, cached=True)
    atomic_json(cached / "result.json", result)
    print(json.dumps({"prepared": True, "stream": 16, "probe": 16, "seeds": [45, 46, 47],
                      "manifest_sha256": digest, "labels_read": False}))


if __name__ == "__main__":
    main()
