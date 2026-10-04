"""Create a label-blind, image-grouped development manifest; never reads labels."""
import hashlib
import json
import os
import random
from pathlib import Path

ROOT = Path(os.environ["P0_ROOT"])
FOLDER = ROOT / "outputs" / os.environ["P2_CAMPAIGN"]
KEYS = {"id", "source_id", "dataset", "split", "question", "options", "images", "image_paths", "language"}


def write_manifest(path, value):
    raw = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()  # Explicitly requested manifest checksum.
    path.with_suffix(".sha256").write_text(digest + "\n")
    return digest


def prepare():
    FOLDER.mkdir(parents=True, exist_ok=False)
    (FOLDER / "inputs").mkdir()
    rows = []
    for split in ["dev", "test"]:
        source = ROOT / "views/adaptation_inputs/MedXpertQA-MM" / f"{split}.jsonl"
        part = [json.loads(line) for line in source.read_text().splitlines()]
        if not all(set(row) == KEYS and row["split"] == split for row in part):
            raise ValueError("Unexpected fields in label-free source")
        rows.extend(part)
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    image_owner = {}
    for i, row in enumerate(rows):
        for path in row["image_paths"]:
            if path in image_owner:
                parents[find(i)] = find(image_owner[path])
            else:
                image_owner[path] = i
    groups = {}
    for i in range(len(rows)):
        groups.setdefault(find(i), []).append(i)
    dev = [i for i, row in enumerate(rows) if row["split"] == "dev"]
    original_smoke = next(i for i, row in enumerate(rows) if row["id"] == "MM-7")
    engineering = [original_smoke, dev[0], dev[1]]
    excluded = {find(i) for i in engineering}
    rng = random.Random(42)
    candidates = []
    for group, indices in sorted(groups.items()):
        eligible = [i for i in indices if rows[i]["split"] == "test"]
        if eligible and group not in excluded:
            candidates.append(rng.choice(eligible))
    rng.shuffle(candidates)
    if len(candidates) < 80:
        raise ValueError("Insufficient independent image-reference groups")
    probe, stream = candidates[:16], candidates[16:16 + 128]

    def entries(indices, prefix):
        output = []
        for index, row_index in enumerate(indices):
            row = rows[row_index]
            relative = f"inputs/{prefix}{index:03d}.json"
            (FOLDER / relative).write_text(json.dumps(row, ensure_ascii=False) + "\n")
            output.append({"id": row["id"], "split": row["split"], "group": f"g{find(row_index):04d}",
                           "input": relative, "images": row["images"]})
        return output

    manifest = {
        "version": 1, "seed": 42, "source_revision": "7e7c465a68eb2b866926bfa59c8c9d17a8daba65",
        "scope": "user-authorized fixed test development subset; no longer a final-test subset",
        "selection": "random representative per image-reference connected component, then seeded random order",
        "grouping": "shared resolved image paths; no patient IDs; no content-hash deduplication",
        "patient_disjointness_verified": False, "ordering": "fixed random single pass; no domain-boundary signal",
        "labels_read": False, "correctness_used_for_selection": False,
        "available_dev_questions": len(dev), "available_test_questions": len(rows) - len(dev),
        "available_image_groups": len(candidates), "engineering": entries(engineering, "e"),
        "probe": entries(probe, "p"), "stream": entries(stream, "s"),
        "target_n": 64, "candidate_n": [16, 32, 64, 96, 128],
        "probe_decoding": "greedy for all actors; Frozen SC-8 is a stream decoding baseline",
        "scoring_rule": "invalid or missing parsed choice is incorrect; no removal from denominator",
    }
    all_groups = [entry["group"] for section in ["engineering", "probe", "stream"] for entry in manifest[section]]
    if len(all_groups) != len(set(all_groups)):
        raise ValueError("Engineering, stream or probe image groups overlap")
    digest = write_manifest(FOLDER / "manifest_candidate.json", manifest)
    engineering_manifest = {**manifest, "kind": "engineering", "stream": manifest["engineering"][:2],
                            "probe": manifest["engineering"][2:], "selected_n": 2,
                            "probe_cursors": [0, 1, 2], "drift_cursors": [1, 2]}
    write_manifest(FOLDER / "manifest_engineering.json", engineering_manifest)
    print(json.dumps({"candidate_manifest_sha256": digest, "stream_candidates": len(stream),
                      "probe": len(probe), "engineering": len(engineering), "labels_read": False}))


def seal():
    if (FOLDER / "manifest.json").exists():
        raise FileExistsError("A sealed manifest may not be overwritten")
    candidate = FOLDER / "manifest_candidate.json"
    manifest = json.loads(candidate.read_text())
    n = int(os.environ["P2_STREAM_N"])
    if n not in manifest["candidate_n"] or n > len(manifest["stream"]):
        raise ValueError("Stream length is outside predeclared candidates")
    manifest["candidate_manifest_sha256"] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    manifest["stream"] = manifest["stream"][:n]
    manifest["selected_n"] = n
    manifest["kind"] = "development"
    manifest["probe_cursors"] = [0, n // 2, n]
    manifest["drift_cursors"] = [1, n // 2, n]
    manifest["selection_of_n"] = "locked from engineering throughput and time budget before any correctness scoring"
    digest = write_manifest(FOLDER / "manifest.json", manifest)
    print(json.dumps({"manifest_sha256": digest, "stream": n, "probe": len(manifest["probe"])}))


if __name__ == "__main__":
    if os.environ.get("P2_PREPARE_STAGE", "prepare") == "seal":
        seal()
    else:
        prepare()
