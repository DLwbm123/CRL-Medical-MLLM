"""CPU-only structural checks and separate, label-free input views.

Original split membership, question/option text and image order are preserved.
These are storage views, not a decision about which split is used for adaptation.
"""
import itertools
import json
import os
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(os.environ["P0_ROOT"])
ADAPT = ROOT / "views/adaptation_inputs"
LABELS = ROOT / "views/evaluation_labels"
ADAPT.mkdir(parents=True, exist_ok=True)
LABELS.mkdir(parents=True, exist_ok=True, mode=0o700)
INPUT_KEYS = {"id", "source_id", "dataset", "split", "question", "options", "images", "image_paths", "language"}
checked_paths = set()
decoded = []
report = {"datasets": {}, "label_isolation": "explicit adaptation-field allowlist; separate label files; path separation is not a security sandbox", "gpu_used": False}


def image_paths(image_root, names):
    paths = []
    for name in names:
        p = (image_root / name).resolve()
        assert p.is_relative_to(image_root.resolve()), "Image escapes dataset root"
        if p not in checked_paths:
            assert p.is_file(), f"Missing referenced image: {p}"
            checked_paths.add(p)
        paths.append(str(p))
    return paths


def decode(paths):
    for path in paths:
        if any(d["path"] == path for d in decoded):
            continue
        with Image.open(path) as image:
            image.load()
            decoded.append({"path": path, "size": list(image.size), "mode": image.mode})


def save(dataset, split, inputs, labels):
    assert len({r["id"] for r in inputs}) == len(inputs), "Duplicate sample ID within split"
    assert [r["id"] for r in inputs] == [r["id"] for r in labels]
    assert all(set(r) == INPUT_KEYS for r in inputs), "Unexpected adaptation field"
    for folder, rows in [(ADAPT, inputs), (LABELS, labels)]:
        path = folder / dataset / (split + ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def slake():
    base = ROOT / "data/SLAKE"
    assert json.loads((ROOT / "metadata/slake_download.json").read_text())["status"] == "complete"
    stats, sets, raw_ids = {}, {}, {}
    for split in ["train", "validation", "test"]:
        rows = json.loads((base / "raw" / (split + ".json")).read_text())
        inputs, labels = [], []
        for r in rows:
            names = [r["img_name"]]
            paths = image_paths(base / "extracted/imgs", names)
            sample_id = f"SLAKE/{split}/{r['qid']}"
            inputs.append(dict(id=sample_id, source_id=r["qid"], dataset="SLAKE", split=split, question=r["question"], options=None, images=names, image_paths=paths, language=r["q_lang"]))
            labels.append(dict(id=sample_id, source_id=r["qid"], answer=r["answer"], answer_type=r["answer_type"], language=r["q_lang"]))
        save("SLAKE", split, inputs, labels)
        sets[split] = {r["img_name"] for r in rows}
        raw_ids[split] = {r["qid"] for r in rows}
        stats[split] = {
            "samples": len(rows), "unique_source_ids": len(raw_ids[split]), "unique_images": len(sets[split]),
            "language": dict(Counter(r["q_lang"] for r in rows)),
            "answer_type": dict(Counter(r["answer_type"] for r in rows)),
            "language_by_answer_type": dict(Counter(r["q_lang"] + "/" + r["answer_type"] for r in rows)),
        }
        # A few image examples per split; cover the three source modalities.
        for modality in sorted({r["modality"] for r in rows}):
            i = next(i for i, r in enumerate(rows) if r["modality"] == modality)
            decode(inputs[i]["image_paths"])
    report["datasets"]["SLAKE"] = {
        "version": "Official website SLAKE 1.0 re-cleaned release, official HF mirror; paper-version equivalence unconfirmed",
        "split_filename_note": "HF publisher renamed validate.json to validation.json",
        "splits": stats, "unique_images_all_splits": len(set.union(*sets.values())),
        "image_overlap_between_original_splits": {a + "/" + b: len(sets[a] & sets[b]) for a, b in itertools.combinations(sets, 2)},
        "source_id_overlap_between_original_splits": {a + "/" + b: len(raw_ids[a] & raw_ids[b]) for a, b in itertools.combinations(raw_ids, 2)},
        "filters_applied": [], "missing_images": 0,
    }


def medxpert():
    base = ROOT / "data/MedXpertQA-MM"
    assert json.loads((ROOT / "metadata/medxpert_download.json").read_text())["status"] == "complete"
    stats, sets, ids = {}, {}, {}
    first = json.loads((base / "raw/MM/dev.jsonl").read_text().splitlines()[0])["images"][0]
    candidates = [p for p in [base / "extracted/images", base / "extracted"] if (p / first).is_file()]
    assert len(candidates) == 1, "Expected exactly one official image root"
    image_root = candidates[0]
    for split in ["dev", "test"]:
        rows = [json.loads(line) for line in (base / "raw/MM" / (split + ".jsonl")).read_text().splitlines() if line.strip()]
        inputs, labels = [], []
        for r in rows:
            assert isinstance(r["images"], list) and r["images"]
            assert r["label"] in r["options"], "Answer label not in option keys"
            paths = image_paths(image_root, r["images"])
            inputs.append(dict(id=r["id"], source_id=r["id"], dataset="MedXpertQA-MM", split=split, question=r["question"], options=r["options"], images=r["images"], image_paths=paths, language="en"))
            labels.append(dict(id=r["id"], source_id=r["id"], label=r["label"], medical_task=r["medical_task"], body_system=r["body_system"], question_type=r["question_type"]))
            assert inputs[-1]["images"] == r["images"]  # Never sort the source image list.
        save("MedXpertQA-MM", split, inputs, labels)
        sets[split] = {x for r in rows for x in r["images"]}
        ids[split] = {r["id"] for r in rows}
        stats[split] = {"samples": len(rows), "unique_source_ids": len(ids[split]), "unique_images": len(sets[split]), "image_references": sum(len(r["images"]) for r in rows), "language": {"en": len(rows)}, "language_source": "official dataset card; not automatic language detection", "images_per_sample": dict(sorted(Counter(len(r["images"]) for r in rows).items())), "multi_image_samples": sum(len(r["images"]) > 1 for r in rows)}
        decode(inputs[0]["image_paths"])
        decode(max(inputs, key=lambda r: len(r["images"]))["image_paths"])
    report["datasets"]["MedXpertQA-MM"] = {"splits": stats, "unique_images_all_splits": len(set.union(*sets.values())), "dev_test_id_overlap": len(ids["dev"] & ids["test"]), "dev_test_image_overlap": len(sets["dev"] & sets["test"]), "missing_images": 0, "filters_applied": [], "text_subset_downloaded": False}


if __name__ == "__main__":
    slake()
    medxpert()
    report.update(referenced_unique_image_paths_checked=len(checked_paths), decoded_examples=decoded)
    (ROOT / "metadata/data_checks.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))
