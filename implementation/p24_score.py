"""Offline selected-training truth only, after all ten main workers seal and end."""
import json
from p10_score import main
from p3_diagnostics import read_labels
from evaluate import is_correct


def selected_labels(root, manifest):
    if manifest.get("p24_version") != 1 or len(manifest["stream"]) != 64 or len(manifest["probe"]) != 16:
        raise ValueError("Complete locked new development task differs")
    labels = read_labels(root, {"stream": [], "probe": manifest["probe"]})
    wanted = {e["id"] for e in manifest["stream"]}
    selected = {}
    for line in (root / "views/evaluation_labels/SLAKE/train.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["id"] not in wanted: continue
        answer = row["answer"].strip().lower()
        if row["id"] in selected or answer not in {"yes", "no"}: raise ValueError("Sealed binary training domain differs")
        selected[row["id"]] = "A" if answer == "yes" else "B"
    if set(selected) != wanted: raise ValueError("Selected training labels missing")
    return {**labels, **selected}


def original_probes(folder, initial, labels):
    anchor = json.loads((folder / "reference/original_probe_predictions.json").read_text())
    if [p["id"] for p in anchor] != [p["id"] for p in initial]: raise ValueError("Original retention probe IDs/order differ")
    original = {p["id"] for p in anchor if is_correct(p["answer"], labels[p["id"]])}
    current = {p["id"] for p in initial if is_correct(p["answer"], labels[p["id"]])}
    if len(original) != 3 or current != original: raise ValueError("Same three original initially correct probes must remain the anchors")


if __name__ == "__main__": main(selected_labels, original_probes)
