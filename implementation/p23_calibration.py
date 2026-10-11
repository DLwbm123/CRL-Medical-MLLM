"""Read only retired selected training truth after its GPU worker/session closes."""
import json
import os
from pathlib import Path
from p23_reward import calibrate, REVISION
from state import atomic_json


def main():
    folder = Path(os.environ["P0_ROOT"]) / "outputs" / os.environ["P2_CAMPAIGN"]
    closure = json.loads((folder / "calibration_owned_ended.json").read_text())
    ledger = json.loads((folder / "controllers/calibration-plan.json").read_text())
    if not closure["owned_main_processes_ended"] or ledger["status"] != "all_planned_jobs_completed" or len(ledger["jobs"]) != 1 or ledger["jobs"][0]["return_code"] != 0:
        raise ValueError("Retired GPU readouts must seal and session must end before labels")
    manifest = json.loads((folder / "manifest.json").read_text())
    signal = json.loads((folder / "reference/calibration/readouts.json").read_text())
    if not signal["sealed"] or signal["labels_read"] or not signal["weights_unchanged"] or signal["teacher_revision"] != REVISION or signal["manifest_sha256"] != (folder / "manifest.sha256").read_text().strip() or signal["code_commit"] != os.environ["P2_CODE_COMMIT"]:
        raise ValueError("Retired calibration provenance differs")
    ids = [e["id"] for e in manifest["calibration_entries"]]
    found = {}
    for line in (Path(os.environ["P10_SOURCE_ROOT"]) / "views/evaluation_labels/SLAKE/train.jsonl").open():
        row = json.loads(line)
        if row["id"] in ids:
            answer = row["answer"].strip().lower()
            if answer not in {"yes", "no"}: raise ValueError("Retired task domain differs")
            found[row["id"]] = "A" if answer == "yes" else "B"
    if set(found) != set(ids): raise ValueError("Incomplete retired labels")
    result = calibrate(signal["stream"], {key: found[key] for key in ids})
    result.update(sealed=True, source_campaign=manifest["calibration_source"], teacher_revision=REVISION,
                  manifest_sha256=signal["manifest_sha256"], code_commit=os.environ["P2_CODE_COMMIT"], fitted_before_fresh_readouts=True)
    if (folder / "calibration.json").exists(): raise ValueError("Calibration cannot be overwritten")
    atomic_json(folder / "calibration.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__": main()
