"""Small CPU checks for calibration, immutable signals and complete negative reports."""
import copy
import json
import os
import tempfile
from pathlib import Path
from p23_reward import SCHEME, REVISION, calibrate, decision, checked_signal, probabilities, messages
from state import atomic_json
from report_p23 import export


def rejects(call):
    try: call()
    except (ValueError, KeyError): return
    raise AssertionError("Invalid reward contract accepted")


def main():
    orders = [{"reverse": reverse, "real": {"A": -1., "B": -3.}, "masked": {"A": -1., "B": -1.}} for reverse in [False, True]]
    rows = [{"id": str(i), "orders": copy.deepcopy(orders)} for i in range(64)]
    labels = {str(i): "A" for i in range(64)}
    calibration = calibrate(rows, labels)
    assert calibration["admitted"] and calibration["accepted"] == 64 and calibration["correct"] == 64
    assert calibrate(rows, labels) == calibration
    failed = calibrate(rows, {str(i): "B" for i in range(64)})
    assert not failed["admitted"] and failed["threshold"] is None
    calibration.update(sealed=True, manifest_sha256="fixture")
    target, confidence, agree = decision(orders, calibration); assert target == "A" and agree and confidence >= .75
    reverse = copy.deepcopy(orders); reverse[1]["real"] = {"A": -3., "B": -1.}
    assert decision(reverse, calibration)[0] is None
    rejects(lambda: probabilities({"A": float("nan"), "B": 0.}, 1.))
    rejects(lambda: decision(orders[:1], calibration))
    row = {"id": "synthetic", "source_id": 0, "dataset": "SLAKE", "split": "train", "question": "Synthetic?", "options": {"A": "yes", "B": "no"}, "images": ["synthetic"], "image_paths": ["/synthetic"], "language": "en"}
    _, mapping = messages(row, True); assert mapping == {"A": "B", "B": "A"}
    rejects(lambda: messages({**row, "answer": "yes"}))
    record = {"id": "synthetic", "orders": orders, "target": target, "option_keys": ["A", "B"], "confidence": confidence, "same_canonical_choice": agree}
    manifest = {"p23_version": 1, "teacher": {"revision": REVISION}, "stream": [{"id": "synthetic"}]}
    signal = {"reward_scheme": SCHEME, "labels_read": False, "weights_unchanged": True, "sealed": True, "manifest_sha256": "fixture", "teacher_revision": REVISION, "calibration": calibration, "stream": [record]}
    assert checked_signal(signal, manifest, "fixture") == {"synthetic": "A"}
    from verified_reward import checked_targets
    assert checked_targets(signal, manifest, "fixture") == {"synthetic": "A"}
    for mutation in [lambda s: s.update(labels_read=True), lambda s: s["stream"][0].update(target="B"), lambda s: s["calibration"].update(new_labels_used=True), lambda s: s.update(manifest_sha256="wrong")]:
        bad = copy.deepcopy(signal); mutation(bad); rejects(lambda: checked_targets(bad, manifest, "fixture"))
    with tempfile.TemporaryDirectory() as temp:
        folder, public = Path(temp) / "private", Path(temp) / "public"
        atomic_json(folder / "FINAL.json", {"finished_utc": "2026-10-11T00:00:00+00:00", "status": "retired_calibration_negative", "code_commit": "synthetic-source", "gpu_process_seconds_used": 7., "cumulative_gpu_process_seconds": 120698.1088064673, "remaining_gpu_process_seconds": 397701.89119353273})
        atomic_json(folder / "controllers/calibration-plan.json", {"jobs": [{"label": "retired-calibration", "wall_seconds": 7., "return_code": 0}]})
        atomic_json(folder / "calibration.json", failed)
        counts = export(folder, public)
        assert counts["v"] == {"teacher_rows": 64, "pool_rows": 6, "case_rows": 384, "check_rows": 24}
        result = json.loads((public / "p23_results.json").read_text())
        assert result["candidate_qualified"] is None and result["compute"]["total_authorized_seconds"] == 518400 and len(result["compute"]["gpu_jobs"]) == 1
        for file in public.iterdir():
            text = file.read_text(); assert "Synthetic?" not in text and "/synthetic" not in text
    print(json.dumps({"passed": True, "calibration_and_two_order_decision_checked": True, "label_free_signal_boundary_checked": True, "complete_negative_NA_and_unique_cost_checked": True}))


if __name__ == "__main__": main()
