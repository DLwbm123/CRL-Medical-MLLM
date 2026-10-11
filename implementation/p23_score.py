"""Both fixed reward definitions seal before fresh truth is read."""
import json
import os
from pathlib import Path
from p23_reward import validate
from state import atomic_json
from verified_reward import checked_targets


def main():
    folder = Path(os.environ["P0_ROOT"]) / "outputs" / os.environ["P2_CAMPAIGN"]
    manifest = json.loads((folder / "manifest.json").read_text()); validate(manifest)
    digest = (folder / "manifest.sha256").read_text().strip()
    calibration = json.loads((folder / "calibration.json").read_text())
    results = {}
    for key in ["v", "c"]:
        signal = json.loads((folder / "reference" / key / "targets.json").read_text())
        checked_targets({**signal, "stream": signal["stream"] + signal["probe"]},
                        {**manifest, "stream": manifest["stream"] + manifest["probe"], "teacher": manifest["teachers"][key]}, digest)
        if signal["code_commit"] != os.environ["P2_CODE_COMMIT"] or len(signal["stream"]) != 32 or len(signal["probe"]) != 32:
            raise ValueError("Both teacher definitions must seal completely")
        if key == "v" and signal["calibration"] != calibration: raise ValueError("Calibration changed during fresh readouts")
    from p16_score import main as score
    for key in ["v", "c"]:
        score("reference/" + key, "scores/" + key, manifest["teachers"][key])
        results[key] = json.loads((folder / "scores" / key / "qualification.json").read_text())
    atomic_json(folder / "scores/qualification.json", {"go": results["v"]["go"], "primary_teacher": "v", "teachers": results,
        "primary_selected_after_scoring": False, "calibration_frozen_before_fresh_readouts": True, "training_performed": False})


if __name__ == "__main__": main()
