"""One medical checkpoint replacement; retain the completed continuous ledger."""
import json
import os
from pathlib import Path
from prepare_p10 import prepare


def main():
    source = Path(os.environ["P10_SOURCE_ROOT"])
    prior = json.loads((source / "workspaces/p10-20261009T152542/outputs/p10-20261009T152542/FINAL.json").read_text())
    if prior["status"] != "reward_qualification_negative" or prior["cumulative_gpu_process_seconds"] != 67363.9203985543:
        raise ValueError("P10 completion or continuous cost differs from the frozen P11 admission")
    prepare("p11", (60, 61, 62), prior["cumulative_gpu_process_seconds"],
            "lingshu-medical-mllm/Lingshu-7B", "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9")


if __name__ == "__main__":
    main()
