"""Complete P28 training, memory acceptance and continuous cost delivery."""
import json
import os
from pathlib import Path
from report_p24 import export

if __name__ == "__main__":
    print(json.dumps(export(Path(os.environ["P10_LOCAL_FOLDER"]),Path(__file__).resolve().parents[1]/"reports",prefix="p28")))
