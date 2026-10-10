"""Export the separately frozen binary-domain recovery with all missing outcomes."""
import json
import os
from pathlib import Path
from report_p16 import export

if __name__ == "__main__":
    print(json.dumps(export(Path(os.environ["P10_LOCAL_FOLDER"]), Path(__file__).resolve().parents[1] / "reports", "p17", (78, 79, 80))))
