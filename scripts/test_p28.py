"""Full-scope gates, actual-memory barrier and changed cache storage."""
import json
import os
import runpy
import tempfile
from pathlib import Path
from unittest.mock import patch

if __name__ == "__main__":
    from test_p27 import main as admission
    admission()
    from test_generation_replay import main as replay
    replay()
    from p24_contract import configurations
    from state import atomic_json
    root=Path(__file__).resolve().parents[1]
    configs=configurations(json.loads((root/"configs/p2_c.json").read_text()))
    with tempfile.TemporaryDirectory() as name:
        folder=Path(name)/"outputs/fixture"
        manifest={"p24_version":1,"memory_acceptance_required":True,"configurations":configs}
        accepted={"passed":True,"arms":{a:{"passed":True} for a in ["s","t","v"]}}
        atomic_json(folder/"manifest.json",manifest);atomic_json(folder/"acceptance.json",accepted)
        with patch.dict(os.environ,{"P0_ROOT":name,"P2_CAMPAIGN":"fixture"}),patch("continual.main") as model_entry:
            try:runpy.run_path(str(root/"implementation/p24_run.py"),run_name="__main__")
            except ValueError:pass
            else:raise AssertionError("Incomplete larger-prefix acceptance admitted")
            assert not model_entry.called
            for a in accepted["arms"].values():a["memory_challenge"]={"passed":True,"largest_prefix":{"passed":True}}
            atomic_json(folder/"acceptance.json",accepted)
            runpy.run_path(str(root/"implementation/p24_run.py"),run_name="__main__")
            model_entry.assert_called_once()
    print(json.dumps({"incomplete_memory_challenge_rejected_before_main":True,"all_three_memory_challenges_required":True}))
