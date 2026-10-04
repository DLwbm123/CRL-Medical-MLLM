# Reproduce or resume the continual development experiment

Read [the locked protocol](continual_protocol.md) before interpreting results. This is an independent implementation, using the pilot's engineering settings. It is not verified author code. The old single-update entry remains available.

Use the pinned model and label-free views prepared by the original resource scripts. The runtime is Python 3.12.3, torch 2.7.1+cu128 and transformers 4.51.3. Both RL methods require substantial CPU RAM for full FP32 master weights, gradients and Adam states, as well as a GPU with headroom for two BF16 models and activations. A full trained checkpoint is about 45 GB; retain enough allocated storage for two per run, engineering checkpoints, and atomic-write temporary space. Use an authorized GPU and check actual available memory before each job. Shared-GPU wall times are environment dependent.

The campaign directory is private and excluded from Git. It contains manifests, label-free input JSON, raw predictions, logs and complete checkpoints. Never copy that directory into a public repository. Published figures and summaries come from the evaluator's explicit aggregate schema.

## Environment and a neutral launcher

From the repository root, activate the tested environment. Put the launcher in allocated storage under a neutral name; select work using environment variables so process arguments contain neither a project name nor a method name.

```bash
export P0_ROOT="$PWD"
export P2_ENTRY=/allocated/storage/p2.py
cp scripts/p2_entry.py "$P2_ENTRY"
export P2_CAMPAIGN=p2-unique-run-id
export P2_CODE_COMMIT="$(git rev-parse HEAD)"
export P2_GPU_INDEX=4
export P2_GPU_UUID="$(nvidia-smi -i "$P2_GPU_INDEX" --query-gpu=uuid --format=csv,noheader)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

The four JSON configurations must name the same authorized GPU. The campaign's `metadata/campaign_budget.json` supplies fixed ISO-8601 UTC fields `t0_utc`, `freeze_scope_utc`, `gpu_stop_utc` and `hard_deadline_utc`. Create these once when beginning a new campaign; do not reset them on resume. The original campaign's timestamps are documented in the protocol. Keep runtime metadata private.

## Acceptance before the main stream

```bash
export CUDA_VISIBLE_DEVICES=""
export P2_MODE=test
export P2_TEST_DIR=/allocated/storage/p2-tests-unique
python -u "$P2_ENTRY"

export P2_MODE=score
export P2_EVAL_SELFTEST=1
python -u "$P2_ENTRY"
unset P2_EVAL_SELFTEST

export P2_MODE=prepare
python -u "$P2_ENTRY"
```

The preparation step rejects an existing campaign directory, creates a fixed candidate manifest and a separate two-case engineering manifest, and computes their requested checksums. It does not open the separate label files. Run the original `test_core.py` through the pilot's neutral launcher as well.

Engineering GPU jobs use `P2_MANIFEST=manifest_engineering.json`: run SPINE continuously for two cases, independently for one case followed by a fresh-process resume to two, then TTRL and both frozen baselines for two cases. Enable `P2_PROBABILITY_CHECK=1` on fresh SPINE engineering starts. Use separate run directories. `P2_MODE=compare` compares the full continuous/restored states and discrete outputs; set `P2_COMPARE_RUNS` to a JSON pair of private run IDs. `scripts/accept_campaign.py` documents the required CPU, evaluator, recovery and actual-model result records; `P2_MODE=accept` seals them only after all checks pass. Acceptance is mandatory for the development manifest.

To materialize the CPU acceptance records after candidate preparation, use the already completed tiny-model result, then execute the core and evaluator tests before writing their success records:

```bash
python - <<'PY'
import json, os, runpy, shutil, sys
from pathlib import Path
root = Path(os.environ["P0_ROOT"])
folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
sys.path.insert(0, str(root / "implementation"))
tiny = Path(os.environ["P2_TEST_DIR"]) / "result.json"
assert json.loads(tiny.read_text())["passed"]
runpy.run_path(str(root / "implementation/test_core.py"), run_name="__main__")
os.environ["P2_EVAL_SELFTEST"] = "1"
runpy.run_path(str(root / "implementation/evaluate.py"), run_name="__main__")
shutil.copyfile(tiny, folder / "cpu_acceptance.json")
(folder / "core_acceptance.json").write_text(json.dumps({"passed": True, "tests": ["test_core.py"]}))
(folder / "evaluator_acceptance.json").write_text(json.dumps({"passed": True, "real_labels_read": False}))
PY
```

An executable engineering plan can be created as follows. It compares the two SPINE paths before spending time on the other methods and seals acceptance only after every required job succeeds. The controller stops on any failure; preserve that directory and choose new engineering run IDs when fixing the cause.

```bash
python - <<'PY'
import json, os
from pathlib import Path
folder = Path(os.environ["P0_ROOT"]) / "outputs" / os.environ["P2_CAMPAIGN"]
ids = {key: f"eng-{key}-full" for key in "abcd"}
def run(label, key, run_id, stop, resume=False):
    return {"label": label, "gpu": True, "max_seconds": 3600, "expected_cursor": stop,
            "environment": {"P2_MODE": "run", "P2_RUN": run_id,
                            "P2_CONFIG": f"configs/p2_{key}.json",
                            "P2_MANIFEST": "manifest_engineering.json",
                            "P2_RESUME": "1" if resume else "0",
                            "P2_STOP_CURSOR": str(stop),
                            "P2_PROBABILITY_CHECK": "1" if key == "d" and not resume else "0"}}
jobs = [run("eng-d-full", "d", ids["d"], 2),
        run("eng-d-split-first", "d", "eng-d-split", 1),
        run("eng-d-split-rest", "d", "eng-d-split", 2, True),
        {"label": "eng-compare", "gpu": False, "max_seconds": 1200,
         "environment": {"P2_MODE": "compare", "P2_COMPARE_RUNS": json.dumps([ids["d"], "eng-d-split"])}}]
jobs += [run(f"eng-{key}-full", key, ids[key], 2) for key in "cab"]
jobs.append({"label": "eng-accept", "gpu": False, "max_seconds": 300,
             "environment": {"P2_MODE": "accept", "P2_ENGINEERING_RUNS": json.dumps(ids)}})
with (folder / "plan_engineering.json").open("x") as stream:
    json.dump({"requires_acceptance": False, "minimum_free_mib": 48000, "jobs": jobs}, stream, indent=2)
PY
export P2_MODE=supervise
export P2_PLAN=plan_engineering.json
python -u "$P2_ENTRY"
```

Estimate the full four-method run, including three 16-case probes per method, model loading, state restoration, checkpoint writing and at least 30% remaining-time reserve. Lock a supported sample count from measured engineering costs before scoring any real predictions:

```bash
export P2_MODE=prepare
export P2_PREPARE_STAGE=seal
export P2_STREAM_N=16  # this campaign: use its recorded budget decision
python -u "$P2_ENTRY"
unset P2_PREPARE_STAGE
```

## Start and resume

For an individual first segment, the following environment selects the SPINE configuration; use `p2_a.json`, `p2_b.json`, `p2_c.json` for the other methods and distinct run IDs. Each starts independently from the same original model. This direct form has a worker generation deadline. The controller form below adds an external timeout for all phases and is preferred for unattended execution.

```bash
export CUDA_VISIBLE_DEVICES="$P2_GPU_UUID"
export P2_MODE=run
export P2_CONFIG=configs/p2_d.json
export P2_MANIFEST=manifest.json
export P2_RUN=main-d
export P2_RESUME=0
export P2_STOP_CURSOR=8
export P2_MAX_JOB_SECONDS=7200
python -u "$P2_ENTRY"

# A new process restores the complete latest state and advances the same stream.
export P2_RESUME=1
export P2_STOP_CURSOR=16
python -u "$P2_ENTRY"
```

Never change the configuration, model revision, manifest or recorded code commit while resuming a run. A partial tail after the last checkpoint is preserved under `interrupted/` before replay. Only the latest two complete checkpoints of each new run are retained; older pilot artifacts are untouched.

The following creates the complete main plan for the sealed 16-case manifest. It writes only a new private plan and refuses to overwrite an existing one. The sample count must already have been chosen from engineering costs, and all acceptance records must exist.

```bash
python - <<'PY'
import json, os
from pathlib import Path
folder = Path(os.environ["P0_ROOT"]) / "outputs" / os.environ["P2_CAMPAIGN"]
manifest = json.loads((folder / "manifest.json").read_text())
assert json.loads((folder / "acceptance.json").read_text())["passed"]
n, jobs = manifest["selected_n"], []
for stop in range(8, n + 1, 8):
    for key in "abcd":
        jobs.append({"label": f"main-{key}-{stop}", "gpu": True,
                     "max_seconds": 7200, "expected_cursor": stop,
                     "environment": {"P2_MODE": "run", "P2_RUN": f"main-{key}",
                                     "P2_CONFIG": f"configs/p2_{key}.json",
                                     "P2_MANIFEST": "manifest.json",
                                     "P2_RESUME": "0" if stop == 8 else "1",
                                     "P2_STOP_CURSOR": str(stop),
                                     "P2_PROBABILITY_CHECK": "0"}})
with (folder / "plan_main.json").open("x") as stream:
    json.dump({"requires_acceptance": True, "minimum_free_mib": 48000,
               "jobs": jobs}, stream, indent=2)
PY
```

For bounded unattended execution, write a private plan under the campaign directory. It has `requires_acceptance: true`, `minimum_free_mib`, and an ordered `jobs` array. Each job has a unique `label`, `gpu: true`, `max_seconds`, `expected_cursor`, and `environment` containing the same variables shown above. Order the jobs A8, B8, C8, D8, A16, B16, C16, D16. This provides a four-way common prefix before expanding. Run:

```bash
export P2_MODE=supervise
export P2_PLAN=plan_main.json
python -u "$P2_ENTRY"
```

Use the environment's established detached-process launcher for that command. The supervisor records process identities, checks full arguments and GPU process names, applies per-job and campaign timeouts, and signals only its own recorded worker sessions. It stops at the first failed job. It neither changes another job nor chooses a checkpoint by accuracy. Inspect its startup log once after dispatch. Set `P2_FORBIDDEN_ARGV` to a private JSON array of process-argument strings forbidden by your environment.

## Independent scoring and figures

After prediction artifacts are closed, run the separate evaluator. It validates the manifest and prediction order before opening labels. It uses the longest prefix completed by all four methods and reports actual coverage separately. Invalid parsed answers remain incorrect entries in the denominator. Probe decoding is greedy for every actor, including the frozen SC-8 model.

```bash
export CUDA_VISIBLE_DEVICES=""
export P2_MODE=score
unset P2_EVAL_SELFTEST
python -u "$P2_ENTRY"
```

Default run IDs are `main-a` through `main-d`; override them with `P2_SCORE_RUNS`, a JSON object keyed by `a`, `b`, `c`, `d`. Private case scoring and the redacted public summary are written separately under `scores/`. Trainer processes never read evaluator outputs.

`scripts/plot_continual.py` accepts `P2_SUMMARY` and `P2_PLOT_DIR` through the environment and uses matplotlib 3.10.6. Run it via a neutral copied filename or Python standard input. It exports unsmoothed eight-case cumulative accuracy points and the initial/midpoint/final probe plot as PNG/PDF. For fewer than 32 common cases it emits no trend plots; report exact-count tables instead. Plotting dependencies can be installed in a separate environment without changing the model runtime.
