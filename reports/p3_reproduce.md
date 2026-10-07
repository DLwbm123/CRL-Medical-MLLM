# P3 launch, resume and offline evaluation

Use the P3 branch with the pinned P2 runtime and authorized private data. Set `WORKSPACE`, `CAMPAIGN`, `P2_ARTIFACTS`, `RUNTIME` and `NEUTRAL_ENTRY` privately. The workspace contains the source, pinned model/views, its own `metadata/campaign_budget.json`, and a separate output campaign. Do not replace the P2 budget or source. Set the UTC/monotonic clock before the first operation; the saved budget has the absolute main/all-GPU/delivery deadlines described in `p3_protocol.md`.

All values containing data locations or method selection travel through environment variables. The launched interpreter and entry path must be neutral. For example, copy the entry using Python stdin, avoiding sensitive paths in child arguments:

```sh
export P0_ROOT="$WORKSPACE" P2_CAMPAIGN="$CAMPAIGN" P3_P2_FOLDER="$P2_ARTIFACTS"
export P2_ENTRY="$NEUTRAL_ENTRY" P2_GPU_INDEX=4 P2_GPU_UUID="$AUTHORIZED_GPU_UUID"
export CUDA_VISIBLE_DEVICES="$P2_GPU_UUID"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export P2_FORBIDDEN_ARGV="$PRIVATE_FORBIDDEN_ARGV_JSON"
"$RUNTIME" - <<'PY'
import os, shutil
from pathlib import Path
root = Path(os.environ['P0_ROOT'])
shutil.copyfile(root / 'scripts/p2_entry.py', os.environ['P2_ENTRY'])
PY
```

Prepare the label-free manifest and permitted frozen cache, then run CPU checks with `CUDA_VISIBLE_DEVICES` empty. Preparation refuses an existing campaign:

```sh
export P2_MODE=p3_prepare CUDA_VISIBLE_DEVICES=""
"$RUNTIME" -u "$P2_ENTRY"
export P2_MODE=test P2_TEST_DIR="$WORKSPACE/outputs/$CAMPAIGN/cpu-regression"
"$RUNTIME" -u "$P2_ENTRY"
export P2_MODE=p3_test
"$RUNTIME" -u "$P2_ENTRY"
export P2_MODE=p3_audit
"$RUNTIME" -u "$P2_ENTRY"
```

Reconstruct the P2 original scores before proceeding. The audit is offline and can open only the explicitly wanted old labels. Its format rules remain outside training. Review all invalid-output marker/ending evidence and parsed controls privately.

The initial supervisor plan runs `p3_readout` with a 2700-second external limit (2650 internal), followed by one TTRL engineering case using `manifest_throughput.json`. The full main plan runs `main-c43`, `main-d43`, `main-c44`, `main-d44`, `main-b43`, `main-b44`, each to cursor 16, with configuration paths `configs/p3_<key><seed>.json`; frozen greedy is the validated cache. Every job's environment selects `P2_MODE`, `P2_RUN`, `P2_CONFIG`, `P2_MANIFEST` and `P2_STOP_CURSOR`. Before launch, lock the measured budget and at least 20% reserve, and set `P2_CODE_COMMIT` to the training-source commit recorded in the report. Do not join P3 labels yet.

```sh
export CUDA_VISIBLE_DEVICES="$P2_GPU_UUID" P2_MODE=supervise
export P2_PLAN=initial-plan.json
"$RUNTIME" -u "$P2_ENTRY"
# After the short check and budget lock:
export P2_PLAN=main-plan.json
"$RUNTIME" -u "$P2_ENTRY"
```

The executed private plan JSONs and controller receipts are retained. Each plan has `requires_acceptance`, `minimum_free_mib` and a `jobs` array; each job has `label`, `gpu`, `max_seconds`, `expected_cursor` where applicable, and `environment`. Supervision records PID/start ticks, checks `ps`/`nvidia-smi`, and stops only owned workers on failure or deadline. Background execution should use the established private launcher so disconnects cannot interrupt jobs. Do not launch a second supervisor while one is active.

Resume is supported by the unchanged P2 trainer, but P3 did not add restarts solely to demonstrate it. Resume only an explicitly authorized interrupted trajectory with exactly its original configuration, manifest and training commit, under the original deadline. Do not load a P2 adapted actor into P3 main runs:

```sh
export P2_MODE=run P2_MANIFEST=manifest.json P2_RUN="$ORIGINAL_RUN"
export P2_CONFIG="$ORIGINAL_CONFIG" P2_CODE_COMMIT="$ORIGINAL_TRAINING_COMMIT"
export P2_RESUME=1 P2_STOP_CURSOR="$LOCKED_NEXT_CURSOR"
"$RUNTIME" -u "$P2_ENTRY"
```

When the main controller closes, confirm every recorded main worker/session has ended and save the private `owned_cleanup_before_score.json` attestation. The scorer validates both seeds' configuration, contiguous coverage, original parse outputs and all probe/readout records, writes `scores/prediction_seal.json`, and only then reads truth labels:

```sh
export CUDA_VISIBLE_DEVICES="" P2_MODE=p3_score
unset P2_RESUME
"$RUNTIME" -u "$P2_ENTRY"
```

Publish only aggregate reports and source. Generated text, patient data, truth labels/IDs, full states and private plans/paths remain private. GitHub access must use the configured proxy without direct fallback. The original P2/main branches and reports are unchanged.
