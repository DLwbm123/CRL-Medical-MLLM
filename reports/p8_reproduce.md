# P8 reproduction and source boundaries

Use the source commit in p8_compute_receipt.json; diagnostic source was frozen before GPU execution. Old trainer/core/state/readout helpers remain unchanged. P8 does not instantiate Engine or PersistentAdam, reconstruct a trainer, rerun P2 recovery, or call any optimizer step. Source artifacts and clinical inputs remain on the authorized private host; public reports alone cannot reconstruct raw clinical inputs.

All path variables below are privately resolved from configuration/receipts, never guessed from public report aliases. Use the pinned existing runtime, neutral Python stdin or the authorized neutral launcher, and proxy all GitHub access. There is no dependency addition. Commands are templates; execution requires the recorded private manifests and existing authorization.

CPU tests, with no visible CUDA device:

```sh
CUDA_VISIBLE_DEVICES='' "$RUNTIME" - <<'PY'
import os, sys, unittest
sys.path.insert(0, os.environ['P8_IMPLEMENTATION'])
import test_p8
unittest.main(module=test_p8, argv=['checks'])
PY
```

A/B: invoke once per P5/P6/P7, setting P8_SOURCE_WORKSPACE to that round's actual executed checkout, P8_SOURCE_FOLDER to the corresponding artifact folder, P8_ROUND to its round identifier, P8_ROOT to the private artifact root and P8_DEST to a new private destination. The script imports that source's evaluator helpers, verifies provenance/parsing, then independently reads only previously authorized development labels. It never writes old results:

```sh
CUDA_VISIBLE_DEVICES='' "$RUNTIME" - <<'PY'
import os, runpy
runpy.run_path(os.environ['P8_OFFLINE_ENTRY'], run_name='__main__')
PY
```

C/D: freeze diagnostic_manifest.json and freeze_receipt.json, protocol, actor provenance, complete label-free inputs, original pool token IDs/length/decode evidence and the five named modules before this command. The supervisor refuses any existing worker_started/receipt, starts one stdin worker, enforces whole-life/UTC limits, waits for exit and verifies the owned PID is absent from CUDA. It is bounded and does not create a persistent monitor:

```sh
python3 - <<'PY'
import os, runpy
runpy.run_path(os.environ['P8_SUPERVISOR_ENTRY'], run_name='__main__')
PY
```

The actor path, model path, authorized GPU UUID, original source/configuration/seed/arm/cursor/manifest, pool and budget come from that frozen private manifest. P8_IMPLEMENTATION and P8_FOLDER are private environment variables. No clinical labels appear there. Worker output logs remain private; reports do not expose choices, token IDs or raw scores keyed by real option identifiers.

Only after completion, readout seal and confirmed worker exit, run the separate offline scorer with P8_ROOT/P8_FOLDER/P8_P7_FOLDER:

```sh
CUDA_VISIBLE_DEVICES='' "$RUNTIME" - <<'PY'
import os, sys
sys.path.insert(0, os.environ['P8_IMPLEMENTATION'])
from p8_score import main
main()
PY
```

Then set P8_PUBLIC to the sanitized public_diagnostics.json, P0_ROOT to the local checkout, and P8_SOURCE_COMMIT to the frozen source:

```sh
python3 - <<'PY'
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(os.environ['P0_ROOT']) / 'scripts'))
from report_p8 import main
main()
PY
```

Public table assertions require 18 trajectories,288 stream group observations,1152 probe/before/after output observations,112 legal-readout rows and all paired actors. N/A preserves undefined denominators. New matched scores never replace original metrics or training rewards. Optional token/text mask span was omitted. The implementation checks full parameter version counters and buffers, but only three parameter values per tensor; no full weight-byte equality claim is made.

P9 is a preregistration draft only. Do not launch it from these commands or the existing hourly monitor.
