# Medical MLLM adaptation: independent SPINE pilot

An independent implementation based on [SPINE v2](https://arxiv.org/html/2511.17938v2), with a completed **single optimizer-update diagnostic** on Qwen2.5-VL-3B-Instruct. This is not the authors' implementation or a reproduction of their accuracy results.

The corrected run used one predetermined six-image MedXpertQA-MM input, eight sampled responses, and a 2048-token output cap. It completed sampling, label-free voting, backward propagation through the language and vision modules, an AdamW update, greedy decoding, and checkpoint saving in 101.50 seconds. See [the report](reports/pilot_20261004.md) and [aggregate results](reports/pilot_results.json).

## Implementation

`implementation/core.py` contains exact full-vocabulary entropy and KL calculations in FP32, response-wise Otsu selection with 100 bins, detached median/MAD entropy bands, a masked clipped policy objective, and label-free answer voting. Policy and band terms use the total response-token denominator; KL uses the number of selected tokens. Since selection is guaranteed nonempty, the KL denominator omits the paper's additional epsilon. This is a declared numerical difference.

`implementation/run_update.py` implements one update with BF16 model parameters, gradient checkpointing, serialized rollout/backward work, and native PyTorch AdamW on CPU FP32 master parameters and states. It does not freeze vision, add adapters, quantize weights, reduce the eight-response group, shorten the configured output limit, or override image resolution. The CPU environment used about 61 GiB peak RSS; GPU memory alone is not the resource requirement.

Before this only update, actor, behavior, and reference parameters are identical. Detached full-vocabulary base logits therefore serve as the fixed reference for the same sampled states. **The driver deliberately rejects more than one update.** A continuing trainer needs an independently preserved reference policy and a separately specified adaptation protocol. Checkpoints contain model and processor files, not optimizer states.

The `TTRL` and `No adaptation` code paths are available for controlled follow-up work. Only the SPINE branch has received this GPU integration test; no comparison between methods has been run. The answer parser is an independent implementation, not the unavailable author `grade_answer` implementation.

## Reproduce the diagnostic

Use an available, authorized GPU and sufficient CPU RAM/storage. The tested environment was Python 3.12.3, torch 2.7.1+cu128, torchvision 0.22.1+cu128, transformers 4.51.3, CUDA runtime 12.8, driver 595.58.03, and one RTX PRO 5000 72GB Blackwell. The package snapshot is in [metadata/gpu_environment.freeze.txt](metadata/gpu_environment.freeze.txt).

Install the matching CUDA PyTorch wheels from the PyTorch cu128 index, then install the remaining versions in the package snapshot. `pip check` must pass. The snapshot records the tested environment; it is not a promise of compatibility with another CUDA/driver combination.

Keep the interpreter, neutral launcher, cache, and outputs on your allocated storage. Run these commands from the repository root after activating the environment:

```bash
export P0_ROOT="$PWD"
export P1_ENTRY=/path/on/your/storage/p1.py
cp scripts/entry.py "$P1_ENTRY"

# Downloads pinned public resources into data/ and models/.
# Set working HTTP(S) proxies where required by your environment.
export P1_TASK=download_resources.py
python "$P1_ENTRY"

# Checks source structure/images and writes separate input and label views.
export P1_TASK=prepare_views.py
python "$P1_ENTRY"

# Small CPU numerical and label-isolation tests.
export CUDA_VISIBLE_DEVICES=""
export P1_TASK=test_core.py
python "$P1_ENTRY"

# Choose an authorized GPU; keep this value consistent with the config.
export CUDA_VISIBLE_DEVICES=7
export P1_CONFIG=configs/pilot.json
export P1_RUN=p1-unique-run-id
export P1_TASK=run_update.py
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=16
python -u "$P1_ENTRY"
```

The launcher passes task selection and root paths through environment variables. Existing output directories are rejected. Confirm full process arguments and GPU process names after starting. The runner requires locally downloaded weights and runs Hugging Face in offline mode.

`metadata/pinned_downloads.json` records publisher revisions and file sizes for the original model, official SLAKE mirror, and MedXpertQA-MM. The downloader uses Hugging Face resume support and ZIP's built-in CRC checks. It does not add a separate full-file hash pass. Retain the publishers' licenses and access conditions; this repository does not redistribute their resources.

## Data and evaluation boundaries

The adaptation view has exactly these fields: `id`, `source_id`, `dataset`, `split`, `question`, `options`, `images`, `image_paths`, and `language`. The runner rejects extra fields and never opens `views/evaluation_labels`. View construction reads source labels only to save them separately and validate source structure. This is data-flow separation, not an operating-system security sandbox.

The pilot input was fixed as the first maximum-image-count MedXpertQA-MM test example (MM-7, six images) before examining generated outputs. Its identity, seed, prompt, parser, coefficients, optimizer settings, and other engineering choices are explicit in [configs/pilot.json](configs/pilot.json). They have not been verified as author settings. No ground-truth correctness, Pass@1 accuracy, benefit from adaptation, or SLAKE training result is claimed.

Raw datasets, images, prompts/completions, checkpoints, private paths, credentials, and complete runtime logs are excluded from Git. The first parser failure and its cost are retained in the public aggregate report. No continuing experiment or background monitor is part of this release.
