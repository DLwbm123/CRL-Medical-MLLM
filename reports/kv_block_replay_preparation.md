# KV block replay: preparation only

User authorizes separate implementation and CPU preparation while P30 continues unchanged. No new GPU experiment, training seed, campaign, launch plan, deadline or budget admission is created. This checkout starts at P30 source `10847487db98bbea9c16ddaa319ebabb056fdbec`. No existing implementation, Engine caller, configuration, manifest, optimizer, loss, worker or monitor is modified. This is not a completed GPU experiment.

## Fixed candidate

`implementation/block_generation_replay.py` implements the same response-logit interface as the frozen token replay. Checkpoint blocks contain **8 tokens**, including a final short block. This is one fixed engineering candidate, not a block-size search. Each model forward still handles the original prefill or one subsequent token, with identical generation inputs, cache positions and attention masks. The existing no-gradient path is reused.

One fresh DynamicCache is reconstructed per checkpoint block and mutated only inside that block. Only cloned KV slices for newly consumed positions cross block boundaries as immutable linked history. No past tensor is detached, no parameters are frozen, and later-token loss retains gradient through the visual prefill and all earlier blocks. Non-reentrant checkpoint and RNG preservation remain enabled. Model modes are restored on success, interruption and failure, including recomputation.

This reduces history traversal/reconstruction and mode/checkpoint setup frequency; it does **not** eliminate token forwards, full-vocabulary KL, CPU Adam, or generation. For 17 tokens the tested forward history-rebuild count is 17 versus 3. This is a structural count, not a GPU speedup measurement. Total retained inter-block KV elements remain linear in prefix plus response length; transient block activation/cache memory can increase. Real 3B-model peak memory and accumulation-order differences remain unverified.

## CPU evidence

The separate CPU sandbox used the already installed runtime with CUDA hidden, one CPU thread and offline model access. No model weights/data were loaded and no dependency was installed. P30 files and processes were not changed. See `kv_block_replay_cpu_checks.json`.

Tests passed on lengths 1/7/8/9/17/33, including exact boundary and cross-block cases:

- Original/candidate no-grad and grad logits equal; last-token gradients equal (maximum observed absolute error 0) across every tiny-model parameter, including nonzero visual gradients.
- All four original s/t/v/h objectives use the unchanged shared loss; gradients agree within CPU tolerance 1e-6. h retains its policy mask and full-response KL denominator/coefficient.
- Independently cloned linear retained KV storage; multiple live graphs do not share mutable cache; empty input rejected; forward/backward deadlines and mixed model modes preserved.
- Candidate two updates versus restore-after-one plus one: FP32 masters and Adam state exactly equal in a CPU in-memory state round trip. This is **not** actual-model checkpoint hardware acceptance or a fresh-process recovery claim.
- Existing P30 CPU regression passed, retaining the original 0.1/1e-5 guards, four-arm/label-closure barriers and original per-token replay checks.

An expanded error-injection test initially used a MagicMock as the model forward and caused a later checkpoint metadata mismatch. Replacing that instrumentation with a plain raising function preserved the error-propagation test and made the subsequent multi-graph/recovery checks pass. Candidate implementation did not need to change for this test-harness failure. The test remains in the final suite.

Reproduce in the existing PyTorch/Transformers environment with CUDA hidden:

```sh
CUDA_VISIBLE_DEVICES='' PYTHONPATH=implementation:scripts python scripts/test_block_generation_replay.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=implementation:scripts python scripts/test_p30.py
```

## Proposed GPU verification, not authorized to start

After P30 closure and explicit GPU authorization, freeze a separate validation source/manifest. Use only GPU2 and serial workers; no automatic fallback, search, retries or training launch. Baseline and candidate each cover the complete s/t/v/h acceptance suite: first fixed input, original OOM index 5 with the complete eight private engineering responses and fresh probabilities, and maximum-prefix index 37 with the original fresh eight-response sampler. Twelve challenges **per implementation**, with matched initial weights/RNG and paired generated prefixes for numerical/timing comparison. Reused engineering responses are not scientific training targets.

Require original four-token/full-raw-answer probability <=0.1; statistics-versus-gradient <=1e-5 and exact selections; original loss/full gradients; finite nonzero visual/language gradients; actual BF16 weight changes; persistent FP32 masters/moments; fixed reference unchanged; actual t/h same-process 2-versus-1+1 recovery exact. Also compare original/candidate gradients, masters and moments on the paired inputs, reporting discrepancies rather than assuming mathematical equivalence implies BF16 equality. Any original gate failure stops validation. Synthetic positive advantages are engineering only; temporary weights are discarded.

Measure synchronized behavior/reference/backward/CPU optimizer time, complete worker lifetime, response/prefix lengths, update count and peak allocated/reserved memory for both paths. No speed claim from cache-call counts, shorter answers, skipped updates or mask sparsity. No scientific labels or probes are scored during this engineering validation. Hardware failure does not trigger another block size or retry.

## Budget, continuous ledger preserved

Basis: latest verified ledger **2026-10-14 01:31:20 UTC / 09:31:20 CST**, not a new live GPU check. Full worker lifetime includes loading, CPU optimizer, checkpoint I/O and failures; CPU-only preparation is separate. Reserves are admission margins, not charged consumption. No measured speed discount is applied. Exact arithmetic is in `kv_block_replay_budget.json`.

| Proposed work | Stop-cost basis | With 20% reserve |
|---|---:|---:|
| Original full 12-challenge acceptance | 19,200 s | 23,040 s |
| Candidate full 12-challenge acceptance | 19,200 s | 23,040 s |
| Paired validation total | 38,400 s / 10.6667 h | **46,080 s / 12.8 h** |
| Hypothetical future base + four full arms | 7,200 + 4 x 40,000 = 167,200 s | **200,640 s / 55.7333 h** |
| Validation plus hypothetical full training | 205,600 s | **246,720 s / 68.5333 h** |

No future scientific seed is selected. The hypothetical full training retains all four arms, 256 cases, 64 probes and full acceptance/closure barriers; it is cost planning only, not preparation or authorization for another training round.

Continuous balance at the snapshot is 224,079.797791 s / 62.2444 h. P30 still reserves 99,508.018037 s of remaining t/v/h stop-cost, or 119,409.621644 s with 20%. Balance after that reservation is **104,670.176147 s / 29.0750 h**. The proposed paired validation fits this conservative projection; after reserving validation, 58,590.176147 s / 16.2750 h remains. A full subsequent training round does **not** fit: projected shortfall is **142,049.823853 s / 39.4583 h** under these caps.

Even before reserving any remaining P30 work, the combined validation/training requirement exceeds the snapshot balance by 22,640.202209 s / 6.2889 h. Actual P30 savings may improve the balance, but cannot alone make this unchanged conservative combined budget fit. New measured cost evidence or additional explicit budget would be needed for training admission. Do not reset the 144-hour ledger or move P30 deadlines.

No new GPU work is admitted, scheduled or started. The validation caps bound stopping, not guaranteed completion. CPU tests establish limited implementation evidence only; hardware memory safety, numerical gates and end-to-end acceleration remain pending.
