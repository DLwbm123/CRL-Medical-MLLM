# P21 paired frozen teacher-source reward validation

Predeclared candidate q is the pinned Qwen2.5-VL-7B source; m is the pinned Lingshu-7B medical control. Both sources and all48 gates are reported. A passing control cannot replace a failed primary. The same64 fresh SLAKE training image groups and3072 frozen candidate answers are used for both audits. No optimizer/backward/weight update or automatic RL.

Training-answer vocabulary membership defines this binary task and is not label-blind; no correctness/class balancing/model-score selection. Validation/test and every P16/P17 image reference plus conservative thumbnail duplicates are excluded. No heldout labels/generation/scoring. Patient and pretraining independence remain unverified. This new binary development task does not reverse MedXpertQA negatives or establish clinical generalization.

Both reports contain64 teacher rows, six pool summaries,384 case rows and24 checks including every failure/NA. The paired768 reward observations reuse384 underlying groups on64 images; q/m compute receipts are views of the same eight GPU workers, never separate costs to add together. Frozen cached checks retain original0.1 and uncached observations; they do not validate training loss.

Primary reward qualified: False. Stable RL improvement: false. See p21_q_report.md and p21_m_report.md for complete blocks, negatives and the continuous72h receipt. Raw medical data/text/IDs/labels/images/targets/readouts/checkpoints/full logs/private paths are omitted. Actual closure and effective-proxy final SHA/anonymous verification are required for delivery.

## Closed outcome

All eight GPU workers exited0 and all owned sessions ended. Both256 teacher readouts and all3072 candidates sealed before offline labels were joined. The original FINAL is retained. No optimizer/backward, actor update or RL training occurred.

| Source | Accepted /64 | Correct accepted | Accepted precision | Checks passed /24 | Qualified |
| --- | --- | --- | --- | --- | --- |
| q | 17 | 12 | 70.59% | 9 | False |
| m | 20 | 17 | 85.00% | 14 | False |

The fixed primary q failed15/24 conditions. All four blocks failed coverage, correct-count and precision. Calibration s failed minority rescue; both calibration samplers failed the10pp wrong-positive reduction. Correct-negative rates decreased in all four blocks, and verification wrong-positive reductions passed, but these partial favorable reward diagnostics do not qualify the source. Per-seed harm is retained: q/raw seed90 gave wrong-positive rewards to23.29% of positively rewarded candidates versus18.12% for matched majority. The medical control m also failed qualification (10/24 conditions); its higher conditional precision occurs on a different accepted subset and cannot replace q or establish a causal advantage.

Unique complete GPU process lifetime was 23865.921439802s (6.629423h); continuous usage is 120691.108806467s (33.525308/72h), leaving 138508.891193533s (38.474692h). CPU resource lifetime1.466934042s is recorded separately. The q/m compute receipts are duplicate views of these same jobs and must not be added.

P22 conditional RL preparation remains unstarted because the fixed primary reward gate failed. Its full conservative acceptance/base/nine-trajectory admission including20% reserve requires233280s64.8h; the final continuous remainder is short by 94771.108806467s (26.325308h). No GPU acceptance or partial training is launched. Additional budget alone would not repair the failed reward qualification. No unchanged-source rerun, control substitution or automatic prompt search is admitted. Numerical cached guards passed but native training loss/backward remains unexecuted.
