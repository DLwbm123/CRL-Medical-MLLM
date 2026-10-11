# P23_V medical reward validation

Final status: reward_qualified. Executed source: 93015111134282e3f7d61bb43ab26d4355456bb9.

64 SLAKE English training image groups selected for an intended yes/no task, fixed32 calibration +32 verification; six frozen SC-8 pools at seeds96/97/98 and two sampling distributions. Validation/test image-reference overlaps and conservative thumbnail duplicates were excluded. Patient independence and absence of pretraining overlap remain unverified.

This new binary development task differs from retired MedXpertQA and cannot reverse its negative results or establish independent clinical generalization. v uses fixed calibrated real/masked answer scores; c uses fallible self-reported visual support.

| Block / sampler | Accepted /32 | Correct accepted /32 | Six checks passed |
| --- | --- | --- | --- |
| calibration_s | 21 | 21 | 6/6 |
| calibration_t | 21 | 21 | 6/6 |
| verification_s | 23 | 23 | 6/6 |
| verification_t | 23 | 23 | 6/6 |

Reward qualified: True. No optimizer/backward/training was performed; actor retention transitions are NA. This qualification does not establish stable RL improvement, causality or success on the original task.

All64 teacher rows, six planned pool summaries,384 planned case observations and24 checks are retained, including negative/invalid/NA outcomes. Planned rows are not completed observations or384 independent images. Retired P21 training labels fit the frozen v calibration before fresh readouts. Fresh labels never entered either signal; fresh accuracy scoring requires complete worker closure.

Complete GPU-worker lifetime: 4.635694h this round; 38.161002/144h cumulative; 105.838998h remaining. CPU resource cost is separate; no budget reset or archive double counting.

Private identifiers, medical text/options/labels/images, teacher targets/readout text, checkpoints/full logs and private paths are omitted. Future training requires a separately frozen complete protocol and budget admission. Public delivery requires effective-proxy push and final remote SHA/anonymous access verification.
