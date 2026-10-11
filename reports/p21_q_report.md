# P21_Q medical reward validation

Final status: reward_qualification_negative. Executed source: 443e63e33f91e9b2eb530c84da75804afa0eb75f.

64 SLAKE English training image groups selected for an intended yes/no task, fixed32 calibration +32 verification; six frozen SC-8 pools at seeds90/91/92 and two sampling distributions. Validation/test image-reference overlaps and conservative thumbnail duplicates were excluded. Patient independence and absence of pretraining overlap remain unverified.

This new binary development task differs from retired MedXpertQA and cannot reverse its negative results or establish independent clinical generalization. Teacher self-reported visual support is a fallible proxy.

| Block / sampler | Accepted /32 | Correct accepted /32 | Six checks passed |
| --- | --- | --- | --- |
| calibration_s | 7 | 5 | 1/6 |
| calibration_t | 7 | 5 | 2/6 |
| verification_s | 10 | 7 | 3/6 |
| verification_t | 10 | 7 | 3/6 |

Reward qualified: False. No optimizer/backward/training was performed; actor retention transitions are NA. This qualification does not establish stable RL improvement, causality or success on the original task.

All64 teacher rows, six planned pool summaries,384 planned case observations and24 checks are retained, including negative/invalid/NA outcomes. Planned rows are not completed observations or384 independent images. No truth entered the reward signal; accuracy scoring requires the complete scope to seal and every worker to end.

Complete GPU-worker lifetime: 6.629423h this round; 33.525308/72h cumulative; 38.474692h remaining. CPU resource cost is separate; no budget reset or archive double counting.

Private identifiers, medical text/options/labels/images, teacher targets/readout text, checkpoints/full logs and private paths are omitted. Future training requires a separately frozen complete protocol and budget admission. Public delivery requires effective-proxy push and final remote SHA/anonymous access verification.
