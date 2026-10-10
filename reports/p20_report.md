# P20 medical reward validation

Final status: reward_qualification_negative. Executed source: f7f68bbfe8078bdfa4c75cdfc9108e64e19a8e47.

64 SLAKE English training image groups selected for an intended yes/no task, fixed32 calibration +32 verification; six frozen SC-8 pools at seeds87/88/89 and two sampling distributions. Validation/test image-reference overlaps and conservative thumbnail duplicates were excluded. Patient independence and absence of pretraining overlap remain unverified.

This new binary development task differs from retired MedXpertQA and cannot reverse its negative results or establish independent clinical generalization. Teacher self-reported visual support is a fallible proxy.

| Block / sampler | Accepted /32 | Correct accepted /32 | Six checks passed |
| --- | --- | --- | --- |
| calibration_s | 11 | 8 | 2/6 |
| calibration_t | 11 | 8 | 2/6 |
| verification_s | 13 | 10 | 3/6 |
| verification_t | 13 | 10 | 4/6 |

Reward qualified: False. No optimizer/backward/training was performed; actor retention transitions are NA. This qualification does not establish stable RL improvement, causality or success on the original task.

All64 teacher rows, six planned pool summaries,384 planned case observations and24 checks are retained, including negative/invalid/NA outcomes. Planned rows are not completed observations or384 independent images. No truth entered the reward signal; accuracy scoring requires the complete scope to seal and every worker to end.

Complete GPU-worker lifetime: 3.556716h this round; 26.895885/72h cumulative; 45.104115h remaining. CPU resource cost is separate; no budget reset or archive double counting.

Private identifiers, medical text/options/labels/images, teacher targets/readout text, checkpoints/full logs and private paths are omitted. Future training requires a separately frozen complete protocol and budget admission. Public delivery requires effective-proxy push and final remote SHA/anonymous access verification.

P20 reuses the exact64 P17 admitted binary training inputs; no annotation selection or new data access occurred. Its sole engineering change is the frozen-pool probability check using P19's same-token incremental cached path. The original0.1 threshold and uncached control observations are retained. The normal Engine probability check and RL loss were not modified; this frozen-pool path is not admitted for training. P19's cached192/192 versus uncached172/192 was numerical evidence only. Teacher, prompt, parsers, six reward conditions and both sampling distributions are unchanged, with fresh128 teacher readouts and six complete SC8 pools. All24 block checks must pass; this still does not establish RL improvement or independent clinical generalization.

## Full negative result and useful partial findings

All seven GPU workers exited0 and all owned sessions ended. The complete128 teacher readouts,3072 candidate answers and384 repeated groups sealed before label scoring. All six cached checks passed the original0.1 tolerance with maximum error0; the uncached controls remained visible. This removes this round's frozen-inference numerical blocker and does not validate gradients, optimizer updates or training loss.

| Block / sampler | Accepted /32 | Accepted precision | Majority wrong-positive % | Teacher wrong-positive % | Reduction pp | Checks passed |
| --- | --- | --- | --- | --- | --- | --- |
| calibration_s | 11 | 72.73% | 18.03 | 14.29 | 3.74 | 2/6 |
| calibration_t | 11 | 72.73% | 22.80 | 15.57 | 7.22 | 2/6 |
| verification_s | 13 | 76.92% | 22.02 | 7.14 | 14.88 | 3/6 |
| verification_t | 13 | 76.92% | 21.28 | 9.20 | 12.08 | 4/6 |

Teacher acceptance was24/64, with18 correct accepted targets. Calibration precision8/11=72.73% failed the frozen75% minimum; verification10/13=76.92% passed precision but failed coverage and correct-target count. Both blocks accepted fewer than16/32 and had fewer than12 correct accepted targets. Verification with the original sampler rescued no correct minority candidate. Thirteen of24 checks failed. The reductions in wrong-positive reward are conditional on the much smaller accepted set and are not accuracy gains from RL.

Per-seed harm remains in the full tables: original-sampler seed88 correct-negative rate increased from5.72% to6.73%, and seed89 from5.71% to7.62%; raw seed89 stayed at6.07%. Lower pooled rates do not establish improvement for every seed. All six teacher pools abstained on40/64 groups, leaving40–43/64 groups with zero advantage. No actor update or retention transition occurred; retention harm is NA, not zero harm.

The label-free closed-output audit found104/128 strictly parsed teacher readouts,22 visual-uncertain judgments,28 clinical-uncertain judgments, five option-order-disagreement groups and no token-cap group. Reasons overlap. These observations identify abstention and output reliability constraints; they do not establish a causal explanation of the negative reward result.

The historical P17 binary-task membership screen used training annotation vocabulary and was not label-blind. P20 reused that already frozen scope without new selection or annotation rereading during preparation. No original MedXpertQA result is reclassified. A new method must earn separate reward and full-training admission; no automatic training follows this negative result.
