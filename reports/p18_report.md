# P18 paired numerical precision diagnostic

Status: numerical_checks_negative. Executed source: 8a480647b206a2d7b1a7569d31dcfcc52a976080.

The same64 P17 SLAKE training image groups were reused:32 calibration and32 verification. No annotation, correctness score or teacher reward was read. Three new seeds81/82/83 compare BF16 and FP32 at fixed unwarped four-token generation. Both precisions retain the original absolute log-probability error limit0.1; the BF16 control records failures without changing its production guard. The sampled token paths can differ between precisions, so this is not an identical-token causal attribution.

| Seed | Precision | Status | Passed /64 | Max absolute log-p difference |
| --- | --- | --- | --- | --- |
| 81 | bfloat16 | completed | 57 | 0.1586611270904541 |
| 81 | float32 | completed | 19 | 4.438588619232178 |
| 82 | bfloat16 | completed | 58 | 0.15638303756713867 |
| 82 | float32 | completed | 17 | 1.1205940246582031 |
| 83 | bfloat16 | completed | 54 | 0.1255779266357422 |
| 83 | float32 | completed | 19 | 1.4544124603271484 |

All six planned worker outcomes and384 planned measurements, including failed and unstarted NA, are retained. Repeated measurements refer to64 image groups. No optimizer/backward/weight update occurred; all retention transitions are NA. Numerical checks do not establish reward quality, RL improvement, independent medical generalization or a training-cost estimate. No training is automatically launched.

Whole GPU-worker lifetimes: 0.187069h this round; 23.282191/72h cumulative; 48.717809h remaining. No budget reset or archive double counting.

Medical identifiers, questions/options/labels/images, token/readout text, checkpoints/full logs, private paths and keys are omitted. Public completion requires a proxy push, remote SHA and anonymous access verification.
