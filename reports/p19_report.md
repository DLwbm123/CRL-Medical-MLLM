# P19 same-token cache-path numerical diagnostic

Status: numerical_checks_passed. Executed source: b005ad1c190c2f41b1e36bfe7e7a5124709fa43f.

Exactly64 previously admitted P17 SLAKE training inputs were reused,32 calibration and32 verification, without reading annotations or reward accuracy. BF16 actor, original prompt/processor and unwarped four-token generation are frozen. Each new seed84/85/86 draws one response per input; that exact response supplies both complete-prefix uncached and manually incremental cached recomputation. The cached path uses the installed model's generation preparation, cache positions, attention-mask growth and vision-prefill handling. Original0.1 tolerance remains unchanged and all384 paired measurements/NA are retained.

| Seed | Recompute path | Status | Passed /64 | Max absolute log-p difference |
| --- | --- | --- | --- | --- |
| 84 | uncached | completed | 59 | 0.2313539981842041 |
| 84 | cached | completed | 64 | 0.0 |
| 85 | uncached | completed | 57 | 0.17197704315185547 |
| 85 | cached | completed | 64 | 0.0 |
| 86 | uncached | completed | 56 | 0.23172378540039062 |
| 86 | cached | completed | 64 | 0.0 |

Only all192 cached measurements within0.1 plus complete worker closure/provenance/coverage/frozen state constitute this numerical diagnostic passing. No production probability guard or RL loss path was changed. Cache-path agreement does not prove medical reward quality, an uncached training loss is valid, RL improvement, or an underlying library bug. No optimizer/backward/weight update/teacher generation/correctness scoring or automatic training occurred. Retention transitions remain NA; repeated observations are64 image groups, not384 independent patients.

Complete GPU-worker lifetime: 0.056978h this round; 23.339170/72h cumulative; 48.660830h remaining. No reset or archived/active double counting.

Private identifiers, medical text/options/labels/images, token/readout text, checkpoints/full logs, private paths and keys are omitted. Public completion requires proxy push, final remote SHA and anonymous access verification. Reward qualification and future RL require separately frozen full scope and real training-cost admission.
