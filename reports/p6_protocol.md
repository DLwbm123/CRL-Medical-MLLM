# P6 half-step original SPINE: paired development experiment

Preregistered after P5 closed and its complete negative result was publicly delivered, before P6 implementation or GPU outputs. P4/P5 and older branches/workspaces remain immutable. Total authorization is still24 cumulative GPU-process hours. P4 used28.726142450002953 seconds and P5 used21744.03255091561 seconds; cumulative21772.758693365613, remaining64627.24130663439 seconds. No budget reset.

## Evidence and one change

P5 sampler alignment did not establish stable improvement: raw-softmax versus original strict historical correct counts were4/3,3/3,3/5 for seeds48/49/50. Raw-softmax seed49 lost two of the same initially correct probes to unparseable outputs. Original-sampler seed50 also lost one initial correct probe despite a net final total of3. Majority rewards remain wrong for most positively rewarded responses. Neither high vote counts nor the failed P4 legal pseudo-target provides a validated correctness signal.

The next limited hypothesis is that a smaller update step can retain useful historical adaptation while reducing output damage under noisy majority reward. Candidate **learning_rate5e-7**, control **learning_rate1e-6**. The factor1/2 is fixed once; no sweep or output-guided rate selection. Both arms use the original sampler temperature0.7/top-p0.95 and original SPINE objective. P5's candidate sampling change is not combined with this one. A smaller step does not repair clinical reward correctness and may eliminate gains; success is not promised.

Everything else remains fixed: same frozen Qwen2.5-VL-3B revision, original prompt/parser, eight rollouts,2048 cap, full vision/language training, SPINE mask/band/KL, one CPU thread, FP32 optimizer masters/moments, greedy primary and probe decoding, original reference, one update per group, state/RNG rules. No true-label training or medical data expansion.

## Fixed scope and success

Fresh rollout seeds **51,52,53** in both arms. Matrix order **m51,v51,m52,v52,m53,v53**; m=original step, v=half step. Independently initialize every trajectory from the original base. Reuse exactly16 observed stream and16 observed probe groups, orderseed42. No new held-out data or final-test retirement.

Every trajectory runs16 groups, probes0/16 and one final full-state checkpoint. Complete all six without looking at same-round medical grades; no selected seed/checkpoint, identical restart or failed-seed exclusion.

Stable positive development requires all three fixed distinct seeds simultaneously:
1. candidate original strict-parsed greedy-before history correct > frozen2/16;
2. candidate correct > same-seed original-setting control;
3. final candidate retains every one of the same3 initially correct probe groups.

No changed parser/readout, after-only gain or equal net probe total can satisfy this. Report all before/after/parse counts, history/current/paired gain and harm transitions, individual initial-probe status, complete reward direction and resources. Reused small groups do not establish independent generalization.

## Implementation checks, accounting and bounds

Reuse the neutral entry, supervisor and sealed CPU evaluator. Add only exact declared learning-rate-source guards; reject altered sampler/reward/KL/seed/rate. Run the inherited CPU engineering checks plus a meaningful guard check for both rates, invalid changes, and fixed51/52/53 success. Actual-model raw-sampler equality checks are inapplicable because both P6 arms retain the original transformed sampler; the prior documented sampling/loss mismatch remains. No strict on-policy claim.

P6 is a separately bounded second wave after the first wave has fully ended. Do not move P4/P5's earlier deadlines. GPU stop is P6 preparation UTC plus10.5h; wall limit plus12h; per worker max7200 seconds. Count every new GPU worker's whole lifetime, including model loading, CPU optimization and saving, against the same remaining64627.24130663439 seconds. Use the conservative P5 maximum rounded to3966 seconds per job: six-job forecast23796 seconds,20% reserve28555.2 seconds. Both round time and cumulative allowance must cover this before launch. Absolute and cumulative hard stops override completion.

Only existing physical GPU4/UUID, existing model and environment. Check free VRAM, actual mounted prescribed storage, write/read probe and400GB reserve before launch. No other-task process changes, added GPU or paid resources. Use isolated source and neutral argv/environment; verify main/controller/worker PID/start ticks and GPU argv.

The detached pipeline owns the entire matrix, verifies all owned sessions ended, seals configurations/predictions/probes before reading previously observed labels, then scores once. A partial or failed matrix cannot establish fixed three-seed success. Preserve all failed/partial receipts; no unchanged retry.

## Monitoring and public delivery

Update existing hourly heartbeat crl to this protocol and private handoff. One bounded check per hour; quiet on unchanged/nonactionable state. All round compute stays in optimization_control.json with no reset. After actual completion publish source/config/full outcomes/report/compute receipt to codex/p6-half-step using the required proxy; verify final native Git SHA and anonymous report access and preserve all older refs. Raw medical inputs/text/IDs/labels/checkpoints/logs/private paths/credentials stay private. Stop new compute when24h budget ends; disable monitoring only after fixed stable positive plus public delivery. Any further hypothesis must be individually grounded and locked before its outputs.

Reproduction uses the inherited neutral environment, modes p6_prepare, p4_test, p6_pipeline and p6_score, six configs p6_<arm><seed>.json, one main-plan.json, and no reference plan. Paths remain private environment bindings. See p4_reproduce.md for the isolation and entry pattern.
