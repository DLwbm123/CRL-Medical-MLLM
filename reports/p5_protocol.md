# P5 sampling/loss alignment: paired development experiment

Preregistered after the delivered P4 diagnostic and before any P5 GPU output. P4's immutable frozen legal pseudo-target scored only1/16 stream groups and worsened both reward-direction criteria; its main matrix was not launched. Its negative result remains in the P4 branch and reports. Total new optimization authorization remains24 cumulative GPU process-hours, with28.72614245 seconds already consumed. No budget renewal.

## One change and its evidence

The production sampler uses temperature0.7/top-p0.95, while policy loss, behavior log-probability and KL use raw untempered full-softmax probabilities. This documented mismatch prevents treating the existing sampled updates as validated strict on-policy updates. The next hypothesis removes these sampling transforms: candidate temperature1, top-p1, top-k0, repetition penalty1. Original settings remain the paired control. Temperature and top-p are jointly the one mechanism under test: make the sampler match the existing raw-softmax loss. No true-label reward or frozen legal pseudo-target enters either arm.

Original SPINE binary majority reward, mask, band, KL,8 fresh responses,2048 cap, original prompt, strict parser, full vision/language training, optimizer masters/moments, one CPU thread and fixed original reference stay the same. Greedy primary/probe decoding is unchanged. This does not establish that majority answers are clinically correct; P3's misleading reward remains a limitation, and the change can yield a negative result. It is not a parameter sweep or a claimed author implementation reproduction.

## Scope and fixed comparison

Reuse exactly16 known stream and16 known probe groups, orderseed42. No new final-test retirement or patient-independence claim. Fresh rollout seeds **48,49,50** are fixed for both arms. Matrix order **m48,v48,m49,v49,m50,v50**, all initialized independently from the original frozen base; m=original sampler, v=raw-softmax sampler. No same-round seed/grade peek, checkpoint selection or restart. Every trajectory runs16 groups, probes0/16, one final full state.

Primary remains strict-parsed reasoning/free-generation greedy-before. Stable positive development success requires, in **all three distinct fixed seeds**:
1. candidate correct count strictly above frozen greedy2/16;
2. candidate correct count strictly above its same-seed original-sampler control;
3. final candidate probe preserves all the same3 originally correct probe groups.

Report both arms' before/after accuracy and parse rates, paired history/current gains and harms, complete probe transitions and reward-direction numerators/denominators. A changed readout, chosen seed, after-only gain or equal net probe total cannot meet success. Reused small development groups do not establish generalization.

## Checks and bounded execution

Existing CPU guards must still reject undeclared changes. Add a guard for the exact declared sampler pairs and distinct seed sets. Every candidate also runs the inherited four-token actual-model generation/teacher-forcing probability check before adaptation, with preserved RNG/modes and BF16 tolerance0.1. Its production sampling configuration must match that check. This checks a short actual-model prefix and the configured transformations, not every rollout or bitwise kernel equivalence. Full output cap stays2048; no new labels enter training.

Use an isolated source workspace and the inherited neutral entry/supervisor. No reference gate in P5, since no reference pseudo-target is used. One six-job main plan; each trajectory max7200s. Both supervisor and trainer enforce the original first-wave absolute GPU stop at preparation start2026-10-07T15:10:51.779889+00:00 plus10.5h; hard first-wave wall limit plus12h. This earlier bound is retained rather than granting a fresh12h for this round. Count every GPU worker's whole lifetime, including CPU optimization/checkpoint time. Before launch, remaining cumulative GPU budget is86371.27385755 seconds; measured six-trajectory estimate21324s plus20% reserve25588.8s must fit both remaining round time and total allowance. P5 sampling can change runtime; these are forecasts, and hard stops take precedence over a complete matrix.

Verify GPU4's existing UUID, actual free VRAM, mounted prescribed storage, write probe and at least400GB state reserve. Preserve prior campaigns and unrelated jobs. No new GPU, paid resources, extra data or model. PID/start-tick/session and GPU argv checks must use the existing neutral launcher policy.

The detached pipeline runs the whole fixed matrix, verifies owned sessions ended, seals all configurations/predictions/probes, and only then invokes independent CPU scoring with previously observed development labels. Any failed/partial trajectory means this fixed three-seed success has not been established; keep the partial and failed receipt. No identical automatic restart or deadline/budget reset.

## Hourly continuation and delivery

Reuse the one active hourly heartbeat idcrl. Private optimization_control.json points to the active campaign and contains every earlier round's actual compute and publication receipt. Each hourly turn checks once, remains quiet on unchanged state, and reports completion, meaningful failure, required action, budget exhaustion or the fixed positive criterion.

After actual completion, publish sanitized source/configuration/all outcomes/resource counts/report to **codex/p5-sampling-alignment** via the required proxy and verify anonymous access. Retain P4 and earlier branches unchanged. No raw medical data/text/IDs/labels/checkpoints/logs or private paths. If negative, the next grounded change needs new preregistration/control/fresh seeds within the remaining24h total. Stop new compute at budget exhaustion; disable monitoring after stable development success plus public delivery. No guaranteed positive result is promised.

Reproduction follows p4_reproduce.md's neutral environment and isolated-copy approach, using modes p5_prepare, p4_test, p5_pipeline and p5_score, configurations configs/p5_<arm><seed>.json, the main plan above and no reference-plan.json. Set P2_PROBABILITY_CHECK=1 only for each candidate job. P4 files and executed source remain immutable in their prior remote workspace.
