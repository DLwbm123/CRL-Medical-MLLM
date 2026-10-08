# P7 stronger fixed-base KL anchor: paired development experiment

Preregistered after P6 fully closed, sealed and publicly delivered at commit e820756cc01a895568592ea1527adfffe656c5b6, before P7 implementation or outputs. The authorization remains24 cumulative GPU-process hours. P4 used28.726142450002953 seconds, P5 used21744.03255091561 and P6 used21149.256308959797. Cumulative42922.01500232541; remaining43477.98499767459 seconds. No budget reset; earlier branches, workspaces and wave deadlines remain unchanged.

## Grounding and single change

P6 half-step candidate strict historical counts4/3/1 versus controls5/2/3 failed the fixed criterion. Candidate retention was2/1/3 of the same initial3 correct probes. Smaller steps reduced average full-vocabulary KL on sampled prefixes but did not consistently protect strict outputs or historical accuracy. Majority rewards were wrong for77.27–89.47% of positively rewarded responses across the six arms.

P6 original-step mean weighted KL terms were0.0000313278,0.0000192181,0.0000238735; mean entropy-band terms were0.00115612,0.00126871,0.00124306. These scalar loss values do not establish relative gradient influence or a causal diagnosis of probe damage. They motivate one bounded hypothesis: a stronger explicit anchor may constrain accumulated distribution drift under noisy rewards more directly than reducing every update.

Candidate **kl_coefficient1.0**, control **kl_coefficient0.01**. This one factor100 is chosen once, with no sweep or intermediate result selection. Both arms restore/retain original learning_rate1e-6, temperature0.7/top-p0.95 and majority reward. P5 sampler alignment and P6 half-step are not combined with this change. The existing exact forward KL to a separate frozen original base is computed on selected generated tokens and current response prefixes; its masks and normalizations are unchanged. This does not repair reward correctness or constrain every possible probe/prefix. At initial base equality the KL gradient is zero. No medical correctness or positive-result guarantee is implied.

Everything else is fixed: original Qwen2.5-VL-3B revision, prompt/strict parser, eight rollouts,2048 cap, full vision/language parameters, SPINE token mask and entropy band, one CPU thread, FP32 optimizer masters/moments, greedy primary/probe generation, fixed reference and one update/group, state/RNG/checkpoint rules. No added labels, data, model, GPU or paid resource.

## Matrix, scope and success

Fresh rollout seeds **54,55,56** in both arms, independent initializations from the original base. Order **m54,v54,m55,v55,m56,v56**, m=original anchor, v=strong anchor. Exactly16 previously observed stream groups and16 previously observed probe groups, orderseed42, probes0/16, one final full-state checkpoint. Complete all six without same-round medical grading, early best-seed selection, identical retry or failed-seed exclusion.

Stable positive development requires every fixed seed simultaneously:
1. candidate original strict-parsed free greedy-before historical correct > frozen2/16;
2. candidate historical correct > same-seed original-setting control;
3. candidate retains all of the same3 initially correct probe groups.

After-only, changed-parser, net probe total or selected-checkpoint gains do not substitute. Publish all before/after/parse counts, all history/current/paired transitions, individual anonymous initial-correct probe outcomes, reward-direction fractions, format diagnostics and full compute/resources. Different rollout seeds on reused small groups do not establish independent generalization.

## Checks, time and budget

Reuse the existing objective, neutral entry, supervisor and sealed offline evaluator. Only add exact KL-source/configuration guards and P7 routing. Meaningful CPU checks must cover permitted0.01/1.0, invalid rates/sampler/reward/KL/seed/source, the coefficient's actual loss and gradient scaling on a synthetic nonzero-KL tensor, and fixed three-seed success/retention. Inherited label-free state/RNG and source guards must pass. No new real-model probability experiment is needed: both arms retain the original documented transformed-sampler/raw-loss mismatch, with no strict on-policy claim.

P7 is a separately frozen third wave. GPU stop is preparation UTC plus10.5h; wall deadline plus12h, capped to cumulative allowance where necessary; per worker max7200 seconds. Its preparation receipt records absolute deadlines before launch. The remaining43477.98499767459-second cumulative authorization covers this10.5h GPU bound with over1.5h margin. All GPU worker lifetime including loading, CPU optimization, probe generation and saving counts. Conservative forecast remains6*3966=23796 seconds;20% reserve28555.2. Admit only if both own round time and cumulative allowance cover that reserve. No deadline moves or budget reset.

Only approved physical GPU4/UUID, existing environment/model. Fresh preflight checks actual mounted prescribed storage, read/write probe,400GB free reserve,48,000MiB GPU free and100GB CPU available. Do not alter other tasks. Verify neutral parent/controller/worker and GPU argv, recorded PID/start ticks. Isolated executed source is immutable.

The detached pipeline owns all six, records failures/whole lifetimes, verifies owned sessions/GPU workers ended, validates and seals the entire matrix, then reads only the already observed development labels and scores once. Partial/failed work cannot satisfy the fixed criterion and must remain in accounting and delivery. Hard budget/time stops override completing the matrix.

## Monitoring and delivery

Update existing hourly heartbeat crl and private handoff to P7. One bounded check per hour, quiet on unchanged/nonactionable states. Preserve all round ledger costs in optimization_control.json. Publish source/configs and complete sanitized result/report/compute receipt to codex/p7-reference-anchor via the required effective proxy; verify final native Git commit and anonymous access while preserving main and P2/P3/P4/P5/P6 refs. Raw medical inputs/text/IDs/labels/checkpoints/full logs/private paths/secrets remain private.

Stop new compute at24 cumulative GPU-process hours and report honestly. Disable this monitor only after the fixed stable positive criterion and public delivery are both complete. Any further change requires a new grounded freeze before outputs. Reproduction uses modes p7_prepare, p4_test, p7_pipeline and p7_score, configs p7_<arm><seed>.json, one main-plan.json and no reference gate. Bind paths privately as in p4_reproduce.md.
