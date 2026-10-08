# P6 completed half-step experiment: stable criterion not met

All six declared trajectories completed, without failed workers or retries. Half-step learning rate5e-7 produced primary counts4,3,1; paired original1e-6 produced5,2,3. The frozen baseline is2/16. No candidate seed meets all three conditions. Seed51 is below its control and loses probe B; seed52 gains history but loses A and B; seed53 preserves all three probes but is below both frozen baseline and control. The preregistered all-three-seed success is false.

## Scope and executed source

Six independent initializations, seeds51/52/53, the same16 known stream groups and16 known probe groups, orderseed42. Control m uses learning rate1e-6; candidate v uses5e-7. Both retain original SPINE majority reward, temperature0.7/top-p0.95, full parameters, original strict parser, greedy primary/probe generation and all other locked settings. Executed source and evaluator: b8c4c63b851a7eb901cbd8873c5fd310b786ce51. See p6_protocol.md and p4_reproduce.md.

## All outcomes

| Seed | Arm | Before /16 | After /16 | Parsed before/after | Probe final /16 | Original correct retained /3 |
| --- | --- | --- | --- | --- | --- | --- |
| 51 | m | 5 | 5 | 12/13 | 3 | 3 |
| 51 | v | 4 | 2 | 14/12 | 2 | 2 |
| 52 | m | 2 | 4 | 11/13 | 2 | 2 |
| 52 | v | 3 | 5 | 13/14 | 1 | 1 |
| 53 | m | 3 | 3 | 12/12 | 1 | 1 |
| 53 | v | 1 | 4 | 13/12 | 3 | 3 |

Before denotes each group's greedy prediction before its update and measures historical adaptation. After denotes the current group's prediction after its update. No after-only result, changed parser or selected checkpoint substitutes for the primary. CSVs include all history/current/paired transitions and gains/harms; p6_results.json retains the complete aggregate diagnostics.

## Individual retention of the same initial correct probes

| Seed | Arm | Probe A | Probe B | Probe C |
| --- | --- | --- | --- | --- |
| 51 | m | correct | correct | correct |
| 51 | v | correct | unparseable | correct |
| 52 | m | unparseable | correct | correct |
| 52 | v | parsed wrong | unparseable | correct |
| 53 | m | unparseable | unparseable | correct |
| 53 | v | correct | correct | correct |

A/B/C are anonymous ordinal aliases of the same three initially correct probe groups. They expose no original IDs, choices, text or labels. Candidate seed51 B becomes unparseable. Candidate seed52 A becomes parsed wrong and B becomes unparseable. Control seed52 A and control seed53 A/B become unparseable. Every original correct probe outcome is included above and in the CSV. Unparseability establishes strict-output failure, not a medical knowledge diagnosis.

## Reward-direction audit

| Seed | Arm | Correct response negative advantage | Wrong response positive reward | Vote correct /16 |
| --- | --- | --- | --- | --- |
| 51 | m | 14/22 | 59/67 | 2/16 |
| 51 | v | 8/23 | 51/66 | 4/16 |
| 52 | m | 9/17 | 68/76 | 2/16 |
| 52 | v | 14/24 | 58/68 | 3/16 |
| 53 | m | 6/17 | 63/74 | 3/16 |
| 53 | v | 8/16 | 62/70 | 2/16 |

Response-level ratios count repeated sampled responses, not independent patients. Wrong majority rewards remain common in both arms. Halving the step does not validate clinical reward correctness or establish a causal explanation of an individual loss.

## Closure, numerical check and resources

All six workers exited0, completed16 optimizer updates, retained probes0/16 and one final full state. Owned sessions and GPU workers ended before scoring. Configurations, primary parsing, declared rewards, frozen initial probes, source and reference integrity passed the offline checks. Predictions were sealed before the main development label read.

Both arms retain the original transformed sampler with raw-softmax policy loss. Its documented sampling/loss mismatch remains; raw-sampler equality checks are inapplicable here and no strict on-policy claim is made. Exact behavior recomputation and fixed-reference checks passed.

P6 consumed 21149.256309 whole GPU-worker seconds (5.8748 hours). Including P4 and P5, cumulative use is 42922.015002 seconds (11.9228 hours); remaining authorization is 43477.984998 seconds (12.0772 hours). Model loading, CPU optimization and checkpoint saving are counted. p6_compute_receipt.json and p6_resources.csv record every arm; full phase timings remain in p6_results.json.

## Limits and public boundary

This is development on reused, previously observed small groups. Different rollout seeds do not create independent clinical samples or establish generalization. There was no held-out access, true-label training, new model/GPU or budget reset. All negative outcomes remain reported. No raw medical data, images/text/IDs/labels, checkpoints, full logs, private paths or credentials are included.
