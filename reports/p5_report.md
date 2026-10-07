# P5 completed sampling-alignment experiment: stable criterion not met

All six declared trajectories completed, without failed workers or retries. Raw-softmax sampling produced primary counts 4,3,3; paired original sampling produced 3,3,5. The frozen baseline is 2/16. Only seed48 meets all three conditions. Seed49 ties its control and loses two initially correct probes; seed50 is below its control. The preregistered all-three-seed success is false.

## Scope and executed source

Six independent initializations, seeds48/49/50, the same16 known stream groups and16 known probe groups, orderseed42. Control m uses temperature0.7/top-p0.95; candidate v uses temperature1/top-p1. Both retain original SPINE majority reward, full parameters, learning rate1e-6, original strict parser, greedy primary/probe generation and all other locked settings. Executed source and evaluator: e844f23e840b5f7decdd891b696637fc155c204e. See p5_protocol.md and p4_reproduce.md.

## All outcomes

| Seed | Arm | Before /16 | After /16 | Parsed before/after | Probe final /16 | Original correct retained /3 |
| --- | --- | --- | --- | --- | --- | --- |
| 48 | m | 3 | 3 | 13/13 | 4 | 3 |
| 48 | v | 4 | 2 | 12/13 | 4 | 3 |
| 49 | m | 3 | 3 | 12/12 | 3 | 3 |
| 49 | v | 3 | 4 | 13/12 | 1 | 1 |
| 50 | m | 5 | 3 | 14/13 | 3 | 2 |
| 50 | v | 3 | 2 | 10/11 | 3 | 3 |

Before denotes each group's greedy prediction before its update and measures historical adaptation. After denotes the current group's prediction after its update. No after-only result, changed parser or selected checkpoint substitutes for the primary. CSVs include all history/current/paired transitions and gains/harms; p5_results.json retains the complete aggregate diagnostics.

## Individual retention of the same initial correct probes

| Seed | Arm | Probe A | Probe B | Probe C |
| --- | --- | --- | --- | --- |
| 48 | m | correct | correct | correct |
| 48 | v | correct | correct | correct |
| 49 | m | correct | correct | correct |
| 49 | v | unparseable | unparseable | correct |
| 50 | m | correct | parsed wrong | correct |
| 50 | v | correct | correct | correct |

A/B/C are anonymous ordinal aliases of the same three initially correct probe groups. They expose no original IDs, choices, text or labels. Seed49 candidate A and B become unparseable. Original-sampler seed50 loses B to a parsed wrong answer and gains a different correct group, so its net total3 does not imply full retention. Unparseability establishes strict-output failure, not a medical knowledge diagnosis.

## Reward-direction audit

| Seed | Arm | Correct response negative advantage | Wrong response positive reward | Vote correct /16 |
| --- | --- | --- | --- | --- |
| 48 | m | 11/19 | 66/74 | 2/16 |
| 48 | v | 8/18 | 55/65 | 2/16 |
| 49 | m | 15/22 | 60/67 | 1/16 |
| 49 | v | 10/16 | 46/52 | 2/16 |
| 50 | m | 10/15 | 66/71 | 1/16 |
| 50 | v | 8/13 | 58/63 | 1/16 |

Response-level ratios count repeated sampled responses, not independent patients. Wrong majority rewards remain common in both arms. Sampling alignment does not validate their clinical correctness or establish a causal explanation of an individual loss.

## Closure, numerical check and resources

All six workers exited0, completed16 optimizer updates, retained probes0/16 and one final full state. Owned sessions and GPU workers ended before scoring. Configurations, primary parsing, declared rewards, frozen initial probes, source and reference integrity passed the offline checks. Predictions were sealed before the main development label read.

The candidate four-token actual-model probability checks passed at BF16 tolerance0.1: maximum log-probability differences0.03994596,0.03994596,0.00915903. Production sampling matched the check configuration. This short-prefix check does not establish consistency for every sequence.

P5 consumed 21744.032551 whole GPU-worker seconds (6.0400 hours). Including P4, cumulative use is 21772.758693 seconds (6.0480 hours); remaining authorization is 64627.241307 seconds (17.9520 hours). Model loading, CPU optimization and checkpoint saving are counted. p5_compute_receipt.json and p5_resources.csv record every arm; full phase timings remain in p5_results.json.

## Limits and public boundary

This is development on reused, previously observed small groups. Different rollout seeds do not create independent clinical samples or establish generalization. There was no held-out access, true-label training, new model/GPU or budget reset. All negative outcomes remain reported. No raw medical data, images/text/IDs/labels, checkpoints, full logs, private paths or credentials are included.
