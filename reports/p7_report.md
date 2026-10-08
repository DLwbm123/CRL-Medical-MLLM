# P7 completed reference-anchor experiment: stable criterion not met

All six declared trajectories completed, without failed workers or retries. The stronger fixed-base KL coefficient1.0 produced primary counts2,3,5; paired original coefficient0.01 produced3,3,5. The frozen baseline is2/16. No candidate seed meets all three conditions. Seed54 equals the frozen baseline and is below control; seeds55/56 equal their controls, and seed55 also loses the original correct probe B. The preregistered all-three-seed success is false.

## Scope and executed source

Six independent initializations, seeds54/55/56, the same16 known stream groups and16 known probe groups, orderseed42. Control m uses fixed-base KL coefficient0.01; candidate v uses1.0. This coefficient is the sole change. Both retain original SPINE majority reward, learning rate1e-6, temperature0.7/top-p0.95, full parameters, entropy mask/band, original strict parser, greedy primary/probe generation and all other locked settings. Executed source and evaluator: a7e8531387997fca8d4e15cda36fd6eaf0b98ea5. See p7_protocol.md and p4_reproduce.md.

## All outcomes

| Seed | Arm | Before /16 | After /16 | Parsed before/after | Probe final /16 | Original correct retained /3 |
| --- | --- | --- | --- | --- | --- | --- |
| 54 | m | 3 | 4 | 10/13 | 3 | 3 |
| 54 | v | 2 | 3 | 13/12 | 3 | 3 |
| 55 | m | 3 | 3 | 11/13 | 3 | 3 |
| 55 | v | 3 | 4 | 13/11 | 3 | 2 |
| 56 | m | 5 | 4 | 14/13 | 5 | 3 |
| 56 | v | 5 | 5 | 14/14 | 3 | 3 |

Before denotes each group's greedy prediction before its update and measures historical adaptation. After denotes the current group's prediction after its update. No after-only result, changed parser or selected checkpoint substitutes for the primary. CSVs include all history/current/paired transitions and gains/harms; p7_results.json retains the complete aggregate diagnostics.

## Individual retention of the same initial correct probes

| Seed | Arm | Probe A | Probe B | Probe C |
| --- | --- | --- | --- | --- |
| 54 | m | correct | correct | correct |
| 54 | v | correct | correct | correct |
| 55 | m | correct | correct | correct |
| 55 | v | correct | unparseable | correct |
| 56 | m | correct | correct | correct |
| 56 | v | correct | correct | correct |

A/B/C are anonymous ordinal aliases of the same three initially correct probe groups. They expose no original IDs, choices, text or labels. Candidate seed55 B becomes unparseable. Its final total remains3/16 because one other probe becomes correct: total accuracy alone hides the lost original correct probe. The other five trajectories preserve all three original correct probes. Every original correct probe outcome is included above and in the CSV. Unparseability establishes strict-output failure, not a medical knowledge diagnosis.

## Reward-direction audit

| Seed | Arm | Correct response negative advantage | Wrong response positive reward | Vote correct /16 |
| --- | --- | --- | --- | --- |
| 54 | m | 15/27 | 58/70 | 3/16 |
| 54 | v | 17/23 | 58/64 | 2/16 |
| 55 | m | 12/19 | 56/63 | 1/16 |
| 55 | v | 15/15 | 65/65 | 0/16 |
| 56 | m | 5/14 | 68/77 | 2/16 |
| 56 | v | 12/25 | 54/67 | 3/16 |

Response-level ratios count repeated sampled responses, not independent patients. Wrong majority rewards remain common in both arms. A stronger anchor does not validate clinical reward correctness or establish a causal explanation of an individual loss.

## Anchor and optimization diagnostics

Mean exact full-vocabulary KL to the fixed base, measured before updates on each arm's own sampled prefixes, is0.00174151 versus0.00086736 for seed54,0.00089177 versus0.00090169 for seed55, and0.00306600 versus0.00095890 for seed56 (control versus candidate). Candidate KL is lower for two seeds, but these prefixes differ between arms and there is no uniform primary or retention improvement. This is a descriptive diagnostic, not a causal effect estimate. Scalar loss magnitudes do not measure component gradient influence. Full loss components, gradient norms, token fractions and post-update prefix diagnostics remain in p7_results.json.

## Closure, numerical check and resources

All six workers exited0, completed16 optimizer updates, retained probes0/16 and one final full state. Owned sessions and GPU workers ended before scoring. Configurations, primary parsing, declared rewards, frozen initial probes, source and reference integrity passed the offline checks. Predictions were sealed before the main development label read. Before launch, an actual synthetic KL objective and gradient check verified the coefficient scaling; it did not predict clinical efficacy.

Both arms retain the original transformed sampler with raw-softmax policy loss. Its documented sampling/loss mismatch remains; raw-sampler equality checks are inapplicable here and no strict on-policy claim is made. Exact behavior recomputation and fixed-reference checks passed.

P7 consumed 22583.527223 whole GPU-worker seconds (6.2732 hours). Including P4, P5 and P6, cumulative use is 65505.542225 seconds (18.1960 hours); remaining authorization is 20894.457775 seconds (5.8040 hours). Model loading, CPU optimization and checkpoint saving are counted. p7_compute_receipt.json and p7_resources.csv record every arm; full phase timings remain in p7_results.json. Whole-device sampled memory may include other jobs and is distinct from this worker allocated/reserved peaks.

## Next-round budget admission

The remaining 5.8040 hours cannot cover another full six-trajectory matrix at the measured P7 cost of 6.2732 hours, even without reserve. With the existing20% reserve that measured cost requires 7.5278 hours. The previous P6 matrix also consumed 5.8748 hours before reserve. No next GPU round is launched. Authorization is not exhausted or reset; unused time remains. A cheaper implementation has not been measured, and an assumed speedup cannot establish admission. See p7_next_admission.json for the exact decision. Stable positive performance remains unestablished.

## Limits and public boundary

This is development on reused, previously observed small groups. Different rollout seeds do not create independent clinical samples or establish generalization. There was no held-out access, true-label training, new model/GPU or budget reset. All negative outcomes remain reported. No raw medical data, images/text/IDs/labels, checkpoints, full logs, private paths or credentials are included.
