# Continual RL adaptation for medical MLLMs

This repository implements resumable, independently scored continual test-time adaptation for Qwen2.5-VL-3B-Instruct. It extends the original single-update SPINE diagnostic with persistent FP32 master weights and AdamW state, a fixed reference model, state/RNG checkpoints, and four methods on the same fixed stream. SPINE and the matched TTRL comparator are independent implementations; author-code equivalence is not established.

## P3: two-seed reward/format replication (2026-10-07)

The completed replication reuses the 16 observed stream groups with rollout seeds 43 and 44. TTRL history gain changes sign (−6.25/+18.75 pp), and the SPINE–TTRL ranking reverses (+6.25/−12.50 pp before). SPINE current-case scores rise in both seeds, but this small reused sample does not establish superiority. Across four RL trajectories, 250/274 positive-reward rollouts are parsed wrong and high-vote groups are correct in 0/13 observations. Old P2 actors preserve the same five correct groups under a separate legal-option protocol, supporting protocol sensitivity without proving knowledge retention.

Read the [P3 report](reports/p3_report.md), [protocol](reports/p3_protocol.md), [main CSV](reports/p3_main_results.csv), [reward audit](reports/p3_reward_audit.csv), [format/readout audit](reports/p3_format_audit.md), [probe CSV](reports/p3_probe_results.csv), [aggregate JSON](reports/p3_results.json), [resources](reports/p3_budget_and_resources.json) and [reproduction guide](reports/p3_reproduce.md). All six new runs completed and all owned GPU jobs ended. P2/main history and reports are preserved.

## P2: original one-seed experiment

The completed P2 development experiment uses **16 stream cases, 16 held-out probe groups, and seed 42**. The official dev split had only five questions, so the user authorized a fixed test-derived development subset that is now retired from future final-test use. Sample count, order and configurations were fixed before correctness scoring.

| Method | Greedy before | Greedy after | Frozen SC-8 vote | Historical gain |
|---|---:|---:|---:|---:|
| Frozen greedy | 2/16 (12.50%) | N/A | N/A | reference |
| Frozen SC-8 | N/A | N/A | 1/16 (6.25%) | N/A |
| Continual TTRL | 5/16 (31.25%) | 5/16 (31.25%) | N/A | +18.75 pp |
| Continual SPINE | 3/16 (18.75%) | 3/16 (18.75%) | N/A | +6.25 pp |

Historical gains correspond to three additional correctly selected options for TTRL and one for SPINE, with no originally correct frozen prediction lost. Current-case TTRL updates correct one case and harm one; SPINE has no correctness transitions. Neither has a net current-case gain.

Probe correctness remains 3/16 for TTRL. SPINE goes from 3/16 to 1/16 at the midpoint and 2/16 at the endpoint. Its lost correct cases become unparseable outputs, so this shows lower valid-answer/MCQ retention without establishing medical-knowledge forgetting. TTRL's stream parse rate improves from 14/16 to 16/16 with unchanged current-case accuracy.

These are engineering-scale results from one dependent trajectory. They do not establish statistical significance, general superiority, patient-level isolation, real domain adaptation, or a reproduction of published accuracy. Eight-sample voting and RL are not matched in total compute. Exact-count tables replace trend plots at this sample size.

The [full report](reports/continual_20261005.md) contains paired effects, online block/cumulative tables, probe changes, reward failure diagnostics, optimization/drift measurements, costs, failures and evidence limits. The machine-readable [aggregate results](reports/continual_results.json), [checkpoint checks](reports/continual_checkpoint_check.json) and [completion record](reports/continual_completion.json) are public. CSV tables cover [accuracy](reports/continual_accuracy.csv), [online blocks](reports/continual_online_blocks.csv), [probes](reports/continual_probe.csv), [reward diagnostics](reports/continual_reward_audit.csv), [parse transitions](reports/continual_parse_transitions.csv) and [resources](reports/continual_resources.csv).

Continuous ten-step versus fresh-process five-plus-five tiny-model recovery passed with identical complete state and outputs. Actual-model two-step versus one-plus-one recovery passed with identical FP32 masters, Adam state, buffers, RNG and stored outputs after setting one CPU intra-op thread. The earlier sixteen-thread output discrepancy and deterministic CUDA histogram failure are retained in the [engineering record](reports/continual_engineering.json). All eight main GPU jobs completed successfully; all task-owned experiment/controller processes ended. Full RL checkpoints are about 45 GB each and remain on allocated private storage.

Use the [reproduction guide](reports/continual_reproduction.md) for acceptance, bounded startup, restart and independent scoring. Read the [locked protocol](reports/continual_protocol.md) and [pre-score budget decision](reports/continual_budget.json) before interpreting the results. Main training used commit `7bf769d420bcb63004295e0d1900e5df6467c2d5`; the post-prediction evaluator audit used `6064999c7d3279d0f25bf870e7a3802eed00d287` without changing the original scores.

The key entries are [the continual trainer](implementation/continual.py), [independent evaluator](implementation/evaluate.py), [state handling](implementation/state.py), [CPU acceptance tests](implementation/test_continual.py), [actual-model recovery comparison](scripts/compare_recovery.py), and [bounded controller](scripts/supervise.py). The four method configurations are [A](configs/p2_a.json), [B](configs/p2_b.json), [C](configs/p2_c.json), and [D](configs/p2_d.json).

The original single-update driver, pilot configuration and results remain available: [pilot report](reports/pilot_20261004.md), [pilot results](reports/pilot_results.json), and [historical reproduction instructions](reports/pilot_reproduction.md). The tested package snapshot is [recorded here](metadata/gpu_environment.freeze.txt). No adapters, quantization, vision freezing, reduced rollout groups or shorter production output cap were introduced.

Raw medical images, questions, completions, labels, manifests, full logs, credentials, private paths, model files and checkpoints are excluded from the public repository. The evaluator alone reads labels after predictions close; this is data-flow separation, not an operating-system security sandbox.
