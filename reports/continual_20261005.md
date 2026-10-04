# Continual medical MLLM adaptation: 12-hour development run

This run tests whether label-free updates on earlier cases help later unseen cases. Its primary comparison is greedy-before versus the frozen greedy model on the same stream; greedy-after minus greedy-before measures the effect of the current case's update. The experiment is an independent implementation and an engineering-scale development result, not an author-code reproduction or clinical deployment validation.

All four methods completed the same 16 stream cases, with 16 independent image-reference groups reserved for probes. The table reports correct/total (percentage); gains are percentage points. The vote column describes Frozen SC-8's answer and, for RL, the pre-update rollout consensus used to define rewards. N/A denotes an unperformed prediction mode.

| Method | Greedy before | Greedy after | 8-rollout vote | Historical gain (pp) | Current-case gain (pp) |
| --- | --- | --- | --- | --- | --- |
| Frozen greedy | 2/16 (12.50%) | N/A | N/A | +0.00 | N/A |
| Frozen SC-8 | N/A | N/A | 1/16 (6.25%) | N/A | N/A |
| Continual TTRL | 5/16 (31.25%) | 5/16 (31.25%) | 2/16 (12.50%) | +18.75 | +0.00 |
| Continual SPINE | 3/16 (18.75%) | 3/16 (18.75%) | 1/16 (6.25%) | +6.25 | +0.00 |

The paired decomposition distinguishes correctness changes from parse recovery. All counts use the same 16-case denominator.

| Method | Comparison | Wrong → correct | Correct → wrong | Parsed wrong → correct | Invalid → correct | Correct → parsed wrong | Correct → invalid |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Continual TTRL | Frozen → before | 3 | 0 | 3 | 0 | 0 | 0 |
| Continual TTRL | Before → after | 1 | 1 | 0 | 1 | 1 | 0 |
| Continual SPINE | Frozen → before | 1 | 0 | 1 | 0 | 0 | 0 |
| Continual SPINE | Before → after | 0 | 0 | 0 | 0 | 0 | 0 |
TTRL's historical benefit is three additional correctly selected options, and SPINE's is one; neither loses an originally correct frozen prediction on this stream. All historical corrections change a parsed wrong option into a parsed correct option. For the current case, TTRL gains one correct answer through parse recovery and loses one through a parsed wrong option, leaving accuracy unchanged. SPINE has no correctness transition on the current cases.

TTRL's greedy parse rate rises from 14/16 (87.50%) before to 16/16 (100.00%) after its current-case updates, while accuracy stays at 5/16. SPINE's parse rate falls from 12/16 (75.00%) to 11/16 (68.75%): two outputs become parseable and three become unparseable, without changing correctness. This establishes a distinction between answer format and this MCQ score; it is not a separate assessment of free-text medical reasoning.

The exact online block and cumulative results are below. Before and after counts may be equal despite paired gains and harms within a block.

| Method | Prediction | Stream positions | Block score | Cumulative score |
| --- | --- | --- | --- | --- |
| Frozen greedy | Before | 1–8 | 1/8 (12.50%) | 1/8 (12.50%) |
| Frozen greedy | Before | 9–16 | 1/8 (12.50%) | 2/16 (12.50%) |
| Frozen SC-8 | Vote | 1–8 | 0/8 (0.00%) | 0/8 (0.00%) |
| Frozen SC-8 | Vote | 9–16 | 1/8 (12.50%) | 1/16 (6.25%) |
| Continual TTRL | Before | 1–8 | 3/8 (37.50%) | 3/8 (37.50%) |
| Continual TTRL | Before | 9–16 | 2/8 (25.00%) | 5/16 (31.25%) |
| Continual TTRL | After | 1–8 | 3/8 (37.50%) | 3/8 (37.50%) |
| Continual TTRL | After | 9–16 | 2/8 (25.00%) | 5/16 (31.25%) |
| Continual SPINE | Before | 1–8 | 2/8 (25.00%) | 2/8 (25.00%) |
| Continual SPINE | Before | 9–16 | 1/8 (12.50%) | 3/16 (18.75%) |
| Continual SPINE | After | 1–8 | 2/8 (25.00%) | 2/8 (25.00%) |
| Continual SPINE | After | 9–16 | 1/8 (12.50%) | 3/16 (18.75%) |

Every probe evaluates the underlying actor by greedy decoding. Thus Frozen SC-8's probe is intentionally the same policy measurement as Frozen greedy, not another eight-sample voting evaluation.

| Method | Initial | After 8 updates/cases | After 16 updates/cases | Final change (pp) | Parse counts at 0 → 8 → 16 |
| --- | --- | --- | --- | --- | --- |
| Frozen greedy | 3/16 (18.75%) | 3/16 (18.75%) | 3/16 (18.75%) | +0.00 | 13/16 → 13/16 → 13/16 |
| Frozen SC-8 | 3/16 (18.75%) | 3/16 (18.75%) | 3/16 (18.75%) | +0.00 | 13/16 → 13/16 → 13/16 |
| Continual TTRL | 3/16 (18.75%) | 3/16 (18.75%) | 3/16 (18.75%) | +0.00 | 13/16 → 12/16 → 13/16 |
| Continual SPINE | 3/16 (18.75%) | 1/16 (6.25%) | 2/16 (12.50%) | -6.25 | 13/16 → 9/16 → 11/16 |
All initial probe predictions match exactly, and both frozen methods reproduce them at each cursor. TTRL preserves all three initially correct probe cases. SPINE loses two initially correct probe cases at the midpoint and one at the endpoint; these losses are entirely correct-to-unparseable transitions. No initially correct probe changes into a parsed wrong option. The measured valid-answer/MCQ retention falls, but these data alone do not identify medical-knowledge forgetting. The post-prediction audit did not manually reinterpret unparseable clinical text.

Offline reward diagnostics show that having a correct candidate frequently fails to produce a correct vote. These correctness fields were never available to the trainer.

| Method | Parsed candidates | Tied groups | All invalid | Zero advantage | Mean top votes | Mean vote margin | Correct candidate exists | Correct candidate, wrong vote / all groups | Wrong vote / groups with correct candidate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen SC-8 | 98/128 (76.56%) | 3/16 (18.75%) | 0/16 (0.00%) | 1/16 (6.25%) | 4.125 | 2.875 | 9/16 (56.25%) | 8/16 (50.00%) | 8/9 (88.89%) |
| Continual TTRL | 102/128 (79.69%) | 1/16 (6.25%) | 0/16 (0.00%) | 0/16 (0.00%) | 4.312 | 2.938 | 9/16 (56.25%) | 7/16 (43.75%) | 7/9 (77.78%) |
| Continual SPINE | 95/128 (74.22%) | 2/16 (12.50%) | 0/16 (0.00%) | 0/16 (0.00%) | 3.875 | 2.500 | 11/16 (68.75%) | 10/16 (62.50%) | 10/11 (90.91%) |
For TTRL, agreement of greedy answers with the same pre-update vote increases from 10/16 (62.50%) to 11/16 (68.75%); for SPINE it rises from 7/16 (43.75%) to 8/16 (50.00%). Both have flat current-case accuracy. This is increased greedy alignment with the earlier vote, not a measurement of post-update eight-rollout consensus strength: that extra sampling was not performed.

Mean top-vote counts rise across the two different halves from 3.875 to 4.750 for TTRL and from 3.625 to 4.125 for SPINE. Case composition is confounded with time, so this does not establish that adaptation strengthened consensus. SPINE's vote accuracy nevertheless goes from 1/8 (12.50%) in the first half to 0/8 in the second. In the complete SPINE stream, 10 of 11 groups containing a correct candidate still choose an incorrect consensus. These are limitations of the observed reward signal, not evidence that a replacement reward has been validated.

Both RL methods performed all 16 updates, with persistent master parameters and optimizer moments, no skipped groups, and no zero-advantage updates. Vision and language gradient norms were nonzero at every update. Mean objective components include their configured coefficients; their different selected-token masks mean policy/band values are not directly comparable quality scores.

| Method | Updates | Skips | Zero-advantage updates | Selected tokens | Mean policy term | Mean band term | Mean KL term | Vision gradient norm range | Language gradient norm range |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Continual TTRL | 16 | 0 | 0 | 100.00% | 0.113229 | 0 | 1.65873e-05 | 0.3905–17.8 | 2.121–8.294 |
| Continual SPINE | 16 | 0 | 0 | 32.84% | 0.0111981 | 0.0012168 | 1.84817e-05 | 0.2631–34.32 | 1.617–7.567 |
Every pre-update importance ratio is one and every pre-update clip fraction is zero, as expected for fresh rollouts followed by a single update. The following diagnostics reevaluate only the first stored rollout's prefixes at predeclared cursors; they are not population-wide policy distances.

| Method | Cursor | Response tokens | Exact full-vocabulary KL to fixed base | Mean sampled-token log-ratio to behavior | Max absolute sampled log-ratio | Post-update raw ratio outside clip interval |
| --- | --- | --- | --- | --- | --- | --- |
| Continual TTRL | 1 | 236 | 0.001041953 | 0.001807885 | 0.1789443 | 0.0000% |
| Continual TTRL | 8 | 377 | 0.001004789 | 0.002537159 | 0.1536053 | 0.0000% |
| Continual TTRL | 16 | 82 | 0.004688039 | -0.001783905 | 0.2027359 | 0.0000% |
| Continual SPINE | 1 | 236 | 0.0009921218 | 0.003914703 | 0.1861687 | 0.4237% |
| Continual SPINE | 8 | 271 | 0.0009365804 | 0.0002147627 | 0.1725488 | 0.0000% |
| Continual SPINE | 16 | 378 | 0.0007485276 | -0.0004184214 | 0.2042338 | 0.0000% |
SPINE's first diagnostic has one of 236 raw ratios outside the clipping interval; every other diagnostic has none. This still does not test whether clipping caused a useful effect. Exact KL and sampled-token log-ratio have different references and estimands and must not be conflated.

Measured main-process costs include both segments, model initialization, probes, checkpoint restoration/writing and interpreter overhead. GiB means 2^30 bytes; checkpoint GB means 10^9 bytes.

| Method | Process wall seconds | Peak GPU allocated GiB | Peak GPU reserved GiB | Peak CPU RSS GiB | Checkpoint GB written |
| --- | --- | --- | --- | --- | --- |
| Frozen greedy | 499.81 | 8.53 | 8.68 | 4.67 | 0.000037 |
| Frozen SC-8 | 1387.59 | 8.53 | 8.68 | 4.67 | 0.000037 |
| Continual TTRL | 4278.62 | 27.03 | 28.40 | 82.25 | 90.112964 |
| Continual SPINE | 4353.13 | 27.01 | 28.38 | 82.13 | 90.112964 |
The main controller took 10,520.150 seconds (2.922 hours); its eight worker process durations sum to 10,519.143 seconds. All eight exited successfully. These are wall-clock intervals of GPU-using jobs, not measured GPU-active compute. Per-process SM activity was not reliably available in the device monitoring output. Whole-device memory samples include unrelated jobs and are retained with that warning in the JSON; allocator peaks above describe this worker. Peaks from this 16-case trajectory are not upper bounds for longer answers or other inputs. Shared load and variable storage throughput prevent an algorithmic throughput ranking from these timings.

| Method | Rollouts | Rollout tokens | Before tokens | After tokens | Probe tokens | Rollout cap rate | Before capped/total | After capped/total | Probe capped/total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen greedy | 0 | 0 | 2769 | 0 | 7551 | 0/0 (N/A) | 0/16 | 0/0 | 0/48 |
| Frozen SC-8 | 128 | 23398 | 0 | 0 | 7551 | 1/128 (0.78%) | 0/0 | 0/0 | 0/48 |
| Continual TTRL | 128 | 23968 | 2520 | 2406 | 7773 | 1/128 (0.78%) | 0/16 | 0/16 | 0/48 |
| Continual SPINE | 128 | 23143 | 2715 | 4371 | 10963 | 1/128 (0.78%) | 0/16 | 1/16 | 2/48 |
A 0/0 entry denotes a prediction mode that was not run, not a measured zero rate. Each method performs 48 probe generations (16 cases at three cursors). Production caps remain 2048 tokens for all methods.

| Measured phase, seconds | Frozen greedy | Frozen SC-8 | TTRL | SPINE |
| --- | --- | --- | --- | --- |
| Model/state construction | 18.639 | 17.939 | 81.127 | 82.366 |
| Probe generation | 342.917 | 326.883 | 337.339 | 456.974 |
| Preprocessing | 3.025 | 5.792 | 1.736 | 1.491 |
| Greedy before | 121.747 | 0.000 | 112.244 | 116.885 |
| Eight-rollout sampling | 0.000 | 1027.155 | 1064.590 | 1012.286 |
| Behavior forward | 0.000 | 0.000 | 93.400 | 76.302 |
| Reference forward + transfer | 0.000 | 0.000 | 89.514 | 72.097 |
| Actor forward + statistics + backward | 0.000 | 0.000 | 809.852 | 743.174 |
| Gradient transfer/norm + CPU Adam + actor sync | 0.000 | 0.000 | 1082.305 | 1104.811 |
| Post-update drift diagnostic | 0.000 | 0.000 | 5.561 | 2.933 |
| Greedy after | 0.000 | 0.000 | 107.453 | 183.137 |
| Reference integrity comparison | 0.000 | 0.000 | 17.504 | 15.605 |
| Checkpoint write | 0.028 | 0.011 | 424.047 | 418.451 |
The phase fields are scoped timers, not a complete disjoint partition of process duration. In particular, the log field named backward includes actor forward, loss statistics and backward; CPU optimizer includes transfers, norm calculation, Adam and synchronization. State restoration after model construction took 0.032/0.005/28.212/39.053 seconds for A/B/C/D respectively, and is already included in process wall time.

| Campaign phase | Outcome | GPU jobs | GPU-job wall seconds | CPU jobs | CPU-job wall seconds | Failed jobs |
| --- | --- | --- | --- | --- | --- | --- |
| plan_acceptance | stopped_after_job_failure | 1 | 128.047 | 0 | 0.000 | 1 |
| plan_acceptance2 | stopped_after_job_failure | 6 | 3056.991 | 1 | 33.446 | 1 |
| plan_acceptance3 | all_planned_jobs_completed | 6 | 3647.096 | 1 | 7.569 | 0 |
| plan_main | all_planned_jobs_completed | 8 | 10519.143 | 0 | 0.000 | 0 |
The failed deterministic-histogram and sixteen-thread recovery attempts remain charged to this campaign; successful diagnostic work performed before the failed comparison is not discarded from its cost. Initial checkpoint validation and offline scoring took 11.214 and 2.505 CPU-job seconds; the later parse-transition audit took 2.056 seconds. CPU tests, coding, environment setup, report assembly and publication overhead are accounted for by the overall elapsed clock, not claimed as GPU time.

The final checkpoint of each RL method is 45,056,481,971 bytes, with two complete states retained per main run (about 90.113 GB). Main checkpoint writes total 180,226,001,120 bytes including frozen states. The private campaign storage snapshot after completion contains 484 files, 540,681,473,999 logical bytes (about 540.681 GB), and 540,682,702,848 allocated bytes; it includes the retained failed engineering checkpoints but excludes the source dataset/model cache. Available filesystem space was 18,057,572,990,976 bytes at that check.

The protocol was locked before any correctness scoring. Official dev contained only five examples, so the user authorized a fixed test-derived development subset; the stream and probe examples used here are retired from future final-test use. The candidate pool was selected with seed 42 using connected components of image references, excluding the prior six-image pilot example and the engineering dev groups. The final stream has 16 groups and the held-out probe has 16 groups. This establishes image-reference separation, not verified patient separation or duplicate-image detection. The stream is a fixed random order, not a validated nonstationary or cross-hospital sequence.

The original target was 64 stream cases. After complete actual-model acceptance and throughput measurement, the 32-case forecast required 7.31 hours, exceeding 70% of the remaining approximately 8.01 GPU-job hours. The selected 16-case plan forecast 4.39 hours and retained about 45% reserve. Sample count was decided before scoring. The exact budget components are in [the budget record](continual_budget.json). All four methods advance to eight cases, save, and then restart their own state to reach sixteen. No method inherits another method's actor, optimizer or RNG. No extra seed, episodic comparator, longer stream or hyperparameter search was run.

Qwen2.5-VL-3B-Instruct uses the same pinned revision and pilot learning rate of 1e-6. The RL configurations train all language and vision parameters, retain CPU FP32 master weights and native AdamW moments/step counters, and maintain a separate nontrainable fixed base reference. There are eight fresh rollouts per case, temperature 0.7, top-p 0.95, top-k 0 and a 2048-token output limit, followed by at most one optimizer step. Loss log-probabilities remain raw and untempered; there is no importance correction for the sampling transformations. Strict on-policy equivalence is therefore not claimed.

TTRL is an implementation-matched comparator using all response tokens and no SPINE entropy band. SPINE retains the independently implemented selection and band objective. These settings and the answer parser have not been verified as the authors' implementations. First-step ratio one, zero clipping fraction or zero initial KL do not establish that clipping or the fixed anchor improves outcomes. Post-update exact full-vocabulary KL to the fixed base and sampled-token log-ratios to behavior are different quantities and are reported separately. Frozen SC-8 and RL do not have matched total compute.

The trainer accepts only the predefined label-free input schema. All predictions and checkpoints are written before a separate evaluator opens labels. This is data-flow separation, not an operating-system security sandbox. Probe predictions use greedy decoding for every underlying actor, including Frozen SC-8; they preserve model modes and Python/NumPy/CPU/CUDA RNG and never update parameters. Probe correctness does not select settings or checkpoints. Invalid parsed answers remain incorrect entries in accuracy denominators. All-invalid rollout groups skip the update; equal-reward groups are marked as lacking group RL advantage even if band/KL terms or Adam momentum still move parameters.

Engineering acceptance passed before the main stream. A tiny model's continuous ten steps and fresh-process five-plus-five steps had identical complete state and outputs. The actual model's two continuous cases and fresh-process one-plus-one cases had zero differences in 824 FP32 master tensors (3,754,622,976 elements), 2,472 Adam tensors (7,509,246,776 elements), and 38 actor buffers (2,388 elements). RNG, counters, configuration and stored outputs also matched. The output comparison covers every rollout token ID, greedy/probe text and token count, votes and rewards; greedy/probe token-ID arrays are not separately serialized. This verifies the tested short recoveries on this environment, not arbitrary long trajectories or cross-hardware identity.

The no-warp probability diagnostic passed on the tiny CPU model with maximum log-probability error 2.3842e-7 at tolerance 1e-6. A separate four-token actual-model diagnostic passed with maximum error 0.007085 at the predeclared BF16 tolerance 0.1. Its four-token limit is only for that diagnostic; production remains capped at 2048. Reference parameters were checked in full against initialization, excluded from optimization and unchanged at each completed segment. Engineering checks also confirmed nonzero vision/language gradients, actor movement, persistent masters/moments, all-invalid and equal-reward boundaries, and label-field rejection.

Failed work is retained. The first actual-model attempt stopped before its first update because deterministic CUDA histogram computation was unsupported; detached histogram/band statistics were moved to CPU in deterministic mode using the same functions. A subsequent sixteen-thread recovery attempt passed floating-state tolerance but failed exact generated-text equality: one visual master tensor differed by at most 3.7253e-9, leading to two different BF16 entries. The exact-output criterion was retained. Fresh runs with one CPU intra-op thread passed exact state and stored-output equality; the original parallel discrepancy's root cause is not conclusively established. CPU fixture/launch errors and the failed isolated plotting-venv setup are also recorded in [engineering acceptance and costs](continual_engineering.json). The ordinary nondeterministic single-update pilot path remains available.

Complete checkpoints, raw predictions, manifests, logs and failed attempts remain in allocated private data storage. Public files contain code, configurations, reproduction commands, aggregate JSON/CSV and this report. They omit original medical material, questions, responses, labels, private filesystem paths, credentials and model/checkpoint files. The main branch and historical pilot results are preserved. See [reproduction commands](continual_reproduction.md) for startup, recovery and independent scoring.

Training and prediction provenance is commit `7bf769d420bcb63004295e0d1900e5df6467c2d5`. The independent evaluator's additional parse-attribution diagnostics use `6064999c7d3279d0f25bf870e7a3802eed00d287`; only evaluation/comment code changed after training. The final manifest checksum is `155afbad741bc9cbf997910d0f89118d176f1f27d252128a8550703d277f31b9`, and the pre-fixed candidate manifest checksum is `60e1ab08460b490dbb8eafdcfd67ac43cabae5d79fe8865bd33261aec09fcb89`. The code preserves the requested manifest digests; no extra model/checkpoint file hashes were calculated.

The first actual correctness scoring occurred at 2026-10-04 20:27:18.242739 UTC, after all predictions closed. The later parse audit completed at 20:34:09.755407 UTC. Synthetic transition checks passed, and recomputation asserted that all original accuracy, probe and paired transition scores were unchanged. The decomposition is a post-prediction diagnostic, not a pre-registered new primary outcome or a basis for sample/configuration selection.

Final CPU checks successfully opened all four final checkpoints with trusted mmap loading and validated tensor schemas, Adam step 16 for each RL state, cursors, processed prefixes, saved RNG, configuration, model revision and manifest metadata. They also confirmed actual main-run resume events, matching initial probes and unchanged frozen probes. This is a targeted final availability/state check; it does not repeat the earlier full-state equality comparison or claim end-to-end equality between a hypothetical uninterrupted 16-case run and the actual resumed run.

T0 is 2026-10-04 14:27:18.205797 UTC. The main GPU controller ended at approximately 20:25:33.827 UTC. At 20:28:36.827 UTC, all 23 recorded campaign job identities and all campaign controllers were inactive, and the selected GPU had no active compute process. The later audit was CPU-only and exited with code 0. Other jobs were not terminated. No background experiment, extension or monitor remains part of this delivery.

This report was sealed at 2026-10-04T20:43:53.353350+00:00, 22595.148 seconds (6.276 hours) after T0. That measured interval includes development, failures, tests, GPU work, independent scoring, checkpoint validation, process cleanup verification and report assembly; the subsequent GitHub commit/publication check is reported with the final handoff elapsed time. The hard deadline is 2026-10-05 02:27:18.205797 UTC. Because the timing requirement was inside the supplied attachment, T0 UTC comes from the initial command dispatch while the monotonic origin was reconstructed from the first recorded UTC/monotonic pair; it was not sampled directly at dispatch.

The answer to the core question is a positive descriptive signal on this particular path: past updates are associated with three extra correct later-case before predictions for TTRL and one for SPINE. Neither current-case update produces a net accuracy gain. SPINE trails TTRL by two cases in both before and after accuracy and shows poorer valid-answer retention on the probe. The experiment establishes neither general method superiority nor statistically reliable adaptation benefit, and it does not validate real domain drift, a multi-seed effect, author-paper reproduction, or safety in clinical deployment.

The most useful next question is whether TTRL's observed historical advantage persists on a larger, prospectively fixed stream while preserving held-out probe correctness and valid-answer rates. A later authorized experiment could test that question without selecting samples or updates from vote correctness. This report does not start or promise any additional run.

These are sixteen-case, single-seed results on one dependent adaptation trajectory. No significance test, independent-sample bootstrap, domain-specific claim or smoothed trend is supplied. Exact eight-case block and cumulative tables and the three-point probe table replace small-sample curves.
