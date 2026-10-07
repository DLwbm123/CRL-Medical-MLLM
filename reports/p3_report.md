# P3 reward/format diagnosis and two-seed continual-RL replication

TTRL's historical gain does not have a consistent sign across the two new rollout seeds, and the SPINE–TTRL ranking reverses. SPINE's current-case correct counts rise under both seeds, but its historical gain and probe behavior are not uniformly improved. The strongest repeated diagnostic finding is disagreement between consensus reward and answer correctness. Old P2 actor legal-option readout preserves the frozen model's five correct probe groups, supporting protocol sensitivity without proving medical-knowledge retention.

This is a complete **16-group development replication with rollout seeds 43 and 44**, using one fixed data order (selection/order seed 42). The same previously scored P2 stream and previously observed probes are reused under the explicit fallback because retirement/authorization of unused candidates could not be established. These are 16 unique stream groups and 16 separate probe groups, not 32 independent cases or an unseen final test. Patient independence is unverified.

## Closed scope and provenance

All six newly generated main runs completed 16/16 groups without a failure or restart. Frozen greedy was reused after equivalence checks of model, inputs, prompt, decoding and original parser. Every new main run's initial probe matched the cached base in ordered ID, parsed answer and token count; frozen SC final probes matched too. Each RL trajectory began from the original pinned base with zero optimizer updates, independently of P2 adapted actors. All four RL states contain 16 updates; the one all-equal-reward TTRL seed-44 group does not imply a skipped optimizer step. Original zero-advantage behavior, persistent moments and parameter updates remain as implemented.

Base P2 commit: 68c1733025c7be4135aba4b0f9db57c0a6ccaf09. Main training commit: 0d3f87831e0d195534715f9eddb8edbe2f46ace0; short engineering/readout source: a0f4d9a60ae4e421b172a49c5244b4b5caa337d5; evaluator: 9abfb9bdfb2c5d908ea392f0c7803c37181da7a5. Pinned model revision: 66285546d2b821cf421d4f5eb2576359d3770cd3. New branch: codex/p3-12h. The only trainer-entry change permits the declared seeds; training arithmetic, reference, masks, optimizer, precision and recovery/RNG routines are unchanged. No expensive P2 actual-model recovery rerun was needed.

The [locked protocol](p3_protocol.md) records the manifest digest and authorization boundary. All main prediction/configuration scopes and four old-actor readouts were sealed before the new evaluator opened labels. Task A had already used previously observed P2 labels independently; sealing does not turn reused cases into unseen data.

## H1: main scores and repeatability

Original free generation plus the P2 strict parser is the primary measure. Before records performance after prior stream updates but before the current case update; after includes that update. SC uses fresh eight-sample votes. Invalid predictions remain in every denominator.

| Seed | Method | Before correct | After correct | SC vote correct | Before parsed | After parsed | SC valid groups |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 43 | Frozen greedy | 2/16 | — | — | 14/16 | — | — |
| 43 | Frozen SC-8 | — | — | 1/16 | — | — | 16/16 |
| 43 | Continual TTRL | 1/16 | 3/16 | 1/16 | 13/16 | 13/16 | — |
| 43 | Continual SPINE | 2/16 | 4/16 | 1/16 | 12/16 | 12/16 | — |
| 44 | Frozen greedy | 2/16 | — | — | 14/16 | — | — |
| 44 | Frozen SC-8 | — | — | 3/16 | — | — | 16/16 |
| 44 | Continual TTRL | 5/16 | 3/16 | 2/16 | 12/16 | 13/16 | — |
| 44 | Continual SPINE | 3/16 | 4/16 | 2/16 | 13/16 | 15/16 | — |

History gain = before − frozen greedy; current gain = after − before; Delta = SPINE before − TTRL before, in percentage points.

| Quantity | Seed 43 | Seed 44 | Descriptive mean | Range |
| --- | ---: | ---: | ---: | --- |
| TTRL history gain | −6.25 | +18.75 | +6.25 | −6.25 to +18.75 |
| SPINE history gain | 0.00 | +6.25 | +3.125 | 0.00 to +6.25 |
| TTRL current-case gain | +12.50 | −12.50 | 0.00 | −12.50 to +12.50 |
| SPINE current-case gain | +12.50 | +6.25 | +9.375 | +6.25 to +12.50 |
| SPINE–TTRL before difference | +6.25 | −12.50 | −3.125 | −12.50 to +6.25 |

TTRL's prior P2 +18.75 pp historical signal recurs at seed 44 but reverses at seed 43; it is not replicated as uniformly positive. SPINE has zero/positive historical gains, but its prior deficit to TTRL is not stable. Both SPINE after scores are 4/16 versus both TTRL after scores of 3/16; this limited current-case observation does not establish general superiority.

Paired decomposition keeps simultaneous gains and harms visible. Retained correct refers to the same initially correct groups.

| Seed | Method/comparison | Stage | Parsed wrong→correct | Invalid→correct | Correct→parsed wrong | Correct→invalid | Retained correct | Net pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 43 | Continual TTRL | Frozen → before | 1 | 0 | 1 | 1 | 0 | -6.25 |
| 43 | Continual TTRL | Before → after | 1 | 1 | 0 | 0 | 1 | 12.5 |
| 43 | Continual SPINE | Frozen → before | 1 | 0 | 1 | 0 | 1 | 0.0 |
| 43 | Continual SPINE | Before → after | 2 | 0 | 0 | 0 | 2 | 12.5 |
| 43 | TTRL before → SPINE before | Between methods | 1 | 0 | 0 | 0 | 1 | 6.25 |
| 44 | Continual TTRL | Frozen → before | 3 | 0 | 0 | 0 | 2 | 18.75 |
| 44 | Continual TTRL | Before → after | 0 | 0 | 1 | 1 | 3 | -12.5 |
| 44 | Continual SPINE | Frozen → before | 1 | 0 | 0 | 0 | 2 | 6.25 |
| 44 | Continual SPINE | Before → after | 1 | 1 | 1 | 0 | 2 | 6.25 |
| 44 | TTRL before → SPINE before | Between methods | 0 | 0 | 1 | 1 | 3 | -12.5 |

Every historical improvement in these four RL trajectories is parsed-wrong→correct; none is invalid→correct. Historical before parse counts decline from frozen 14/16 to 12–13/16, so history gains cannot be explained as a net parsing-rate improvement. TTRL seed 43 loses both originally correct stream predictions and gains one different prediction. Seed 44 retains both and gains three. SPINE seed 43 retains one of two and gains one; seed 44 retains both and gains one.

TTRL historical gain sets contain 1 and 3 groups with intersection 1 and union 3. SPINE improves 1 group in each seed with intersection 0 and union 2. These are repeated development observations, not independently sampled discoveries.

## Probe retention

All methods share probe cursors 0/16. Frozen greedy and SC remain at 3/16 correct and 13/16 parsed in both contexts, preserving the same three correct groups.

| Seed | Method | Correct | Parsed | Same initially correct retained | Parsed wrong→correct | Invalid→correct | Correct→parsed wrong | Correct→invalid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 43 | Continual TTRL | 3 → 4/16 | 13 → 13/16 | 2/3 | 2 | 0 | 0 | 1 |
| 43 | Continual SPINE | 3 → 2/16 | 13 → 10/16 | 2/3 | 0 | 0 | 0 | 1 |
| 44 | Continual TTRL | 3 → 4/16 | 13 → 16/16 | 3/3 | 1 | 0 | 0 | 0 |
| 44 | Continual SPINE | 3 → 4/16 | 13 → 15/16 | 3/3 | 1 | 0 | 0 | 0 |

TTRL reaches 4/16 under both seeds, but seed 43 retains only two of the initial three correct groups and gains two different groups. Seed 44 retains all three and gains one. Equal final totals conceal different retention. SPINE loses one initially correct group to invalid output at seed 43; seed 44 preserves all three and gains one. The seed-43 loss is an answer-validity failure, not evidence identifying a forgotten medical concept.

The uniformly applied extra syntax parser leaves every P3 primary stream before/after correct count and every SC vote correct count unchanged. It changes TTRL seed-43 final probe 4→5/16 and SPINE seed-43 final probe 2→3/16; other final probe correct counts are unchanged. At SPINE seed 43 it recovers the lost correct probe's explicit marked identifier, providing direct evidence of a strict-parser syntax component to that loss. Syntax diagnostics do not replace primary scores or rewards.

## H2: format and old-actor legal-option readout

P2 original method/probe scores, transitions and reward diagnostics reproduced exactly. All 145 invalid records (117 unique texts) and 40 stratified parsed controls were audited. Exclusive P2 invalid categories: missing marker/explicit choice 88, unsupported explicit syntax 49, token cap 6, prose 1, illegal identifier 1, conflict/other 0. The extra parser recovers 14 parsed records, but no P2 main/probe correct count changes; one SPINE rollout correct count increases 14→15/128. P2 reports and the trainer parser remain unchanged. See [the full format audit](p3_format_audit.md).

Legal-option readout uses the original frozen base and P2 TTRL final, SPINE midpoint and SPINE final actors. Each uses all 16 probes, the same answer-only instruction, image/question/options and Final answer: prefix. Complete identifier token log-probabilities are summed; no generated reasoning, EOS or length normalization. Ties select the lexicographically first maximum. All 80 identifiers per actor tokenize as one token; the CPU check also validates a complete two-token sequence. No optimizer/update/label enters inference, and RNG/modes are preserved.

| P2 actor | Free correct | Free parsed | Legal correct | Legal parsed | Frozen legal correct retained |
| --- | --- | --- | --- | --- | --- |
| frozen_base | 3/16 | 13/16 | 5/16 | 16/16 | 5/5 |
| p2_spine_final | 2/16 | 11/16 | 5/16 | 16/16 | 5/5 |
| p2_spine_midpoint | 1/16 | 9/16 | 5/16 | 16/16 | 5/5 |
| p2_ttrl_final | 3/16 | 13/16 | 5/16 | 16/16 | 5/5 |

All four actors select the same five correct probe groups under legal readout: zero correct→wrong and wrong→correct transitions versus frozen legal readout. Wrong options need not be identical. SPINE midpoint and final recover 2/2 and 1/1 of the subsets initially correct under free generation that became invalid in those respective old probes. Subsets are retrospective diagnostics after full-probe evaluation, not selected evaluation samples.

Together with P3 syntax recovery, this supports output-protocol sensitivity. Legal readout changes the conditional protocol, so its score minus free-generation score is not pure format loss. It cannot rule out other capability changes or prove medical knowledge retention. Legal readout was **not run on new P3 adapted actors**.

## H3: reward direction and high-confidence mistakes

The independent offline evaluator reconstructs original consensus rewards and standardized advantages without changing cases, learning rate, checkpoints, parser or updates. Group and rollout denominators differ.

| Seed | RL method | Correct candidate groups@8 | Correct votes | Correct rollout with negative advantage | Wrong parsed rollout with positive reward | Mean correct advantage |
| --- | --- | --- | --- | --- | --- | --- |
| 43 | Continual TTRL | 8/16 | 1/16 | 9/14 | 67/72 | -0.3353 |
| 43 | Continual SPINE | 6/16 | 1/16 | 7/11 | 61/65 | -0.2479 |
| 44 | Continual TTRL | 7/16 | 2/16 | 10/18 | 64/72 | -0.1165 |
| 44 | Continual SPINE | 10/16 | 2/16 | 14/21 | 58/65 | -0.2689 |

Across four RL trajectories, correct parsed candidates receive negative advantage in **40/64** rollout observations (62.5%). **250/274** positive-reward observations are parsed wrong (91.24%). There is a correct candidate in 31/64 group observations, yet only 6/64 votes are correct. These observations reuse the same 16 groups and are dependent; pooling is descriptive.

The preregistered high-vote bin (6–8 votes) has **0/13 correct votes** across the RL trajectories. The large-margin bin (3–8) has **1/25 correct votes**. Frozen SC seed-44 high-vote groups have 1/5 correct votes. The CSV reports all fixed bins, ties, valid counts, coverage and exact denominators, including empty bins. Simple vote-count/margin gating is not validated.

No P3 group is all-invalid. One of 16 TTRL seed-44 groups has all-equal formed rewards/zero policy advantage; other P3 arms have 0/16. All-invalid groups skip reward construction and belong to neither numerator, while remaining in the group denominator. A reporting-counter correction for this convention was made and CPU-checked before P3 grading; training behavior did not change. Positive reward differs from positive advantage. Zero policy advantage need not stop AdamW momentum or other loss contributions.

P2 correct-negative-advantage counts B/C/D were 11/14, 9/17 and 12/14; wrong-parsed-positive-reward counts were 63/66, 61/69 and 60/62. P3 extends frequent reward–correctness disagreement beyond the original seed. This association does not show which erroneous reward caused a particular prediction/probe loss.

## Validation and resources

Existing tiny-model recovery/RNG/probe checks and targeted CPU checks passed after an initial missing test-directory environment variable was fixed. Checks cover the declared-seed guard, label-free syntax, original parser, reward denominators/high-vote mistakes, complete multi-token likelihood, RNG/mode preservation and real cache provenance. One read-only metadata check of seven new states passed for saved seed/config, cursor, ordered groups, provenance, reference, RNG schema, FP32 masters/moments, Adam steps and expected save sizes. It uses mmap; no full state hash/tensor-value comparison or new GPU recovery run.

The actual export passed for 8 main rows, 198 reward rows and 56 probe rows; both seeds have the full four-method N=16 matrix. Each new main run generated 128 fresh rollouts. Frozen greedy costs zero new generation. [Aggregate JSON](p3_results.json) preserves reward, length/token, mask, loss, visual/language gradient, reference KL, post-update drift, online-block and resource summaries; original per-case logs remain private. These diagnostics do not establish causes of method differences.

Main supervision took 4.42 hours; all GPU work ended at 11:21:31 UTC (19:21:31 Beijing), about 5.14 hours after T0. Its measured cost was below the conservative 6.94-hour forecast. Task B took 206.51 seconds including loading/process overhead; the engineering short check took 474.27 seconds including its checkpoint. Total wall time includes the period between GPU completion and the later status request; phase timers are not additive totals.

Peak main allocated GPU memory was 27.39 GiB, reserved memory 49.50 GiB and per-worker CPU RSS 69.19 GiB. Whole-device memory can include unrelated jobs and is separately identified. Four main RL states plus one engineering state total approximately 225.28 GB; two SC states are small. Each main run wrote one final state, with no best-checkpoint selection or demonstration restart. Historical P2 artifacts and unrelated processes were not modified.

All recorded owned controllers/workers/sessions were verified ended before scoring. No GPU process was launched afterward. [Budget/resources](p3_budget_and_resources.json) records phase costs, checkpoint I/O, peaks, failed attempts, completion-to-scoring delay and elapsed delivery time. [Reproduction instructions](p3_reproduce.md) retain launch/resume/scoring boundaries.

## Evidence limits and one next priority

This run adds four independently initialized RL trajectories under two new sampling seeds, two fresh SC controls, uniform parser diagnostics and full old-actor readout comparisons. It does not add unique stream/probe groups, unseen final-test evidence, more data orders, causal reward/entropy-mask ablations or P3 actor legal-readout evaluation. No significance test or independent-case bootstrap. These are independent implementations without established author-code equivalence; raw-softmax loss with temperature/top-p sampling is not newly validated strict on-policy RL.

The **single next priority is an offline, preregistered test of an answer-verification signal independent of the same model's majority vote**, before adding an online reward/gate. This follows repeated wrong positive rewards and 0/13 high-vote correctness. Independent development data and more trajectories remain necessary to judge adaptation benefits, but simple confidence gating should not be assumed to solve the diagnosed problem. No new signal, experiment, hyperparameter search or mechanism was executed.

Public artifacts contain source/configs, protocols, reproducible commands and aggregate results only. Raw images, questions, generated answers, labels/IDs, checkpoints, full logs, credentials and private absolute paths remain excluded.
