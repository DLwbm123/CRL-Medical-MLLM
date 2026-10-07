# P3: reward, format and two-seed replication protocol

Locked before any P3 correctness scoring. Base commit: `68c1733025c7be4135aba4b0f9db57c0a6ccaf09`; new branch: `codex/p3-12h`. This is an independent development experiment, not an author-verified reproduction or a final unbiased test.

## Clock and authorized resources

The first execution command recorded UTC `2026-10-07T06:13:24.657782+00:00` and monotonic `861873.005386125`. Scope freezes at 15:13:24 UTC, main training/sampling stops by 16:43:24 UTC, every GPU diagnosis stops by 17:13:24 UTC, and delivery ends by 18:13:24 UTC on October 7. The final hour is reserved for scoring, reports, publication and owned-process cleanup. Existing bounded supervision supplies actual child-process timeouts, signal handling and recorded process ownership. No continuation after the deadline.

Only the P2 GPU index 4 is authorized. Its device UUID and current memory are verified privately before each GPU job; processes use that UUID as the single visible device. No other jobs are stopped. The existing environment, original pinned base revision `66285546d2b821cf421d4f5eb2576359d3770cd3` and model processor defaults are reused. P2 checkpoints and artifacts remain unchanged.

## Data and matrix

The official development split has only five questions; two are old engineering examples. P2's engineering manifest also contains one previously used test example, and all three engineering image groups remain excluded from stream/probe. The remaining official development examples cannot supply the target new 32-group stream. P2 actually retired and scored 16 stream groups and 16 separate probe groups. Its candidate manifest contains 128 stream candidates, but inclusion alone does not prove the unused 112 groups were separately retired from final testing. Under the explicit P3 fallback, use the same 16 previously scored stream groups in exactly the same order, and the same 16 observed probe groups. No new final-test groups are retired or used. Stream, probe and engineering image groups are disjoint. Patient independence cannot be verified without patient IDs.

Data selection/order seed remains 42; rollout seeds are **43 and 44**. Each seed uses the same 16 groups. These are 16 unique stream groups, not 32 independent cases. No order-effect or domain-drift claim is made. The private manifest, checksum and authorization provenance are written before scoring; its SHA-256 is appended here after preparation.

| Method | Rollout seeds | Initialization | Stream |
|---|---|---|---|
| Frozen greedy | deterministic cache, generation seed 42 | original base | 16 |
| Frozen SC-8 | 43, 44 | fresh original base per seed | 16 per seed |
| Continual TTRL | 43, 44 | fresh original base per seed | 16 per seed |
| Continual SPINE | 43, 44 | fresh original base per seed | 16 per seed |

Frozen greedy predictions may be copied only after confirming identical model revision, configuration, inputs, prompt, original parser and deterministic decoding code. Cache provenance retains the original code and manifest; its P3 generation cost is zero. RL runs never load updated P2 actors. Each newly run trajectory initializes independent FP32 masters, Adam state and RNG from the original base. Same seed does not imply identical rollout histories across different actors.

Probe cursors are uniformly **0 and 16**, with no midpoint probe in the P3 main matrix. Probes use greedy decoding, restore RNG and model modes, and never update parameters. One complete final checkpoint is written per full trajectory using the inherited atomic writer. No per-case states or best-checkpoint selection. Engineering throughput uses one old engineering example excluded from stream/probe, with one complete TTRL step and final checkpoint. P2 actual-model recovery is not repeated because training arithmetic, optimizer, precision, state restoration and RNG routines are unchanged.

## Fixed training behavior

The only training-entry change permits the declared seeds through the existing acceptance guard. Every configuration field other than method and declared seed must match P2 acceptance. Eight fresh rollouts per case; temperature 0.7, top-p 0.95, cap 2048; full visual and language updates; AdamW learning rate 1e-6 and all P2 coefficients unchanged; one CPU thread; at most one optimizer step per case; fixed original reference; BF16 computation with persistent FP32 CPU masters/moments. Original loss, entropy band, mask, rewards and answer parser stay unchanged. Original all-invalid skip and zero-advantage behavior remain. Generation uses temperature/top-p while loss uses raw untempered softmax, without importance correction: strict on-policy equivalence is not claimed.

## A: offline P2 audit

Before using P2 conclusions, reconstruct original four-method scores, before/after transitions, every saved probe and probe transition, candidate coverage and consensus metrics. Stop and locate any mismatch. Reconstruct stored rewards from the original completions and original parser; do not regenerate rollouts.

Review every available unparseable output plus the first two parsed controls per method/output type/probe cursor. Fixed failure categories: token-cap without a parseable final choice; normal end missing final marker/choice; explicit choice in unsupported syntax; conflicting/nonunique final choices; prose without an unambiguous identifier; other/undetermined. Never infer intent from truth labels or medical plausibility.

Optional parser `marked_choice_v1_eval_only` adds only unambiguous bare option identifiers following an explicit answer marker, including newline, Markdown and boxed syntax; conflicting marked identifiers abstain. Label-free format examples test it. Apply it uniformly to all relevant outputs and report beside the original score. It never enters the trainer or rewrites P2 reports.

Independent evaluators report group-level coverage@8, vote correctness, correct-candidate/wrong-vote, ties, fewer than eight valid responses, all-invalid, all-equal-reward and zero-advantage rates; rollout-level parsed counts, negative advantage among correct parsed candidates, wrong parsed candidates among positive rewards, and mean correct-candidate advantage. All-equal rewards includes an all-invalid group's eight zero rewards; the inherited zero-advantage flag excludes that skipped group. These are separate denominators/counters, with no change to training. Every rate includes numerator and denominator. Fixed bins: highest votes 0–3/4–5/6–8, margin 0/1–2/3–8, valid count 0–3/4–7/8. No fitted calibrator, case selection or parameter changes follow this audit. A reward correctness implementation bug stops effect experiments.

## B: legal-option likelihood diagnosis

Maximum new GPU budget: **2700 seconds** including loading. Actor priority: original frozen base, P2 SPINE final cursor 16, P2 TTRL final cursor 16, P2 SPINE midpoint cursor 8. Use all 16 old probe groups for every completed actor. Existing states are opened read-only; no optimizer or training. Missing/unloadable states are reported without P2 retraining.

The common answer-only instruction is: “Select exactly one of the listed options. Return only its option identifier. Do not provide reasoning.” The original image order, question and all options are included; no generated reasoning is included. The assistant prefix is `Final answer:`. For each legal identifier, append a space and the identifier and sum its complete token sequence conditional log-probabilities under the same prefix. Validate common-prefix tokenization. No EOS score and no length normalization; choose the lexicographically first maximum. Save tokenization and option scores privately. Inference must preserve parameter versions, RNG and modes. Labels enter only the separate evaluator after predictions close.

This is a separate diagnostic protocol, never a replacement primary score or reward. Difference from free generation is not a pure-format-loss estimate and cannot prove knowledge retention or forgetting.

## C: primary metrics and sealing

All seven main-run predictions, configuration, provenance and coverage must be sealed before any P3 label join. Seed 43 scores do not determine whether seed 44 runs. For each seed report frozen greedy, SC-8 consensus, TTRL/SPINE before and after correct/N and parsed/N; initial/final probe counts and transitions. Preserve failures/skips in denominators.

`G_history = before accuracy − frozen greedy accuracy`; `G_current = after accuracy − before accuracy`; `Delta_SPINE_vs_TTRL = SPINE before accuracy − TTRL before accuracy`. Decompose every comparison into parsed-wrong→correct, invalid→correct, correct→parsed-wrong and correct→invalid. Compare improvement-set overlap across seeds and preservation of initially correct probe groups. Apply A's format/reward audits to all P3 outputs offline. Preserve inherited length, mask, loss, gradients, KL and resource logs privately.

Report seed values, means and ranges descriptively. No ordinary independent-case bootstrap or small-sample significance claim. Stream predictions depend on earlier updates; two seeds, repeated observed development groups and one order do not establish generalization or order stability. Reward/format associations cannot establish which update caused degradation.

## Budget and downgrade rules

Before full matrix launch, use P2 phase timings plus one full short throughput check to forecast four RL trajectories, two SC-8 runs, all initial/final probes, six final checkpoint writes, loading, B and delivery. Reserve at least **20% above the main forecast**, separately from the last hour. N=16 is already fixed by authorization; no expansion to 32 without an existing separately retired unused development segment. Do not increase scope after seeing results, and finish early when locked work is complete.

Priority: four equal-length RL trajectories across both seeds, frozen controls, diagnoses, then any already-authorized sample expansion. Default full-run schedule avoids unnecessary restart I/O. On unexpected timeout/failure, seal all actual coverage, score only the same-seed fixed common prefix, disclose missing scope and compare no unmatched method. Do not merge prefixes across seeds. Never reduce rollout count, output cap, precision, visual updates/reference or increase CPU threads to enlarge N. No silent retry or changed protocol.

## Public delivery

Publish source, six seed configs, reproducible environment-based launch/resume/scoring commands, aggregate CSV/JSON and final report on the new branch. Exclude raw images, full questions, completions, truth labels/IDs, weights, checkpoints, credentials and private absolute paths. Verify branch publication and anonymous report access using the required proxy. Confirm all owned GPU jobs have ended. The final report answers H1–H3 and selects one evidence-based next research question without executing it.

Private manifest SHA-256: `6076bd13e6c36d0fa12ef88530f5688633d74762e069b9038f8db971dd8b1306`. Its scope/order and two seeds were fixed before any P3 scoring.

Budget locked at 2026-10-07T06:56:18.152491+00:00: main forecast 6.94 h; with 20% reserve 8.32 h; 9.79 h remained before the main stop. Final delivery has a separate one-hour reserve. Fixed main order: c43, d43, c44, d44, b43, b44; job limits 7200 seconds per RL trajectory and 3000 per SC-8. See `p3_budget_plan.json`.
