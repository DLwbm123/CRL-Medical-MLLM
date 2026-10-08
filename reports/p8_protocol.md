# P8 frozen diagnostic protocol

Starting commit: `3ccabb3baf101f574186fcd63abf9c8a6b31d801`; branch: `codex/p8-mechanism-diagnostics`. This is artifact analysis and read-only model diagnosis, not training. No optimizer construction, optimizer step, new checkpoint, true-label model input, new final-test access or P9 launch is permitted. Old source, states, reports and branch refs remain unchanged.

## Clock and admission

T0 is conservatively fixed at 2026-10-08 13:01:41 UTC (21:01:41 China time), the first neutral Python process start. UTC and monotonic were not recorded contemporaneously in that first command: the instruction was read afterward. At 13:06:15 UTC the process start was recovered from `ps`, and monotonic T0 reconstructed from elapsed epoch time. This limitation is retained; the deadline is not restarted after interruption.

No scope increase after 23:01:41 UTC; all GPU workers end by October 9 00:01:41 UTC; all work and publication end by 01:01:41 UTC. P4–P7 prior whole-worker use is 65,505.54222541861 seconds; authorization remains 86,400 seconds. Fresh admission must confirm no additional charged work. P8 whole GPU-worker lifetime cap is min(10,800, latest remaining /1.2) seconds, including imports, model/state loading, CPU waits, failures and exit. One worker; no persistent new monitor.

Conservative stage forecast: C 1,200 seconds; D two groups 7,200 seconds; combined with 20% reserve 10,080 seconds. D one-group fallback forecast: 4,200 seconds. The D allowance uses measured P7 maximum mean full-group backward time 71.68781 seconds, three component reverse passes, seven actors and two groups, with a factor-two allowance plus loading/forward overhead; it assumes no speedup from restricting parameters. These are forecasts, not runtime guarantees. Whole-life supervisor enforces the cap and UTC stop. Before D, remaining worker time alone selects two groups if at least 8,640 seconds, one if at least 5,040, otherwise zero; performance outputs do not determine admission. No seed or arm is dropped. C completes before D.

## A/B: existing complete artifacts

CPU evaluation uses each round's executed source checkout and original evaluator helpers. P5 seeds48/49/50, P6 seeds51/52/53, P7 seeds54/55/56, both m/v arms: 18 complete trajectories. Check source receipt, configuration, model revision, seed, arm, cursor16 and manifest before raw output analysis. Recompute every original metric produced by `score_cases`, history/current parse transitions, initial/final probe statistics, reward-direction audit and locked format summary; require exact equality with sealed aggregates before mechanism analysis. Labels enter only the separate offline evaluator after provenance and prediction parsing checks.

There are 16 distinct known stream groups and16 distinct known probe groups, reused across trajectories. Stable aliases S01–S16 and P01–P16 follow existing manifest order. Private tables retain original IDs and answer counts; public tables retain count profiles and correctness statistics, never option identifiers, text or labels. Positive reward and positive advantage are distinct. Zero-denominator quantities are N/A. Group maps report all three observations per round/arm, consensus changes, correct-candidate counts, vote counts/margins and history gain/loss; no new filtering or threshold.

B covers all initial/final probe outputs and all stream before/after outputs, with the unchanged P3 `marked_choice_v1_eval_only` parser and failure classifier. Strict parsing remains primary. Explicit unsupported syntax is separated from actual unambiguous recovery; classifier tags alone do not establish correctness. Clinical semantic inference is not performed. P7 v55's original probe B is one row of full coverage, with identical rules.

## C: seven actors, all16 probes

Actors: original pinned frozen base plus final P7 m54/v54, m55/v55, m56/v56. Private frozen manifest identifies actual checkpoint paths from latest-state metadata, exact executed source a7e8531387997fca8d4e15cda36fd6eaf0b98ea5, model revision66285546d2b821cf421d4f5eb2576359d3770cd3, seed/configuration, arm, cursor16 and manifest. Restore BF16 parameters from trusted FP32 masters and actor buffers using the existing read-only helper. Optimizer moments never move to GPU; no optimizer is instantiated.

Reuse exactly P3's answer-only image/question/options template, assistant prefix `Final answer:`, and complete option-identifier conditional log probabilities. Prefix and ordered images are identical across actors. Suffix is space plus identifier; score sums all suffix-token log probabilities, no EOS or length normalization; lexicographically first maximum resolves ties. No model-generated reasoning or label-derived prefix. Free-generation strict correctness and legal-readout correctness remain separate; each has its own frozen-base initial-correct set.

At the identical first-answer position, compute exact full-vocabulary KL(actor || base), base top-token retention, legal top1–top2 gap. After output seal and confirmed worker exit, the offline evaluator adds correct-option versus best-wrong gap. These are uncalibrated scores. Fixed-position KL differs from training response-masked KL and does not identify training-path causality. Protocol recovery supports protocol sensitivity only; continued failure does not identify a medical concept as forgotten.

## D: fixed old pool, local component gradients

Preferred pool is original P3 FrozenSC8 seed43, first two stream groups in manifest order, all eight original response-token sequences per group. Require exact stored lengths and tokenizer decoding equality; no resampling, re-tokenization substitute, correctness selection or truncation. If tokens fail provenance, skip D. Under time fallback keep only the first group, uniformly for all seven actors. Private manifests freeze these rows before new GPU outputs.

Named-parameter subset, selected from the actual CPU meta model before GPU work:

- `visual.patch_embed.proj.weight` [1280,3,2,14,14]
- `visual.blocks.31.attn.qkv.weight` [3840,1280]
- `model.layers.0.self_attn.q_proj.weight` [2048,2048]
- `model.layers.18.self_attn.q_proj.weight` [2048,2048]
- `model.layers.35.self_attn.q_proj.weight` [2048,2048]

Recompute current-actor log probabilities, entropy, detached Otsu selection and detached band in training mode; detached current log probabilities define ratio1. Recompute original label-free consensus advantages from the old pool. Reference logits always come from the original base. Use original full-group token/selected-token denominators, clip epsilon0.2. Separate policy, unit-coefficient band and raw KL gradients with `autograd.grad`; apply actual band0.01 and each actor's KL0.01/1.0 afterward. Base uses the original SPINE diagnostic coefficients. No actual Adam step, clipping or weight decay is applied. All-zero advantages and invalid groups stay in scope.

Report module, vision, language and combined selected-subset norms, weighted ratios, cosines and composite/sum-norm cancellation. Zero norms imply N/A ratios/cosines. Base=reference raw-KL gradient subset norm must be at most0.005 (engineering numerical tolerance, frozen before GPU execution). A violation is an engineering failure, not a scientific outcome. These are fixed-old-pool local objective gradients, not on-policy samples or historical gradient replay. The subset does not represent all model gradients; gradient size/direction does not establish accuracy effects or a cause of any particular loss.

Optional mask spans are omitted because token–text span mapping has not been verified. This is the first scope downgrade, before any group reduction.

## Engineering and interpretation

CPU tests cover synthetic exact KL, complete mult-token identifier scoring, component-sum equality to the original objective, coefficient scaling, zero advantages/zero norm, invalid parsing, label-field rejection, no step calls and diagnostic state restoration. Original sealed scores are independently verified on actual artifacts. Actual GPU checks cover every parameter version counter, three sampled values per tensor, all buffer values, module modes and RNG restoration. This is not a full parameter-value byte comparison. Inputs use the strict existing label-free allowlist. GPU entry does not import an evaluator.

Final report distinguishes old facts, new diagnostics, hypotheses and unexecuted validation. Select exactly one priority among new verified reward information, output protocol stabilization, a band on/off causal comparison, or independent development expansion if evidence is insufficient. P9 draft preserves three new unused seeds, paired controls and joint criteria; no new data or compute authorization is implied. Publish all sanitized outcomes, missing evidence, failures, code, tests, commands and whole-life receipt to this new branch, via the authorized proxy, then verify final SHA and anonymous access.
