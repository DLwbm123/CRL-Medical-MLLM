# P10: frozen visual reward qualification and conditional continual RL

The user authorized implementation and experiments on pro5000 physical GPUs 0/1, with **48 additional GPU-process hours**, on 2026-10-09. This is a new independently bounded campaign. P9 remains its unexecuted historical draft; P2–P8 protocols, results, deadlines and branches remain unchanged.

## Hypothesis and scope

Prioritize reward qualification before another parameter update. A separate frozen Qwen2.5-VL-7B-Instruct checkpoint (`cc594898137f460bfe9f0759e9844b3ce807cfb5`) sees the actual ordered images, question and all options. It receives no clinical label, actor completion, answer-frequency count or evaluation result. It reasons briefly, then emits a strict final-line JSON choice plus visual-support and clinical-consistency judgments. These self-reported judgments are fallible proxies, not verified medical evidence. It is a larger distinct checkpoint in the same model family, not an independent family or medically validated expert.

Each input has exactly two deterministic greedy readouts, with sorted and reversed option orders relabeled to A, B, ... . Both canonical choices must agree and both judgments must declare support/consistency. Otherwise the reward source abstains. No parse repair, prompt sweep, teacher replacement or threshold adjustment follows scoring. All images retain their original order and original processor defaults. Teacher output cap is 1024 new tokens per readout, without resolution reduction or quantization. All outputs and failures remain in the denominator.

Use only the **same authorized, already observed 16 stream groups and 16 held-out legacy probes** from P7. They are retired test-derived development groups. No newly held-out final test, new labels or external clinical service is accessed. The teacher independently processes each input; caching targets in advance does not transfer information between cases. The gate uses development correctness only after the frozen worker has ended. Its selection dependence must be disclosed in any later result. No independent generalization or patient-level independence is established.

## Fixed qualification

On GPU1, generate all 32 inputs × two option orders, seal all targets, verify worker exit and process-session cleanup, then run a separate CPU evaluator. It audits all **18 closed P5/P6/P7 trajectories ×16 stream groups ×8 candidates** with their original strict parser. Repeated observations are 288 groups; unique stream groups remain16. Existing probe labels are used only for offline reporting; probe targets do not enter updates.

All six criteria must pass:

1. At least8/16 stream targets are accepted.
2. At least4/16 stream targets are correct.
3. Correctness among accepted targets is at least50%.
4. The fraction of correct parsed candidates receiving negative advantage is strictly below the majority baseline on the identical pool.
5. The wrong-answer fraction among positive-reward candidates is at least10 percentage points below the identical-pool majority baseline. Undefined denominators fail.
6. At least one observed group with a correct candidate and wrong majority is assigned the correct teacher target.

The gate is a small dependent development screen. Passing permits the frozen matrix; it does not demonstrate clinical reliability or adaptation. A negative gate ends the campaign with full negative reporting and no matrix launch. No other teacher or alternate gate is tried in this campaign.

## Frozen conditional training matrix

Only after the gate passes and full compute admission passes, use GPU0 to run a fresh frozen greedy baseline and **all nine trajectories**: three different unused rollout seeds **57/58/59**, each with three arms:

| Arm | Reward | Optimizer and objective | Sampling |
| --- | --- | --- | --- |
| s | Original majority | Original SPINE; original handling of zero-advantage groups | temperature0.7, top-p0.95 |
| t | Original majority | Matched TTRL all-response-token clipped policy objective + fixed-base KL0.01; skip zero-advantage groups | temperature1, top-p1, top-k0 |
| v | Frozen accepted visual target | Same matched TTRL settings as t; abstention and zero-advantage groups skip the entire optimizer step | same as t |

The **v–t comparison changes only the reward source and its necessary abstention**. The original SPINE arm provides continuity with the old stable-positive criterion. This is an independently implemented candidate, not a reproduction of CaT/SR-TTRL or a novelty claim. CaT/RLER motivate decoupled fixed rewards; the concrete medical visual gate is this campaign's declared adaptation.

All arms share the pinned Qwen2.5-VL-3B actor, 8 rollouts, cap2048, one maximum update per case, learning rate1e-6, BF16 full-model training with vision enabled, FP32 CPU AdamW masters/state, one CPU intra-op thread, reference checkpoint, original ordered inputs and score parser. No replay, adapters, extra retention mechanism or checkpoint selection is introduced. t/v use raw-softmax sampling; the actual-backbone probability consistency check must pass before the matrix. Existing full-model recovery evidence is reused because optimizer/checkpoint implementation is unchanged; new synthetic reward, label boundary, skip/Adam-state and state-recovery checks are mandatory. No new full-model optimizer recovery comparison is claimed.

Prequential greedy **before** scores are primary. Current-input after-update scores are secondary. Greedy legacy probes are fixed at cursor0 and16. Complete states are sealed only after all nine trajectories finish; the offline evaluator checks every output, source, reward and common initial probe before joining legacy labels. A single seed's performance never changes the remaining matrix. Any runtime failure retains its costs and stops the campaign; no automatic retry or partial-matrix positive claim.

Stable-positive development requires, for **every** seed, candidate before accuracy strictly greater than the frozen base, same-seed original SPINE and matched TTRL; and retention of all **three same initially correct probes**. Report every probe's correctness and parse harm, all negative seeds, update/abstention coverage, teacher no-update target scores, rollout reward failures, token totals and whole-worker costs. These reused-group findings do not establish unseen clinical or domain-shift generalization.

## Compute and deadlines

Original budget86400s plus newly authorized172800s = **259200 GPU-process seconds (72 hours)**. Prior completed lifetime use is **66990.15299156541s**; available at P10 start is **192209.8470084346s (53.3916 hours)**. Never reset the old budget or count controller archives twice. Count imports, loading, CPU optimization, saving, failures and exit for every GPU worker; sum overlapping workers if introduced. This campaign runs one GPU worker at a time: teacher on GPU1, actor/acceptance on GPU0. No GPU2–7 or paid resources.

The full matrix cost proxy uses the previous slowest observed complete trajectory4590.984604918864s ×9 plus1800s frozen-baseline allowance, with a separate1800s actual-acceptance allowance: **44918.86144426978s estimated;53902.63373312374s including20% reserve**. Freeze this proxy before qualification; it is not a promise of runtime. Check cumulative remaining budget and independent GPU-stop time after qualification. Do not admit if either cannot cover the complete estimate plus reserve. Qualification worker limit7200s; acceptance1800s; frozen baseline1800s; each training worker7200s. No limit may be extended in flight.

At preparation, mint a new **46-hour GPU stop and48-hour wall hard deadline** in this isolated workspace; preserve all old round limits. Public model download is a CUDA-free, two-hour bounded stage and consumes wall time; it makes no clinical uploads. Workers use neutral external entry paths and environment configuration, with full ps/nvidia-smi inspection. Store private artifacts under the established allocated server storage; preserve all old data, checkpoints and processes.

## Delivery and monitoring

Use the existing hourly monitor, once per wake, with quiet unchanged states. Read the new control/protocol/handoff and receipts; never duplicate the pipeline. Only meaningful completion, failure, admission changes, required user action or stable-positive results notify. After actual completion, publish code, complete sanitized aggregates, report, reproduction instructions and continuous compute receipt to the new branch through the effective GitHub proxy. Verify remote SHA and final commit anonymous access. Raw medical IDs, questions, options, labels, images, checkpoints, generated text, full logs, credentials and private paths remain private. Passing this gate or launching jobs is not experimental completion or a guarantee of positive results.
