# P4 fixed-reference reward: first optimization round

Authorization: the user requested performance improvement and hourly monitoring until stable positive development results, and explicitly chose a total budget of **24 GPU process-hours**. This is new work after completed P3, not a reset or extension of P3. Only its existing authorized GPU 4 is used. All owned GPU worker lifetimes (including CPU optimizer/checkpoint time) consume this conservative budget. Preparation/CPU-only work does not consume GPU process-hours, but is recorded. Each round also has its own wall deadline; round one is at most 12 hours with GPU work stopped by +10.5 hours. Budget is cumulative across trials and cannot be reset after negative results.

## Evidence and one change

P3 found 250/274 positive-reward RL responses wrong, 40/64 correct parsed responses assigned negative advantage, and 0/13 high-vote RL groups correct. Old frozen/P2 adapted actors scored the same five correct probe groups under a separate answer-only legal-option readout. The first hypothesis is that the frozen original base's fixed legal-option choice provides a less misleading pseudo-target than each adapting actor's sampled majority.

Candidate: original SPINE training with binary reward 1 for a strictly parsed answer agreeing with that frozen pseudo-target, 0 otherwise. Control: original SPINE majority reward. The critic is the **same backbone under another fixed prompt**, not an independently validated external medical verifier. No claim that its targets are correct by construction.

The identical frozen reference model, SPINE mask/entropy band/KL, raw-softmax loss, 8 fresh rollouts, 2048 cap, temperature/top-p, full visual/language updates, persistent FP32 masters/Adam, one CPU thread and checkpoint/RNG behavior are retained. No confidence gating, parser change, auxiliary loss, replay, new model or parameter search is bundled into this comparison. Original all-invalid skip and zero-advantage behavior remain, including possible Adam/band/KL effects at zero policy advantage.

## Scope, seeds and success

Reuse exactly the 16 observed P2/P3 stream groups and 16 observed probe groups, with original data order seed 42. No unused candidate or final-test group is newly retired; no patient-independence or unseen-test claim. Probes never enter rewards or updates.

Lock rollout seeds **45, 46, 47** for both arms, with independent initialization from the original base in each run. Schedule m45, v45, m46, v46, m47, v47 (m=majority, v=frozen legal pseudo-target). Every run has 16 cases and probe cursors 0/16. Frozen greedy is the verified original cache (2/16 stream; 3/16 initially correct probe). One final complete state per run; no best-checkpoint selection or demonstration restart.

Primary metric remains original reasoning/free generation plus strict P2 parser. Success for this round requires, for **every one of the three seeds**:
1. candidate greedy-before correct count strictly exceeds the frozen greedy count;
2. candidate greedy-before correct count strictly exceeds same-seed majority-control count;
3. candidate final probe retains **all three identical initially correct groups** (not merely an equal total).

After/current-case scores, parsing rates, all paired gains/harms and every control's probes are reported separately. Positive current-case effects alone, new readout scores, a chosen seed or net-only probe counts do not satisfy success. All seeds/failed trials are retained. This criterion establishes only a development repeatability signal on reused small groups; it does not establish generalization. Further rounds need fresh prespecified seeds and a newly locked single change before outputs are scored.

## Frozen signal and diagnostic go/no-go

Before any main run, generate one immutable label-free signal for every stream group, and a separate diagnostic file for all probes. Use the exact P3 answer-only instruction and prefix, full identifier conditional log-probability sum, no EOS/length normalization, lexicographic tie rule. Input includes original images/question/options, no sampled reasoning. Preserve RNG/modes and parameters. The training signal file contains stream groups only; save and compare a private snapshot in every candidate run and on resume.

Then independently grade the closed frozen signal on the already observed development labels, and apply its binary rewards counterfactually to the four saved P3 RL rollout pools. This is an explicitly supervised **offline development audit**, never a training-label path. Its gate was fixed before the new reference output was scored:
- reference stream correctness at least 3/16;
- correct-negative-advantage rate strictly below the P3 four-RL aggregate baseline;
- wrong-parsed-positive-reward rate at least 10 percentage points below that baseline.

Report both reward sources' numerators/denominators on the identical saved rollout pool. If the gate fails, do not launch this main matrix; record the negative diagnostic and use the next hourly turn to select a different evidence-based single hypothesis within the remaining total budget. No silent prompt search or threshold fitting. Offline labels used to choose a development hypothesis make this a development procedure, not prospective independent-data validation.

Main labels are joined only after all six main predictions/configs/scopes and final probes close and owned processes end. Never grade one main seed and use it to decide whether to run the other seeds.

## Execution and hourly continuation

Round-one GPU jobs are bounded by a 900-second reference stage and at most 7200 seconds per main trajectory, with the +10.5-hour absolute round GPU stop enforced by the existing supervisor. Expected main cost from P3 is approximately six hours; reserve at least 20% and final report/publication time before launch. Each full RL state is roughly 45 GB; reserve at least 400 GB and verify actual mount/free space/write probe before writes. Do not touch old P2/P3 artifacts or unrelated processes.

The hourly heartbeat reads private optimization_control.json and the latest controller/receipt. It checks once per hour, avoids duplicate launch, computes cumulative GPU process budget across all archived/current jobs, and advances completed stages. It is quiet while unchanged. It reports completed stages, material failures, exhausted budget or an honestly satisfied criterion. When a round ends, publish sanitized source/config/results/report on the new branch via the required proxy and verify anonymous access. Keep all negative rounds. Stop new GPU launches at 24 process-hours or any earlier safety/deadline limit; no automatic budget renewal. After stable development success and complete public delivery, disable the heartbeat.

Raw questions/images/completions/labels/IDs, checkpoints, full logs, secrets and private absolute paths remain private. Source lineage begins at P3 delivery de25c016cfb3e46c372843b293cc5a5afc1b0632. Branch: codex/p4-reference-reward.

The research context is [TTRL](https://arxiv.org/abs/2504.16084) and [SPINE](https://arxiv.org/abs/2511.17938). Their published claims do not override these medical-development measurements; this remains an independent implementation.
