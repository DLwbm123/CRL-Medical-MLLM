# P4 fixed-reference reward: negative diagnostic

The frozen-original-base legal-option pseudo-target failed all three prespecified go/no-go criteria. No candidate or majority-control main trajectory was launched; there is no training improvement result.

| Diagnostic | Original sampled majority | Frozen legal pseudo-target |
| --- | --- | --- |
| Stream target correctness | 6/64 repeated groups (9.375%) | 1/16 unique groups (6.25%); 4/64 repeated groups |
| Correct parsed candidates assigned negative advantage | 40/64 (62.5%) | 45/64 (70.3125%) |
| Wrong parsed candidates among positive-reward responses | 250/274 (91.2409%) | 224/241 (92.9461%) |

The reward comparison uses the identical four closed P3 RL rollout pools: 64 group observations but only 16 unique reused stream groups. The frozen legal signal used the previously locked P3 answer-only prompt and complete identifier log-probability sums, with no labels, optimizer or parameter update. Its predictions closed before the independent audit read previously observed development labels. Probe targets were generated separately and never entered training. The minimum stream criterion was3/16; the other criteria required a lower correct-negative-advantage rate and at least10pp fewer wrong positive rewards. Every criterion failed.

The small change was implemented and CPU checks passed: original reward/parsing behavior, legal-target validation, declared seed/source guards, all-invalid behavior, reward audit denominators, all-three-distinct-seed success and conservative interrupted-GPU accounting. Reference scoring exited0 in28.7261 seconds of whole GPU-worker lifetime; independent CPU audit exited0. The recorded owner/workers ended, GPU process rows cleared and unrelated GPU jobs remained. Main seeds45/46/47 were reserved but not executed; no main outputs exist to select.

The cumulative optimization authorization is24 GPU process-hours. This completed diagnostic consumed28.7261 seconds, leaving86371.2739 seconds before further work. Earlier P2/P3 runs belong to their already completed budgets. This is a negative development finding on known groups; no patient independence, unseen-test generalization or medical competence gain is established. No model/prompt/threshold sweep or true-label training was used to rescue the rejected hypothesis.

Source lineage: P3 delivery de25c016cfb3e46c372843b293cc5a5afc1b0632; P4 protocol implementation d8455136eb49c9f370b0861882f5c476dfce9d3a; executed source915d8f81fe1ff90d20b5331e14a73ecf7a544fde. Protocol and reproduction are in p4_protocol.md and p4_reproduce.md. Aggregate receipts are in p4_results.json; raw medical inputs, generated text, IDs, checkpoints and private logs remain private.

The next independent hypothesis is to remove the observed sampling/loss distribution mismatch before another label-free adaptation comparison. It needs its own fixed source/configuration/control/seeds and budget receipt; P4's gate and results are unchanged.
