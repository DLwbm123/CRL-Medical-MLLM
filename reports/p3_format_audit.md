# P3 output-format and legal-option readout audit

The P2 original four-method, candidate and probe aggregates/transitions were reproduced exactly. All 145 P2 invalid records and 40 stratified parsed controls were reviewed privately. Classification used answer-marker/ending evidence, without guessing intent from truth labels or clinical plausibility. P3 uses the same locked offline rules.

## Failure counts

Counts cover stream greedy outputs, rollouts and every retained probe output. They are output records, not independent patients. P2 has three probe cursors; P3 has two, so their raw failure totals are not directly comparable. Missing denotes missing final marker/explicit choice; syntax denotes explicit unsupported choice syntax; cap denotes reaching 2048 tokens without a parseable final choice. A syntax failure can include an explicit identifier followed by text, while the extra parser recovers only a bare identifier. Categories are exclusive under the locked priority order; a record counted as syntax can also have reached the cap.

| Stage | Method / seed | Outputs | Invalid | Missing | Syntax | Cap | Conflict | Prose | Illegal ID | Other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P2 | Frozen greedy / 42 | 64 | 11 | 4 | 7 | 0 | 0 | 0 | 0 | 0 |
| P2 | Frozen SC-8 / 42 | 176 | 39 | 24 | 13 | 1 | 0 | 1 | 0 | 0 |
| P2 | TTRL / 42 | 208 | 38 | 26 | 11 | 1 | 0 | 0 | 0 | 0 |
| P2 | SPINE / 42 | 208 | 57 | 34 | 18 | 4 | 0 | 0 | 1 | 0 |
| P3 | Frozen greedy / shared cache | 48 | 8 | 3 | 5 | 0 | 0 | 0 | 0 | 0 |
| P3 | Frozen SC-8 / 43 | 160 | 30 | 18 | 11 | 0 | 0 | 1 | 0 | 0 |
| P3 | TTRL / 43 | 192 | 32 | 21 | 11 | 0 | 0 | 0 | 0 | 0 |
| P3 | SPINE / 43 | 192 | 41 | 18 | 20 | 2 | 0 | 1 | 0 | 0 |
| P3 | Frozen SC-8 / 44 | 160 | 36 | 23 | 13 | 0 | 0 | 0 | 0 | 0 |
| P3 | TTRL / 44 | 192 | 29 | 20 | 9 | 0 | 0 | 0 | 0 | 0 |
| P3 | SPINE / 44 | 192 | 26 | 19 | 6 | 0 | 0 | 1 | 0 | 0 |

## Original versus eval-only syntax parser

`marked_choice_v1_eval_only` adds explicit, unambiguous bare identifiers after marked answer fields, including newline, Markdown and boxed syntax. Conflicting marked identifiers abstain. It does not infer clinical answers or treat option lists/reasoning candidates as final answers. The P2 original parser and consensus rewards remain the primary scoring/training protocol; P2 reports are unchanged. The same extra rules apply to every method, seed, output type and probe cursor. Parsed/correct entries below show original → extra syntax.

| Stage | Method / seed | Output | Cursor | N | Parsed | Correct | New parsed | New invalid | Invalid→correct | Correct→invalid | Correct→parsed wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P2 | Frozen greedy / 42 | before | all | 16 | 14 → 15 | 2 → 2 | 1 | 0 | 0 | 0 | 0 |
| P2 | Frozen greedy / 42 | probe | 0 | 16 | 13 → 13 | 3 → 3 | 0 | 0 | 0 | 0 | 0 |
| P2 | Frozen greedy / 42 | probe | 8 | 16 | 13 → 13 | 3 → 3 | 0 | 0 | 0 | 0 | 0 |
| P2 | Frozen greedy / 42 | probe | 16 | 16 | 13 → 13 | 3 → 3 | 0 | 0 | 0 | 0 | 0 |
| P2 | Frozen SC-8 / 42 | probe | 0 | 16 | 13 → 13 | 3 → 3 | 0 | 0 | 0 | 0 | 0 |
| P2 | Frozen SC-8 / 42 | probe | 8 | 16 | 13 → 13 | 3 → 3 | 0 | 0 | 0 | 0 | 0 |
| P2 | Frozen SC-8 / 42 | probe | 16 | 16 | 13 → 13 | 3 → 3 | 0 | 0 | 0 | 0 | 0 |
| P2 | Frozen SC-8 / 42 | rollout | all | 128 | 98 → 101 | 14 → 14 | 3 | 0 | 0 | 0 | 0 |
| P2 | TTRL / 42 | after | all | 16 | 16 → 16 | 5 → 5 | 0 | 0 | 0 | 0 | 0 |
| P2 | TTRL / 42 | before | all | 16 | 14 → 14 | 5 → 5 | 0 | 0 | 0 | 0 | 0 |
| P2 | TTRL / 42 | probe | 0 | 16 | 13 → 13 | 3 → 3 | 0 | 0 | 0 | 0 | 0 |
| P2 | TTRL / 42 | probe | 8 | 16 | 12 → 13 | 3 → 3 | 1 | 0 | 0 | 0 | 0 |
| P2 | TTRL / 42 | probe | 16 | 16 | 13 → 14 | 3 → 3 | 1 | 0 | 0 | 0 | 0 |
| P2 | TTRL / 42 | rollout | all | 128 | 102 → 102 | 17 → 17 | 0 | 0 | 0 | 0 | 0 |
| P2 | SPINE / 42 | after | all | 16 | 11 → 11 | 3 → 3 | 0 | 0 | 0 | 0 | 0 |
| P2 | SPINE / 42 | before | all | 16 | 12 → 13 | 3 → 3 | 1 | 0 | 0 | 0 | 0 |
| P2 | SPINE / 42 | probe | 0 | 16 | 13 → 13 | 3 → 3 | 0 | 0 | 0 | 0 | 0 |
| P2 | SPINE / 42 | probe | 8 | 16 | 9 → 11 | 1 → 1 | 2 | 0 | 0 | 0 | 0 |
| P2 | SPINE / 42 | probe | 16 | 16 | 11 → 13 | 2 → 2 | 2 | 0 | 0 | 0 | 0 |
| P2 | SPINE / 42 | rollout | all | 128 | 95 → 98 | 14 → 15 | 3 | 0 | 1 | 0 | 0 |
| P3 | Frozen greedy / shared cache | before | all | 16 | 14 → 15 | 2 → 2 | 1 | 0 | 0 | 0 | 0 |
| P3 | Frozen greedy / shared cache | probe | 0 | 16 | 13 → 13 | 3 → 3 | 0 | 0 | 0 | 0 | 0 |
| P3 | Frozen greedy / shared cache | probe | 16 | 16 | 13 → 13 | 3 → 3 | 0 | 0 | 0 | 0 | 0 |
| P3 | Frozen SC-8 / 43 | probe | 0 | 16 | 13 → 13 | 3 → 3 | 0 | 0 | 0 | 0 | 0 |
| P3 | Frozen SC-8 / 43 | probe | 16 | 16 | 13 → 13 | 3 → 3 | 0 | 0 | 0 | 0 | 0 |
| P3 | Frozen SC-8 / 43 | rollout | all | 128 | 104 → 108 | 21 → 21 | 4 | 0 | 0 | 0 | 0 |
| P3 | TTRL / 43 | after | all | 16 | 13 → 14 | 3 → 3 | 1 | 0 | 0 | 0 | 0 |
| P3 | TTRL / 43 | before | all | 16 | 13 → 13 | 1 → 1 | 0 | 0 | 0 | 0 | 0 |
| P3 | TTRL / 43 | probe | 0 | 16 | 13 → 13 | 3 → 3 | 0 | 0 | 0 | 0 | 0 |
| P3 | TTRL / 43 | probe | 16 | 16 | 13 → 15 | 4 → 5 | 2 | 0 | 1 | 0 | 0 |
| P3 | TTRL / 43 | rollout | all | 128 | 108 → 111 | 14 → 15 | 3 | 0 | 1 | 0 | 0 |
| P3 | SPINE / 43 | after | all | 16 | 12 → 13 | 4 → 4 | 1 | 0 | 0 | 0 | 0 |
| P3 | SPINE / 43 | before | all | 16 | 12 → 13 | 2 → 2 | 1 | 0 | 0 | 0 | 0 |
| P3 | SPINE / 43 | probe | 0 | 16 | 13 → 13 | 3 → 3 | 0 | 0 | 0 | 0 | 0 |
| P3 | SPINE / 43 | probe | 16 | 16 | 10 → 15 | 2 → 3 | 5 | 0 | 1 | 0 | 0 |
| P3 | SPINE / 43 | rollout | all | 128 | 104 → 110 | 11 → 11 | 6 | 0 | 0 | 0 | 0 |
| P3 | Frozen SC-8 / 44 | probe | 0 | 16 | 13 → 13 | 3 → 3 | 0 | 0 | 0 | 0 | 0 |
| P3 | Frozen SC-8 / 44 | probe | 16 | 16 | 13 → 13 | 3 → 3 | 0 | 0 | 0 | 0 | 0 |
| P3 | Frozen SC-8 / 44 | rollout | all | 128 | 98 → 104 | 17 → 17 | 6 | 0 | 0 | 0 | 0 |
| P3 | TTRL / 44 | after | all | 16 | 13 → 13 | 3 → 3 | 0 | 0 | 0 | 0 | 0 |
| P3 | TTRL / 44 | before | all | 16 | 12 → 13 | 5 → 5 | 1 | 0 | 0 | 0 | 0 |
| P3 | TTRL / 44 | probe | 0 | 16 | 13 → 13 | 3 → 3 | 0 | 0 | 0 | 0 | 0 |
| P3 | TTRL / 44 | probe | 16 | 16 | 16 → 16 | 4 → 4 | 0 | 0 | 0 | 0 | 0 |
| P3 | TTRL / 44 | rollout | all | 128 | 109 → 112 | 18 → 18 | 3 | 0 | 0 | 0 | 0 |
| P3 | SPINE / 44 | after | all | 16 | 15 → 15 | 4 → 4 | 0 | 0 | 0 | 0 | 0 |
| P3 | SPINE / 44 | before | all | 16 | 13 → 13 | 3 → 3 | 0 | 0 | 0 | 0 | 0 |
| P3 | SPINE / 44 | probe | 0 | 16 | 13 → 13 | 3 → 3 | 0 | 0 | 0 | 0 | 0 |
| P3 | SPINE / 44 | probe | 16 | 16 | 15 → 15 | 4 → 4 | 0 | 0 | 0 | 0 | 0 |
| P3 | SPINE / 44 | rollout | all | 128 | 110 → 112 | 21 → 21 | 2 | 0 | 0 | 0 | 0 |

Consensus is recomputed uniformly for this diagnostic using the extra parser and original majority/tie rule. Stored original rewards and actor updates are never recomputed or replaced.

| Stage | Method / seed | Groups | Original vote correct | Extra-syntax vote correct |
| --- | --- | --- | --- | --- |
| P2 | Frozen SC-8 / 42 | 16 | 1 | 1 |
| P2 | TTRL / 42 | 16 | 2 | 2 |
| P2 | SPINE / 42 | 16 | 1 | 1 |
| P3 | Frozen SC-8 / 43 | 16 | 1 | 1 |
| P3 | TTRL / 43 | 16 | 1 | 2 |
| P3 | SPINE / 43 | 16 | 1 | 1 |
| P3 | Frozen SC-8 / 44 | 16 | 3 | 1 |
| P3 | TTRL / 44 | 16 | 2 | 2 |
| P3 | SPINE / 44 | 16 | 2 | 1 |

## Fixed legal-option likelihood readout on old P2 actors

All four actors used all 16 old probe groups, the same image/question/options, and an answer-only instruction with assistant prefix `Final answer:`. For identifier k with tokens t₁…tₘ, score(k) = Σⱼ log p(tⱼ | common prefix, t<ⱼ). The complete identifier sequence is scored, without EOS or length normalization. Maximum score wins; ties choose the lexicographically first identifier. Tokenization and option scores remain private. No generated reasoning, truth labels, optimizer or parameter update enters this path; checkpoint loading is read-only and inference preserves RNG/model modes. These are P2 actors, not newly adapted P3 actors.

| Actor | N | Free correct | Free parsed | Legal correct | Legal parsed | Parsed wrong→correct | Invalid→correct | Correct→parsed wrong | Correct→invalid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_base | 16 | 3 | 13 | 5 | 16 | 2 | 0 | 0 | 0 |
| p2_spine_final | 16 | 2 | 11 | 5 | 16 | 1 | 2 | 0 | 0 |
| p2_spine_midpoint | 16 | 1 | 9 | 5 | 16 | 1 | 3 | 0 | 0 |
| p2_ttrl_final | 16 | 3 | 13 | 5 | 16 | 1 | 1 | 0 | 0 |

Observed identifier token lengths (length: count) are recorded below. The CPU check also covers a two-token identifier and checks the complete summed score rather than a first-token shortcut.

| Actor | Identifier token-length counts |
| --- | --- |
| frozen_base | {'1': 80} |
| p2_spine_final | {'1': 80} |
| p2_spine_midpoint | {'1': 80} |
| p2_ttrl_final | {'1': 80} |

The following subset is diagnostic only: cases initially correct under frozen free generation that became invalid under the indicated actor's old free generation. Every actor was evaluated on the full probe first.

| Actor | Legal correct in lost subset | Subset denominator | Percent |
| --- | --- | --- | --- |
| frozen_base | 0 | 0 | NA |
| p2_spine_final | 1 | 1 | 100.00 |
| p2_spine_midpoint | 2 | 2 | 100.00 |
| p2_ttrl_final | 0 | 0 | NA |

The free-generation and legal-readout protocols are different conditional distributions. A difference in correct counts is not a pure-format-loss estimate. Legal validity alone is not medical competence; preserved or reduced legal-readout accuracy cannot prove retention or forgetting of a particular kind of medical knowledge. Format/reward associations do not identify the cause of any individual update's degradation.
