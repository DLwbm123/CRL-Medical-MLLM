"""Small runnable checks for the added seed guard and offline diagnoses."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from continual import validate_development_configuration
from core import consensus_rewards, extract_answer
from p3_diagnostics import eval_only_choice, eval_only_consensus, failure_type, reward_direction
from p3_readout import option_log_probabilities, option_tokens
from p3_evaluate import available_run
from state import inference_without_state_change, rng_state
from test_continual import compare_values


def main():
    root = Path(os.environ["P0_ROOT"])
    accepted = json.loads((root / "configs/p2_c.json").read_text())
    acceptance = {"validated_configuration_without_method": {k: v for k, v in accepted.items() if k != "method"}}
    cfg = copy.deepcopy(accepted)
    cfg["seed"] = 43
    validate_development_configuration(cfg, acceptance, {"rollout_seeds": [43, 44]})
    for changed, manifest in [(dict(cfg, seed=45), {"rollout_seeds": [43, 44]}),
                              (dict(cfg, temperature=1.0), {"rollout_seeds": [43, 44]}), (cfg, {})]:
        try:
            validate_development_configuration(changed, acceptance, manifest)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Seed guard accepted an undeclared change")
    validate_development_configuration(accepted, acceptance, {})
    options = {"A": "one", "B": "two"}
    syntax = [("Final answer: A", "A"), ("**Final answer**: **B**", "B"),
              ("Final answer:\nA", "A"), (r"Final answer: \boxed{B}", "B"),
              ("The final answer is:\nB", "B"),
              ("Options: A. one B. two", None), ("I considered A or B.", None),
              ("Final answer: A\nFinal answer: B", None),
              ("Final answer: pulmonary finding", None)]
    for text, expected in syntax:
        assert eval_only_choice(text, options) == expected, text
    assert extract_answer("Final answer:\nA", options) is None
    assert eval_only_consensus(["Final answer:\nB", "Final answer: A"], options) == "A"
    assert eval_only_consensus(["unparsed", "no choice"], options) is None
    assert failure_type("unfinished", options, True) == "token_cap_without_parseable_final_choice"
    assert failure_type("The correct answer is B. listed option text", options, False) == "explicit_choice_unsupported_syntax"
    assert failure_type("The correct answer is (B): listed option text", options, False) == "explicit_choice_unsupported_syntax"
    assert failure_type("Final answer: F.", options, False) == "explicit_choice_outside_legal_options"
    rows = []
    labels = {"x": "B", "y": "B", "z": "B"}
    choices = {key: options for key in labels}
    for index, (identifier, answers) in enumerate([("x", ["A"] * 6 + ["B"] * 2), ("y", ["B"] * 8)]):
        completions = ["Final answer: " + answer for answer in answers]
        rewards, advantages, vote = consensus_rewards(completions, options)
        vote.update(all_unparseable=False, tie=False, top_count=max(vote["counts"].values()),
                    margin=8 if index else 4, zero_advantage_group=bool(torch.count_nonzero(advantages) == 0))
        rows.append({"id": identifier, "index": index, "completions": completions,
                     "rewards": rewards.tolist(), "vote": vote})
    rows.append({"id": "z", "index": 2, "vote": {"answers": [None] * 8, "winner": None,
                 "all_unparseable": True, "valid": 0, "top_count": 0, "margin": 0,
                 "tie": False, "zero_advantage_group": False}})
    audit, _ = reward_direction(rows, choices, labels)
    assert audit["metrics"]["correct_negative_advantage"]["correct"] == 2
    assert audit["metrics"]["correct_negative_advantage"]["total"] == 10
    assert audit["metrics"]["wrong_parsed_positive_reward"]["correct"] == 6
    assert audit["metrics"]["wrong_parsed_positive_reward"]["total"] == 14
    assert audit["counts"]["groups"] == 3 and audit["counts"]["rollouts"] == 24
    assert audit["metrics"]["all_invalid"]["correct"] == 1
    high = next(x for x in audit["fixed_bins"] if x["axis"] == "top_votes" and x["bin"] == "6-8")
    assert high["groups"] == 2 and high["vote_correct"] == 1

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(char) for char in text]

    assert option_tokens(Tokenizer(), "Prefix:", options)["A"] == [32, 65]

    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("fixed", -torch.arange(8, dtype=torch.float32))

        def forward(self, input_ids, attention_mask, use_cache=False, return_dict=True):
            return SimpleNamespace(logits=self.fixed.expand(1, input_ids.shape[-1], 8))

    actor = Actor().train()
    encoded = {"input_ids": torch.tensor([[0, 4, 5]]), "attention_mask": torch.ones(1, 3, dtype=torch.long)}
    before_rng = rng_state()
    with inference_without_state_change(actor):
        scores = option_log_probabilities(actor, encoded, {"A": [1], "B": [2, 3]})
    expected = actor.fixed.log_softmax(-1)
    assert abs(scores["A"] - float(expected[1])) < 1e-6
    assert abs(scores["B"] - float(expected[2] + expected[3])) < 1e-6
    assert actor.training
    compare_values(before_rng, rng_state())
    folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
    manifest = json.loads((folder / "manifest.json").read_text())
    cached_cfg = json.loads((root / "configs/p2_a.json").read_text())
    rows, result, probes = available_run(folder / "runs/main-a", manifest,
                                       (folder / "manifest.sha256").read_text().strip(), cached_cfg,
                                       json.loads((folder / "acceptance.json").read_text()))
    assert len(rows) == result["cursor"] == 16 and set(probes) == {0, 16}
    print(json.dumps({"passed": True, "checks": ["declared-seed-only guard", "label-free syntax examples",
                     "original parser preserved", "reward direction denominators and high-vote errors",
                     "complete multi-token likelihood", "readout RNG/mode preservation",
                     "real cache provenance and probe schema without labels"], "real_labels_read": False}))


if __name__ == "__main__":
    main()
