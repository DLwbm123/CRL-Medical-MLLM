"""Synthetic checks of label isolation, abstention, optimizer skips and admission."""
import contextlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from continual import Engine, group_rewards, validate_development_configuration
from core import consensus_rewards
from state import atomic_json
from verified_reward import (admitted_seconds, agreed_target, checked_targets, judge_messages,
                             parse_judgment, qualification_gate, verified_group_rewards)


def main():
    row = dict(id="synthetic", source_id="synthetic", dataset="synthetic", split="dev", question="Choose.",
               options={"A": "one", "B": "two"}, images=["first", "second"], image_paths=["/first", "/second"], language="en")
    forward, mapping = judge_messages(row)
    backward, reversed_mapping = judge_messages(row, True)
    assert mapping == {"A": "A", "B": "B"} and reversed_mapping == {"A": "B", "B": "A"}
    assert [x["image"] for x in forward[0]["content"][:-1]] == row["image_paths"]
    try: judge_messages(dict(row, label="B"))
    except ValueError: pass
    else: raise AssertionError("Verifier accepted a clinical label")
    text = 'reasoning\n{"choice":"B","visual_support":"supported","clinical_consistency":"consistent"}'
    a = parse_judgment(text, mapping)
    b = parse_judgment(text.replace('"B"', '"A"'), reversed_mapping)
    assert agreed_target([a, b]) == "B"
    for bad in ['{"choice":[],"visual_support":"supported","clinical_consistency":"consistent"}', text+'\nextra', text.replace('"B"','"C"'), '{}']:
        assert parse_judgment(bad, mapping) is None
    assert agreed_target([a, None]) is None
    assert agreed_target([a, dict(b, visual_support="uncertain")]) is None
    assert agreed_target([a, dict(b, choice="A")]) is None
    completions = ["Final answer: A"]*6 + ["Final answer: B"]*2
    old, old_adv, _ = group_rewards(completions, row["options"])
    expected, expected_adv, _ = consensus_rewards(completions, row["options"])
    assert torch.equal(old, expected) and torch.equal(old_adv, expected_adv)
    rewards, advantages, vote = verified_group_rewards(completions, row["options"], "B")
    assert rewards.tolist() == [0.]*6 + [1.]*2 and torch.all(advantages[:6] < 0) and torch.all(advantages[6:] > 0)
    rejected, adv, vote = verified_group_rewards(completions, row["options"], None)
    assert not torch.count_nonzero(rejected) and not torch.count_nonzero(adv) and vote["skip_update_reason"] == "verifier_abstained"
    manifest = {"stream": [{"id": "synthetic"}], "teacher": {"revision": "pinned"}}
    signal = {"labels_read": False, "weights_unchanged": True, "sealed": True, "manifest_sha256": "fixed", "teacher_revision": "pinned",
              "stream": [{"id": "synthetic", "target": "B", "option_keys": ["A", "B"], "judgments": [a, b]}]}
    assert checked_targets(signal, manifest, "fixed") == {"synthetic": "B"}
    for changed in [dict(signal, labels_read=True), dict(signal, sealed=False), dict(signal, teacher_revision="other")]:
        try: checked_targets(changed, manifest, "fixed")
        except ValueError: pass
        else: raise AssertionError("Changed verifier provenance accepted")
    with tempfile.TemporaryDirectory() as temporary:
        engine = object.__new__(Engine)
        engine.cfg = {"rollouts": 8, "max_new_tokens": 2048, "reward_source": "frozen_visual", "zero_advantage_policy": "skip"}
        engine.method, engine.training, engine.out = "TTRL", True, Path(temporary)
        engine.reward_targets = {"synthetic": None}
        engine.state, engine.timings = SimpleNamespace(updates=4), {}
        engine.phase = lambda _: contextlib.nullcontext()
        engine.event = lambda *args, **kwargs: None
        engine.encode = lambda _: {"input_ids": torch.tensor([[1, 2]])}
        engine.generate = lambda _, sample: (torch.tensor([3]), "Final answer: A")
        engine.update = lambda *_: (_ for _ in ()).throw(AssertionError("Skipped group reached optimizer or Adam momentum"))
        record = engine.case(row, 0, False)
        assert not record["update_applied"] and record["skip_reason"] == "verifier_abstained" and record["optimizer_updates_after"] == 4
        engine.reward_targets["synthetic"] = "B"
        record = engine.case(row, 0, False)
        assert not record["update_applied"] and record["skip_reason"] == "zero_advantage"
        engine.cfg.update(reward_source="majority")
        engine.reward_targets = None
        record = engine.case(row, 0, False)
        assert not record["update_applied"] and record["skip_reason"] == "zero_advantage"
        engine.cfg.pop("zero_advantage_policy")
        engine.method = "SPINE"
        engine.update = lambda *_: {"old_behavior_preserved": True}
        assert engine.case(row, 0, False)["update_applied"]
        engine.reward_targets = {"synthetic": "B"}
        engine.cfg.update(reward_source="frozen_legal")
        assert engine.reward_group(row, ["invalid"]*8)[2]["reward_target"] == "B"
    m = {"correct_negative": {"percent": 60.}, "wrong_positive": {"percent": 90.}}
    v = {"correct_negative": {"percent": 50.}, "wrong_positive": {"percent": 70.}}
    assert qualification_gate(m, v, 4, 8, 1)["go"]
    assert not qualification_gate(m, v, 4, 7, 1)["go"] and not qualification_gate(m, v, 4, 8, 0)["go"]
    assert not qualification_gate(m, m, 4, 8, 1)["go"]
    assert admitted_seconds(120., 120., 100.) and not admitted_seconds(119., 200., 100.)
    cfg = {"seed": 57, "p10_arm": "v", "reward_source": "frozen_visual", "method": "TTRL"}
    locked = {"p10_version": 1, "configurations": {"v": {"57": cfg}}}
    validate_development_configuration(cfg, {}, locked)
    for changed in [dict(cfg, seed=58), dict(cfg, reward_source="labels"), dict(cfg, method="SPINE")]:
        try: validate_development_configuration(changed, {}, locked)
        except RuntimeError: pass
        else: raise AssertionError("Undeclared training change accepted")
    result = {"passed": True, "labels_read": False, "checks": ["two_order_mapping", "strict_json_and_label_boundary", "correct_minority_reward",
              "abstention_never_falls_back", "zero_signal_never_steps_optimizer", "legacy_SPINE_behavior", "frozen_source_and_matrix", "full_cost_and_reserve"]}
    destination = Path(os.environ["P0_ROOT"]) / "outputs" / os.environ["P2_CAMPAIGN"] / "reward_acceptance.json"
    atomic_json(destination, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
