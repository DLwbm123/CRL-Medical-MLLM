"""Small numerical and label-isolation checks; no medical data or GPU required."""
import json

import torch
from torch.distributions import Categorical, kl_divergence

from core import consensus_rewards, extract_answer, objective, prompt_messages, select_and_band, token_statistics


def test_statistics_and_gradient():
    torch.manual_seed(13)
    values = torch.randn(7, 11, requires_grad=True)
    reference = torch.randn(7, 11)
    ids = torch.arange(7)
    lp, entropy, kl = token_statistics(values, ids, reference, chunk_tokens=2)
    distribution, ref_distribution = Categorical(logits=values), Categorical(logits=reference)
    assert torch.allclose(lp, distribution.log_prob(ids), atol=1e-6)
    assert torch.allclose(entropy, distribution.entropy(), atol=1e-6)
    assert torch.allclose(kl, kl_divergence(distribution, ref_distribution), atol=1e-6)
    actual = torch.autograd.grad((lp + 0.13 * entropy + 0.07 * kl).sum(), values, retain_graph=True)[0]
    expected = torch.autograd.grad((distribution.log_prob(ids) + 0.13 * distribution.entropy() + 0.07 * kl_divergence(distribution, ref_distribution)).sum(), values)[0]
    assert torch.allclose(actual, expected, atol=1e-6), "Chunked/recomputed gradients differ from dense reference"


def test_selection_and_band():
    mask, lower, upper, threshold = select_and_band(torch.tensor([0.0, 0.0, 2.0, 3.0], requires_grad=True))
    assert mask.tolist() == [False, False, True, True]
    assert torch.isclose(upper, torch.tensor(2.5))
    assert torch.isclose(lower, torch.tensor(2.5 - 1.4826 * 0.5))
    assert not lower.requires_grad and not upper.requires_grad and not threshold.requires_grad
    same, _, _, t = select_and_band(torch.tensor([2.0, 2.0]))
    assert same.all() and t == 2
    one, _, _, _ = select_and_band(torch.tensor([1.0]))
    assert one.tolist() == [True]


def test_masked_update_and_global_normalization():
    logits = torch.tensor([[12.0, -5.0, -5.0], [10.0, -5.0, -5.0], [0.8, 0.1, -0.2], [0.5, 0.2, -0.1]], requires_grad=True)
    target = torch.tensor([0, 0, 1, 2])
    lp, h, kl = token_statistics(logits, target, logits.detach() + torch.tensor([0.1, 0.0, -0.1]))
    mask, lower, upper, _ = select_and_band(h)
    assert mask.tolist() == [False, False, True, True]
    cfg = {"method": "SPINE", "clip_epsilon": 0.2, "beta_lower": 0.01, "beta_upper": 0.01, "kl_coefficient": 0.01}
    loss, _ = objective(lp, lp.detach(), h, kl, 1.0, mask, lower, upper, 4, 2, cfg)
    grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    assert torch.count_nonzero(grad[:2]) == 0 and torch.count_nonzero(grad[2:]) > 0
    # Splitting micro-batches with shared global denominators must preserve loss.
    pieces = [objective(lp[s], lp[s].detach(), h[s], kl[s], 1.0, mask[s], lower, upper, 4, 2, cfg)[0] for s in [slice(0, 2), slice(2, 4)]]
    assert torch.allclose(sum(pieces), loss, atol=1e-7)
    cfg["method"] = "TTRL"
    loss_all, metrics = objective(lp, lp.detach(), h, kl, 1.0, torch.ones_like(mask), lower, upper, 4, 4, cfg)
    assert metrics["band"] == 0
    grad_all = torch.autograd.grad(loss_all, logits)[0]
    assert torch.count_nonzero(grad_all[1]) > 0


def test_label_free_io_and_consensus():
    row = dict(id="example", source_id="example", dataset="synthetic", split="test", question="Which option?", options={"A": "one", "B": "two"}, images=["first", "second"], image_paths=["/first", "/second"], language="en")
    messages = prompt_messages(row)
    assert [entry["image"] for entry in messages[0]["content"][:-1]] == ["/first", "/second"]
    try:
        prompt_messages(dict(row, label="A"))
    except ValueError:
        pass
    else:
        raise AssertionError("Ground-truth field accepted")
    texts = ["Final answer: A", "Final answer: (B).", "Final answer: B", "Final answer: A", "no marker here", "Final answer: Z", "Final answer: A", "Final answer: B"]
    rewards, advantages, detail = consensus_rewards(texts, row["options"])
    assert detail["winner"] == "A" and detail["valid"] == 6
    assert rewards.tolist() == [1, 0, 0, 1, 0, 0, 1, 0]
    assert abs(float(advantages.mean())) < 1e-6
    assert extract_answer("最终答案：左肺。") == "左肺"
    assert extract_answer("Final answer:  LEFT   LUNG.") == "left lung"
    options = {"A": "example one", "D": "example two"}
    assert extract_answer("Final answer: D. Figure D", options) == "D"
    assert extract_answer("The correct answer is: A. Figure C", options) == "A"
    assert extract_answer("Answer:A", options) == "A"
    assert extract_answer("The correct answer is: \n\nFinal answer: A. Figure C", options) == "A"
    assert extract_answer("A lengthy explanation without an answer marker", options) is None
    _, equal_advantages, _ = consensus_rewards(["Final answer: A"] * 8, row["options"])
    assert torch.count_nonzero(equal_advantages) == 0
    try:
        consensus_rewards(["unparseable"] * 8, row["options"])
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid completions formed a rewarded consensus")


if __name__ == "__main__":
    tests = [test_statistics_and_gradient, test_selection_and_band, test_masked_update_and_global_normalization, test_label_free_io_and_consensus]
    for test in tests:
        test()
    print(json.dumps({"status": "passed", "tests": [test.__name__ for test in tests]}))
