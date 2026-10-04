"""State/restart, probability convention, invalid reward and label-flow acceptance."""
import contextlib
import copy
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GenerationConfig

from core import objective, prompt_messages, select_and_band, token_statistics
from continual import Engine, group_rewards, load_input
from state import PersistentAdam, atomic_json, inference_without_state_change, load_checkpoint, restore_rng, rng_state, save_checkpoint

CFG = {"method": "SPINE", "learning_rate": 1e-3, "adam_betas": [0.9, 0.999], "adam_epsilon": 1e-8,
       "weight_decay": 0, "max_grad_norm": 1, "clip_epsilon": 0.2,
       "beta_lower": 0.01, "beta_upper": 0.01, "kl_coefficient": 0.01}


class TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = torch.nn.Linear(4, 8)
        self.text = torch.nn.Embedding(11, 8)
        self.head = torch.nn.Linear(8, 11)

    def forward(self, tokens, image):
        return self.head(torch.tanh(self.text(tokens) + self.visual(image)))


def initialize():
    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)
    actor = TinyActor()
    reference = copy.deepcopy(actor).requires_grad_(False).eval()
    initial = {name: p.detach().clone() for name, p in reference.named_parameters()}
    state = PersistentAdam(actor, CFG)
    assert not ({id(p) for p in reference.parameters()} & {id(p) for group in state.optimizer.param_groups for p in group["params"]})
    return actor, reference, initial, state


def small_step(actor, reference, state):
    inputs = torch.tensor([random.randrange(11) for _ in range(4)])
    image = torch.tensor(np.random.standard_normal(4), dtype=torch.float32)
    with torch.no_grad():
        distribution = actor(inputs, image).softmax(-1)
        responses = [torch.multinomial(distribution, 1).squeeze(-1) for _ in range(8)]
        completions = ["Final answer: " + "ABC"[int(response[-1]) % 3] for response in responses]
        rewards, advantages, vote = group_rewards(completions, {"A": "x", "B": "y", "C": "z"})
        old_logits = actor(inputs, image)
        reference_logits = reference(inputs, image)
        masks, bands, old_logp = [], [], []
        for response in responses:
            lp, entropy, _ = token_statistics(old_logits, response)
            mask, lower, upper, _ = select_and_band(entropy)
            masks.append(mask)
            bands.append((lower, upper))
            old_logp.append(lp)
    state.zero_grad()
    selected = sum(int(mask.sum()) for mask in masks)
    for i, response in enumerate(responses):
        lp, entropy, kl = token_statistics(actor(inputs, image), response, reference_logits)
        loss, _ = objective(lp, old_logp[i], entropy, kl, advantages[i], masks[i], *bands[i], 32, selected, CFG)
        loss.backward()
    result = state.step()
    assert result["gradient_norms"]["vision"] > 0 and result["gradient_norms"]["language"] > 0
    return {"responses": [response.tolist() for response in responses], "winner": vote["winner"]}


def toy_payload(actor, reference, initial, state, outputs):
    assert all(torch.equal(parameter, initial[name]) and parameter.grad is None for name, parameter in reference.named_parameters())
    assert any(not torch.equal(parameter, initial[name]) for name, parameter in actor.named_parameters())
    return {**state.pack(), "rng": rng_state(), "cursor": len(outputs), "processed_ids": [str(i) for i in range(len(outputs))],
            "configuration": CFG, "code_commit": "synthetic-test", "model_revision": "synthetic-fixed-initialization",
            "manifest_sha256": "synthetic-manifest", "method": "SPINE", "run_id": "toy", "outputs": outputs,
            "reference_unchanged": True}


def compare_values(a, b, atol=0, rtol=0, path="state", summary=None):
    if summary is None:
        summary = {"tensor_count": 0, "tensor_elements": 0, "nonexact_tensors": 0, "max_abs_difference": 0.0}
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype, path
        summary["tensor_count"] += 1
        summary["tensor_elements"] += a.numel()
        if not torch.equal(a, b):
            assert a.is_floating_point() and torch.allclose(a, b, atol=atol, rtol=rtol), path
            summary["nonexact_tensors"] += 1
            summary["max_abs_difference"] = max(summary["max_abs_difference"], float((a - b).abs().max()))
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), path
    elif isinstance(a, dict):
        assert set(a) == set(b), path
        for key in a:
            compare_values(a[key], b[key], atol, rtol, f"{path}/{key}", summary)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            compare_values(x, y, atol, rtol, f"{path}/{i}", summary)
    else:
        assert a == b, path
    return summary


def restart_half(folder):
    torch.set_num_threads(1)
    actor, reference, initial, state = initialize()
    saved = load_checkpoint(folder, CFG, "synthetic-manifest", "SPINE")
    state.load(saved)
    restore_rng(saved["rng"])
    outputs = list(saved["outputs"])
    for _ in range(saved["cursor"], 10):
        outputs.append(small_step(actor, reference, state))
    save_checkpoint(folder, toy_payload(actor, reference, initial, state, outputs))


def test_restart(folder):
    torch.set_num_threads(1)
    actor, reference, initial, state = initialize()
    outputs = [small_step(actor, reference, state) for _ in range(10)]
    uninterrupted = copy.deepcopy(toy_payload(actor, reference, initial, state, outputs))
    assert state.updates == 10
    assert any(torch.count_nonzero(value["exp_avg"]) > 0 for value in state.optimizer.state.values())
    actor, reference, initial, state = initialize()
    outputs = [small_step(actor, reference, state) for _ in range(5)]
    save_checkpoint(folder, toy_payload(actor, reference, initial, state, outputs))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", P2_RESTART_FOLDER=str(folder))
    # A genuinely fresh interpreter; task paths and selection stay out of argv.
    script = "import os,sys,subprocess\nfrom pathlib import Path\nprint(subprocess.check_output(['ps','-ww','-p',str(os.getpid()),'-o','args='],text=True).strip())\nsys.path.insert(0,str(Path(os.environ['P0_ROOT'])/'implementation'))\nfrom test_continual import restart_half\nrestart_half(Path(os.environ['P2_RESTART_FOLDER']))\n"
    child = subprocess.run([sys.executable, "-"], input=script, text=True, env=env,
                           capture_output=True, timeout=180)
    assert child.returncode == 0, child.stderr
    assert not any(word.lower() in child.stdout.lower() for word in json.loads(os.environ.get("P2_FORBIDDEN_ARGV", "[]")))
    restored = load_checkpoint(folder, CFG, "synthetic-manifest", "SPINE")
    comparison = compare_values(uninterrupted, restored)
    assert restored["cursor"] == restored["optimizer_updates"] == 10
    return {"passed": True, "continuous_steps": 10, "restart_after": 5, "fresh_process": True,
            "atol": 0, "rtol": 0, **comparison}


def test_probability():
    torch.manual_seed(27)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=31, n_positions=32, n_embd=16, n_layer=1, n_head=2,
                                     resid_pdrop=0, embd_pdrop=0, attn_pdrop=0, bos_token_id=1, pad_token_id=0, eos_token_id=None)).eval()
    prompt = torch.tensor([[1, 4, 7]])
    with torch.no_grad():
        result = model.generate(prompt, attention_mask=torch.ones_like(prompt),
                                generation_config=GenerationConfig(do_sample=True, temperature=1, top_p=1, top_k=0,
                                                                   max_new_tokens=4, pad_token_id=0, eos_token_id=None),
                                return_dict_in_generate=True, output_scores=True)
        response = result.sequences[0, prompt.shape[1]:]
        sampled = torch.stack([score[0].log_softmax(-1)[token] for score, token in zip(result.scores, response)])
        logits = model(result.sequences).logits[0, prompt.shape[1] - 1:-1]
        recomputed = logits.log_softmax(-1).gather(-1, response[:, None]).squeeze(-1)
        assert torch.allclose(sampled, recomputed, atol=1e-6, rtol=0)
    return {"passed": True, "temperature": 1, "top_p": 1, "top_k": 0, "max_error": float((sampled - recomputed).abs().max()), "atol": 1e-6}


def test_probe_preservation():
    actor, _, _, _ = initialize()
    actor.train()
    actor.visual.eval()
    before = rng_state()
    modes = [module.training for module in actor.modules()]
    parameters = {name: p.detach().clone() for name, p in actor.named_parameters()}
    with inference_without_state_change(actor):
        assert not any(module.training for module in actor.modules())
        random.random()
        np.random.random()
        torch.rand(5)
        actor(torch.tensor([1, 2]), torch.ones(4))
    compare_values(before, rng_state())
    assert modes == [module.training for module in actor.modules()]
    assert all(torch.equal(p, parameters[name]) for name, p in actor.named_parameters())
    return {"passed": True, "rng_restored": True, "all_module_modes_restored": True, "parameters_unchanged": True}


def test_invalid_and_label_boundaries(folder):
    options = {"A": "one", "B": "two"}
    rewards, advantages, vote = group_rewards(["invalid"] * 8, options)
    assert rewards is None and advantages is None and vote["all_unparseable"]
    _, advantages, vote = group_rewards(["Final answer: A"] * 8, options)
    assert vote["zero_advantage_group"] and torch.count_nonzero(advantages) == 0
    rewards, _, vote = group_rewards(["Final answer: B", "Final answer: A"] * 4, options)
    assert vote["tie"] and vote["winner"] == "A" and vote["margin"] == 0
    assert rewards.tolist() == [0, 1] * 4
    row = dict(id="synthetic", source_id="synthetic", dataset="synthetic", split="dev", question="Choose.",
               options=options, images=["first", "second"], image_paths=["/first", "/second"], language="en")
    assert [item["image"] for item in prompt_messages(row)[0]["content"][:-1]] == row["image_paths"]
    (folder / "inputs").mkdir()
    atomic_json(folder / "inputs/example.json", {**row, "label": "A"})
    try:
        load_input(folder, {"input": "inputs/example.json", "id": row["id"]})
    except ValueError:
        pass
    else:
        raise AssertionError("Trainer accepted a label field")
    try:
        load_input(folder, {"input": "evaluation_labels/example.json", "id": row["id"]})
    except ValueError:
        pass
    else:
        raise AssertionError("Trainer accessed a label path")
    # Exercise the production all-invalid branch without a GPU or model download.
    engine = object.__new__(Engine)
    engine.cfg = {"rollouts": 8, "max_new_tokens": 2048}
    engine.method, engine.training, engine.out = "SPINE", True, folder
    engine.state, engine.timings = SimpleNamespace(updates=4), {}
    engine.event = lambda *args, **kwargs: None
    engine.phase = lambda _: contextlib.nullcontext()
    engine.encode = lambda _: {"input_ids": torch.tensor([[1, 2]])}
    engine.generate = lambda _, sample: (torch.tensor([3]), "unparseable" if sample else "Final answer: A")
    engine.update = lambda *_: (_ for _ in ()).throw(AssertionError("All-invalid group reached optimizer"))
    record = engine.case(row, 7, False)
    assert record["id"] == row["id"] and record["index"] == 7
    assert not record["update_applied"] and record["skip_reason"] == "all_unparseable"
    assert record["optimizer_updates_before"] == record["optimizer_updates_after"] == 4
    assert record["before"]["answer"] == record["after"]["answer"] == "A"
    return {"passed": True, "all_invalid_skips_without_dropping_prediction": True,
            "tie_rule_preserved": True, "zero_advantage_identified": True, "label_fields_and_paths_rejected": True}


def main():
    folder = Path(os.environ["P2_TEST_DIR"])
    folder.mkdir(parents=True, exist_ok=False)
    result = {"restart": test_restart(folder / "checkpoints"), "probability": test_probability(),
              "probe": test_probe_preservation(), "boundaries": test_invalid_and_label_boundaries(folder)}
    result["passed"] = all(value["passed"] for value in result.values())
    atomic_json(folder / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
