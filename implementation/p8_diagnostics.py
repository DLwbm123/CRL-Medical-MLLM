"""Label-free read-only diagnostics. Never construct an optimizer or train."""
import contextlib
import gc
import inspect
import json
import os
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from core import INPUT_KEYS, prompt_messages, select_and_band, token_statistics
from continual import forward_response, group_rewards
from p3_readout import ASSISTANT_PREFIX, messages, option_tokens, option_log_probabilities, restore_read_only_actor
from state import atomic_json, inference_without_state_change, rng_state, restore_rng


def exact_kl(logp, reference_logp):
    return (logp.exp() * (logp - reference_logp)).sum(-1)


def components(logp, old_logp, entropy, kl, advantage, mask, lower, upper, tokens, selected, epsilon):
    ratio = (logp - old_logp).exp()
    surrogate = torch.minimum(ratio * advantage, ratio.clamp(1-epsilon, 1+epsilon) * advantage)
    return (-(surrogate * mask).sum()/tokens,
            (((lower-entropy).relu() + (entropy-upper).relu()) * mask).sum()/tokens,
            (kl * mask).sum()/selected)


def gradient_metrics(vectors, band_coefficient, kl_coefficient):
    p, b, k = vectors
    b, kw = b*band_coefficient, k*kl_coefficient
    norm = lambda v: float(torch.linalg.vector_norm(v.double()))
    pn, bn, kn = [norm(v) for v in (p, b, kw)]
    cosine = lambda a, b: float(torch.dot(a.double(), b.double())/(norm(a)*norm(b))) if norm(a) and norm(b) else None
    combined = norm(p+b+kw)
    return {"policy_norm": pn, "band_raw_norm": norm(vectors[1]), "band_weighted_norm": bn,
            "kl_raw_norm": norm(k), "kl_weighted_norm": kn,
            "band_policy_ratio": bn/pn if pn else None, "kl_policy_ratio": kn/pn if pn else None,
            "policy_band_cosine": cosine(p,b), "policy_kl_cosine": cosine(p,kw),
            "band_kl_cosine": cosine(b,kw), "composite_norm": combined,
            "composite_over_sum_norms": combined/(pn+bn+kn) if pn+bn+kn else None}


def snapshot(actor):
    parameters = dict(actor.named_parameters())
    return {"versions": {n: p._version for n,p in parameters.items()},
            "samples": {n: p.detach().reshape(-1)[[0,p.numel()//2,p.numel()-1]].cpu().clone() for n,p in parameters.items()},
            "buffers": {n: b.detach().cpu().clone() for n,b in actor.named_buffers()}}


def check_snapshot(actor, before):
    after = snapshot(actor)
    assert before["versions"] == after["versions"]
    assert before["samples"].keys() == after["samples"].keys()
    assert all(torch.equal(v, after["samples"][n]) for n,v in before["samples"].items())
    assert before["buffers"].keys() == after["buffers"].keys()
    assert all(torch.equal(v, after["buffers"][n]) for n,v in before["buffers"].items())


@contextlib.contextmanager
def diagnostic_state(actor, gradient=False):
    saved_rng = rng_state()
    modes = [(m,m.training) for m in actor.modules()]
    flags = [(p,p.requires_grad) for p in actor.parameters()]
    before = snapshot(actor)
    try:
        actor.train(gradient)
        yield
        check_snapshot(actor, before)
    finally:
        for m, mode in modes: m.training = mode
        for p, flag in flags: p.requires_grad_(flag)
        restore_rng(saved_rng)
        assert [m.training for m,_ in modes] == [mode for _,mode in modes]
        assert [p.requires_grad for p,_ in flags] == [flag for _,flag in flags]


def encode(processor, row, answer_only):
    if set(row) != INPUT_KEYS:
        raise ValueError("Diagnostic input is outside the label-free allowlist")
    prompt = processor.apply_chat_template(messages(row) if answer_only else prompt_messages(row),
                                            tokenize=False, add_generation_prompt=True)
    if answer_only: prompt += ASSISTANT_PREFIX
    images = []
    for path in row["image_paths"]:
        with Image.open(path) as image: images.append(image.convert("RGB"))
    encoded = processor(text=[prompt], images=images, return_tensors="pt").to("cuda:0")
    if len(encoded["image_grid_thw"]) != len(images):
        raise ValueError("Image count/order mismatch")
    return encoded, option_tokens(processor.tokenizer, prompt, row["options"]) if answer_only else None


@torch.no_grad()
def first_distribution(actor, encoded):
    kwargs = dict(encoded, use_cache=False, return_dict=True)
    if "logits_to_keep" in inspect.signature(actor.forward).parameters: kwargs["logits_to_keep"] = 1
    return actor(**kwargs).logits[0,-1].float().log_softmax(-1).cpu()


def restore(actor, specification, revision):
    path = Path(specification["state_path"])
    saved = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    for key in ["code_commit", "model_revision", "manifest_sha256", "method", "run_id", "cursor", "configuration"]:
        if saved[key] != specification["metadata"][key]:
            raise ValueError("Actor checkpoint differs: " + key)
    del saved
    gc.collect()
    return restore_read_only_actor(actor, path, revision, 16)


def local_gradients(actor, processor, pool, references, subset, cfg, check_deadline):
    assert cfg["beta_lower"] == cfg["beta_upper"]
    parameters = dict(actor.named_parameters())
    selected_parameters = [parameters[n] for n in subset]
    results = []
    with diagnostic_state(actor, gradient=True):
        actor.requires_grad_(False)
        for p in selected_parameters: p.requires_grad_(True)
        actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for group_index, group in enumerate(pool):
            check_deadline()
            started = time.monotonic()
            encoded, _ = encode(processor, group["input"], False)
            responses = [torch.tensor(ids, device="cuda:0") for ids in group["response_token_ids"]]
            rewards, advantages, vote = group_rewards(group["completions"], group["input"]["options"])
            if advantages is None: advantages = torch.zeros(len(responses))
            cached = []
            with torch.no_grad():
                for response in responses:
                    check_deadline()
                    logits = forward_response(actor, encoded, response)
                    logp, entropy, _ = token_statistics(logits, response, chunk_tokens=cfg["statistics_chunk_tokens"])
                    mask, lower, upper, _ = select_and_band(entropy, cfg["histogram_bins"])
                    cached.append((logp.detach(), mask, lower, upper))
                    del logits
            total = sum(len(r) for r in responses)
            selected = sum(int(item[1].sum()) for item in cached)
            vectors = [[torch.zeros(p.numel(), dtype=torch.float32) for p in selected_parameters] for _ in range(3)]
            loss_sums, max_error = [0.,0.,0.], 0.
            for slot, (response, item) in enumerate(zip(responses, cached)):
                check_deadline()
                logits = forward_response(actor, encoded, response)
                logp, entropy, kl = token_statistics(logits, response, references[group_index][slot], cfg["statistics_chunk_tokens"])
                max_error = max(max_error, float((logp.detach()-item[0]).abs().max()))
                assert max_error <= cfg["behavior_recompute_atol"]
                assert torch.equal(select_and_band(entropy, cfg["histogram_bins"])[0], item[1])
                losses = components(logp, item[0], entropy, kl, advantages[slot].to("cuda:0"),
                                    item[1], item[2], item[3], total, selected, cfg["clip_epsilon"])
                for component, loss in enumerate(losses):
                    loss_sums[component] += float(loss.detach())
                    gradients = torch.autograd.grad(loss, selected_parameters, retain_graph=component < 2, allow_unused=True)
                    for i, gradient in enumerate(gradients):
                        if gradient is not None: vectors[component][i].add_(gradient.detach().float().cpu().reshape(-1))
                    del gradients
                del logits, logp, entropy, kl, losses
            scopes = [(n,[i]) for i,n in enumerate(subset)] + [
                ("vision",[i for i,n in enumerate(subset) if n.startswith("visual.")]),
                ("language",[i for i,n in enumerate(subset) if not n.startswith("visual.")]),
                ("all_selected",list(range(len(subset))))]
            for scope, indices in scopes:
                values = [torch.cat([v[i] for i in indices]) for v in vectors]
                results.append({"group": group["alias"], "module": scope, "tokens": total,
                                "selected_tokens": selected, "zero_advantage": bool(torch.equal(advantages,torch.zeros_like(advantages))),
                                "all_invalid": vote["all_unparseable"], "band_coefficient": cfg["beta_lower"],
                                "kl_coefficient": cfg["kl_coefficient"], "policy_loss": loss_sums[0],
                                "band_raw_loss": loss_sums[1], "kl_raw_loss": loss_sums[2],
                                "recompute_max_error": max_error,
                                **gradient_metrics(values,cfg["beta_lower"],cfg["kl_coefficient"])})
            del vectors, encoded, responses, cached
            print(json.dumps({"stage":"gradient_group", "group_index":group_index,"seconds":time.monotonic()-started}),flush=True)
        actor.gradient_checkpointing_disable()
    return results


def main():
    started = time.monotonic()
    folder = Path(os.environ["P8_FOLDER"])
    plan = json.loads((folder / "diagnostic_manifest.json").read_text())
    limit = float(os.environ["P8_WORKER_SECONDS"])
    def check_deadline():
        if time.monotonic()-started >= limit-30: raise TimeoutError("Whole diagnostic worker budget")
    def forbid_optimizer(*args, **kwargs): raise RuntimeError("Optimizers are forbidden in P8")
    torch.optim.Optimizer.__init__ = forbid_optimizer
    assert torch.cuda.device_count() == 1 and os.environ["CUDA_VISIBLE_DEVICES"] == plan["gpu_uuid"]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    model_path = plan["model_path"]
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False, use_fast=False)
    def load_base():
        value = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16,
                  attn_implementation="sdpa", local_files_only=True, trust_remote_code=False).to("cuda:0").eval().requires_grad_(False)
        value.config.use_cache = False
        return value
    actor = load_base()
    base_distributions, readouts, gradients = [], [], []
    destination = folder / "diagnostics"
    destination.mkdir(exist_ok=False)
    for specification in plan["actors"]:
        check_deadline()
        name = specification["actor"]
        if name != "base": restore(actor,specification,plan["model_revision"])
        values = []
        with diagnostic_state(actor), inference_without_state_change(actor):
            for index, item in enumerate(plan["probes"]):
                check_deadline()
                encoded,candidates = encode(processor,item["input"],True)
                scores = option_log_probabilities(actor,encoded,candidates)
                distribution = first_distribution(actor,encoded)
                if name == "base": base_distributions.append(distribution)
                reference = base_distributions[index]
                ordered = sorted(scores,key=lambda k:(-scores[k],k))
                values.append({"id":item["input"]["id"],"group":item["alias"],"scores":scores,"answer":ordered[0],
                               "kl_actor_base":float(exact_kl(distribution,reference)),
                               "base_top_token_retained":int(distribution.argmax()) == int(reference.argmax()),
                               "top1_top2_gap":scores[ordered[0]]-scores[ordered[1]],"option_token_ids":candidates})
                del encoded,distribution
        record={"actor":name,"predictions":values,"labels_read":False,"integrity_checked":True}
        atomic_json(destination/(name+"_readout.json"),record)
        readouts.append(name)
        print(json.dumps({"stage":"readout_actor","actor":name,"completed":len(readouts),"seconds":time.monotonic()-started}),flush=True)
    atomic_json(destination/"readout_seal.json",{"actors":readouts,"probes_per_actor":16,"labels_read":False})
    # Admit D uniformly, after all C outputs, solely from time remaining.
    remaining = limit-(time.monotonic()-started)
    groups = 2 if remaining >= plan["d_two_groups_seconds"]*1.2 else (1 if remaining >= plan["d_one_group_seconds"]*1.2 else 0)
    atomic_json(destination/"d_admission.json",{"groups":groups,"remaining_seconds":remaining,"selection_uses_results":False})
    if groups:
        del actor
        gc.collect(); torch.cuda.empty_cache()
        actor = load_base()
        pool=plan["pool"][:groups]
        references=[]
        with diagnostic_state(actor), inference_without_state_change(actor):
            for group in pool:
                encoded,_=encode(processor,group["input"],False)
                refs=[]
                for ids in group["response_token_ids"]:
                    check_deadline()
                    refs.append(forward_response(actor,encoded,torch.tensor(ids,device="cuda:0")).detach().cpu())
                references.append(refs)
        for specification in plan["actors"]:
            check_deadline()
            name=specification["actor"]
            if name != "base": restore(actor,specification,plan["model_revision"])
            values=local_gradients(actor,processor,pool,references,plan["parameter_subset"],specification["diagnostic_configuration"],check_deadline)
            if name == "base":
                assert all(r["kl_raw_norm"] <= plan["base_kl_gradient_norm_tolerance"] for r in values)
            atomic_json(destination/(name+"_gradients.json"),{"actor":name,"rows":values,"labels_read":False})
            gradients.append(name)
            print(json.dumps({"stage":"gradient_actor","actor":name,"completed":len(gradients),"seconds":time.monotonic()-started}),flush=True)
    atomic_json(destination/"completion.json",{"readout_actors":readouts,"gradient_actors":gradients,
                "gradient_groups":groups,"all_c_completed":len(readouts)==7,"all_d_completed":len(gradients)==7 if groups else False,
                "optimizer_created":False,"optimizer_steps":0,"labels_read":False,
                "parameter_versions_checked_all":True,"parameter_values_sampled_three_per_tensor":True,
                "full_buffers_checked":True,"modes_rng_restored":True,"full_parameter_bytes_compared":False,
                "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                "elapsed_seconds":time.monotonic()-started})
    print(json.dumps({"stage":"diagnostics_complete","seconds":time.monotonic()-started}),flush=True)


if __name__ == "__main__":
    main()
