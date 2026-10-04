"""One independent full-parameter update; no labels or formal benchmark scoring.

The driver accepts SPINE, TTRL or No adaptation for later controlled use. The
checked-in pilot selects SPINE and exactly one update on a predetermined input.
"""
import json
import os
import random
import subprocess
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, GenerationConfig, Qwen2_5_VLForConditionalGeneration

from core import consensus_rewards, objective, prompt_messages, select_and_band, token_statistics

ROOT = Path(os.environ["P0_ROOT"])
CFG = json.loads((ROOT / os.environ.get("P1_CONFIG", "configs/pilot.json")).read_text())
OUT = ROOT / "outputs" / os.environ["P1_RUN"]
OUT.mkdir(parents=True, exist_ok=False)
START = time.time()
STOP_SAMPLER = threading.Event()
DEVICE_MEMORY = {"peak_used_bytes": 0, "samples": 0}


def sample_memory():
    # Bounded in-process profiling for this one job, not a persistent monitor.
    while not STOP_SAMPLER.is_set():
        free, total = torch.cuda.mem_get_info(0)
        DEVICE_MEMORY["peak_used_bytes"] = max(DEVICE_MEMORY["peak_used_bytes"], total - free)
        DEVICE_MEMORY["samples"] += 1
        STOP_SAMPLER.wait(0.25)


def event(stage, **values):
    record = dict(stage=stage, elapsed_seconds=time.time() - START, **values)
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
        record.update(allocated_bytes=torch.cuda.memory_allocated(), reserved_bytes=torch.cuda.memory_reserved(), peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    with (OUT / "events.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    return record


def forward_response(model, encoded, response):
    prompt_length = encoded["input_ids"].shape[-1]
    inputs = dict(encoded)
    inputs["input_ids"] = torch.cat([encoded["input_ids"], response[None]], dim=-1)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    output = model(**inputs, use_cache=False, return_dict=True)
    # Position prompt_length-1 predicts response[0]; final input has no target.
    return output.logits[0, prompt_length - 1:-1, :]


def generation_config(model, sample):
    return GenerationConfig(
        bos_token_id=model.config.bos_token_id,
        pad_token_id=model.generation_config.pad_token_id,
        eos_token_id=model.generation_config.eos_token_id,
        do_sample=sample, temperature=CFG["temperature"] if sample else 1.0,
        top_p=CFG["top_p"] if sample else 1.0, top_k=CFG["top_k"] if sample else 50,
        repetition_penalty=CFG["repetition_penalty"],
        max_new_tokens=CFG["max_new_tokens"], use_cache=True,
    )


@torch.no_grad()
def generate(model, processor, encoded, sample):
    model.eval()
    output = model.generate(**encoded, generation_config=generation_config(model, sample))
    response = output[0, encoded["input_ids"].shape[-1]:]
    text = processor.tokenizer.decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return response, text


def main():
    if CFG["optimizer_updates"] != 1 or CFG["rollouts"] != 8 or CFG["max_new_tokens"] != 2048:
        raise ValueError("This audited pilot requires one update, eight rollouts and a 2048-token cap")
    if CFG["freeze_vision"] or CFG["evaluation_labels_allowed"] or CFG["formal_evaluation"]:
        raise ValueError("Pilot scope mismatch")
    if CFG["method"] not in {"SPINE", "TTRL", "No adaptation"}:
        raise ValueError("Unknown method")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in {str(CFG["gpu_index"]), CFG.get("gpu_uuid")} or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one configured GPU must be visible")
    random.seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    torch.manual_seed(CFG["seed"])
    torch.cuda.manual_seed_all(CFG["seed"])
    torch.set_num_threads(16)
    torch.cuda.reset_peak_memory_stats()
    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    (OUT / "configuration.json").write_text(json.dumps(CFG, indent=2))
    (OUT / "started_at.txt").write_text(datetime.now().isoformat())
    rows = [json.loads(line) for line in (ROOT / CFG["sample_view"]).read_text().splitlines()]
    row = next(r for r in rows if r["id"] == CFG["sample_id"])
    messages = prompt_messages(row)  # Reject, rather than discard, unexpected label fields.
    model_path = ROOT / CFG["model_path"]
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False, use_fast=False)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    assert text.count("<|image_pad|>") == len(row["images"])
    images = []
    for path in row["image_paths"]:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    encoded = processor(text=[text], images=images, return_tensors="pt").to("cuda:0")
    assert len(encoded["image_grid_thw"]) == len(row["images"])
    event("inputs_ready", images=len(images), input_tokens=encoded["input_ids"].shape[-1], device_name=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()))
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True, trust_remote_code=False).to("cuda:0")
    assert all(p.requires_grad for p in model.parameters()), "No parameter freezing is permitted"
    assert not getattr(model, "is_quantized", False)
    event("model_loaded", parameters=sum(p.numel() for p in model.parameters()))
    before_ids, before_text = generate(model, processor, encoded, sample=False)
    (OUT / "greedy_before.json").write_text(json.dumps({"id": row["id"], "text": before_text, "tokens": len(before_ids)}, ensure_ascii=False))
    event("greedy_before", tokens=len(before_ids))
    del before_ids
    if CFG["method"] == "No adaptation":
        (OUT / "result.json").write_text(json.dumps({"status": "completed_no_adaptation_diagnostic", "formal_evaluation": False}))
        return

    responses, completions = [], []
    for i in range(CFG["rollouts"]):
        ids, completion = generate(model, processor, encoded, sample=True)
        responses.append(ids)
        completions.append(completion)
        event("rollout", index=i, tokens=len(ids), hit_length_cap=len(ids) == CFG["max_new_tokens"])
    rewards, advantages, vote = consensus_rewards(completions, row["options"])
    (OUT / "rollouts.json").write_text(json.dumps({"id": row["id"], "completions": completions, "vote": vote, "rewards": rewards.tolist(), "advantages": advantages.tolist()}, ensure_ascii=False, indent=2))
    event("consensus", valid_answers=vote["valid"], reward_counts={str(value): rewards.tolist().count(value) for value in set(rewards.tolist())}, nonzero_advantages=int((advantages != 0).sum()))

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    checkpoint_modules = [name for name, module in model.named_modules() if getattr(module, "gradient_checkpointing", False)]
    if not any(name.startswith("visual") for name in checkpoint_modules):
        raise RuntimeError("Vision gradient checkpointing was not enabled")
    event("gradient_checkpointing", modules=checkpoint_modules)

    cached = []
    for i, response in enumerate(responses):
        with torch.no_grad():
            logits = forward_response(model, encoded, response)
            logp, entropy, _ = token_statistics(logits, response, chunk_tokens=CFG["statistics_chunk_tokens"])
            selection, lower, upper, threshold = select_and_band(entropy, CFG["histogram_bins"])
            if CFG["method"] == "TTRL":
                selection = torch.ones_like(selection)
            cached.append(dict(old_logp=logp.detach(), selection=selection, lower=lower, upper=upper, reference_logits=logits.detach().cpu()))
            event("statistics", index=i, selected=int(selection.sum()), tokens=len(response), threshold=float(threshold), lower=float(lower), upper=float(upper), entropy_mean=float(entropy.mean()))
            del logits
    # Exactly one update starts from the pinned original base. Before this step
    # actor == behavior == frozen reference, so the detached full-vocabulary
    # logits cached above ARE reference-policy logits. They are never recomputed
    # from an updated actor. A multi-update trainer must load a separate reference.
    total_tokens = sum(len(response) for response in responses)
    total_selected = sum(int(item["selection"].sum()) for item in cached)
    assert total_selected > 0
    tracked = {}
    for kind, pattern in [("vision", "visual.patch_embed.proj.weight"), ("language", "model.layers.0.self_attn.q_proj.weight")]:
        parameter = dict(model.named_parameters())[pattern]
        tracked[kind] = (pattern, parameter.detach().cpu().clone())
    model.zero_grad(set_to_none=True)
    accumulated = {"policy": 0.0, "band": 0.0, "kl": 0.0}
    max_recompute_logp_error = 0.0
    for i, (response, item) in enumerate(zip(responses, cached)):
        logits = forward_response(model, encoded, response)
        logp, entropy, kl = token_statistics(logits, response, item["reference_logits"], CFG["statistics_chunk_tokens"])
        max_recompute_logp_error = max(max_recompute_logp_error, float((logp.detach() - item["old_logp"]).abs().max()))
        selection, _, _, _ = select_and_band(entropy, CFG["histogram_bins"])
        if CFG["method"] == "SPINE" and not torch.equal(selection, item["selection"]):
            raise RuntimeError("Gradient-pass token selection differs from statistics pass")
        loss, metrics = objective(logp, item["old_logp"], entropy, kl, advantages[i].to(logp.device), item["selection"], item["lower"], item["upper"], total_tokens, total_selected, CFG)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite loss")
        loss.backward()
        for key, value in metrics.items():
            accumulated[key] += value
        event("backward", index=i, **metrics)
        del logits, logp, entropy, kl, loss
    event("all_backward_complete", **accumulated)

    # Native torch AdamW on FP32 CPU masters; the BF16 model has full gradients.
    # This avoids changing to BF16 Adam state or introducing quantized optimizers.
    masters, params = [], []
    grad_squares = {"vision": 0.0, "language": 0.0}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"Missing gradient for trainable parameter: {name}")
        master = torch.nn.Parameter(parameter.detach().to(device="cpu", dtype=torch.float32))
        master.grad = parameter.grad.detach().to(device="cpu", dtype=torch.float32)
        if not torch.isfinite(master.grad).all():
            raise RuntimeError(f"Nonfinite gradient: {name}")
        role = "vision" if name.startswith("visual.") else "language"
        grad_squares[role] += float(master.grad.square().sum())
        parameter.grad = None
        masters.append(master)
        params.append(parameter)
    grad_norms = {key: value ** 0.5 for key, value in grad_squares.items()}
    if not all(value > 0 for value in grad_norms.values()):
        raise RuntimeError("Both language and vision must receive nonzero gradients")
    norm = torch.nn.utils.clip_grad_norm_(masters, CFG["max_grad_norm"], error_if_nonfinite=True)
    optimizer = torch.optim.AdamW(masters, lr=CFG["learning_rate"], betas=tuple(CFG["adam_betas"]), eps=CFG["adam_epsilon"], weight_decay=CFG["weight_decay"], foreach=False)
    optimizer.step()
    with torch.no_grad():
        for parameter, master in zip(params, masters):
            parameter.copy_(master.to(device=parameter.device, dtype=parameter.dtype))
    updates = {}
    named = dict(model.named_parameters())
    for role, (name, before) in tracked.items():
        after = named[name].detach().cpu()
        updates[role] = {"parameter": name, "changed_elements": int((after != before).sum()), "delta_l2": float((after.float() - before.float()).norm())}
        if updates[role]["changed_elements"] == 0:
            raise RuntimeError(f"No representable BF16 update in tracked {role} parameter")
    event("optimizer_step_complete", update_count=1, pre_clip_global_grad_norm=float(norm), grad_norms=grad_norms, tracked_updates=updates)
    del optimizer, masters, cached
    model.eval()
    after_ids, after_text = generate(model, processor, encoded, sample=False)
    (OUT / "greedy_after.json").write_text(json.dumps({"id": row["id"], "text": after_text, "tokens": len(after_ids)}, ensure_ascii=False))
    event("greedy_after", tokens=len(after_ids))
    model.config.use_cache = True
    model.save_pretrained(OUT / "checkpoint", safe_serialization=True, max_shard_size="4GB")
    processor.save_pretrained(OUT / "checkpoint")
    final = event("completed", update_count=1, **accumulated)
    result = {"status": "passed_single_update_diagnostic", "method": CFG["method"], "official_reproduction": False, "formal_evaluation": False, "dataset": row["dataset"], "sample_id": row["id"], "image_count": len(images), "input_tokens": encoded["input_ids"].shape[-1], "rollout_count": len(responses), "response_lengths": [len(response) for response in responses], "valid_answers": vote["valid"], "reward_values": rewards.tolist(), "nonzero_advantages": int((advantages != 0).sum()), "selected_tokens": total_selected, "response_tokens": total_tokens, "loss_components": accumulated, "gradient_norms": grad_norms, "tracked_updates": updates, "optimizer_steps": 1, "peak_allocated_bytes": final["peak_allocated_bytes"], "peak_reserved_bytes": final["peak_reserved_bytes"], "elapsed_seconds": time.time() - START, "max_recompute_logp_error": max_recompute_logp_error, "evaluation_labels_read": False, "reference_policy": "immutable full-vocabulary base-model logits cached before the only optimizer step", "checkpoint_contains_optimizer": False}
    STOP_SAMPLER.set()
    sampler.join(timeout=2)
    result["sampled_device_memory"] = dict(DEVICE_MEMORY)
    result["device_memory_note"] = "250ms whole-device samples, including driver/other-process memory; transient peaks can be missed"
    result["cpu_peak_rss_kib"] = next(int(line.split()[1]) for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("VmHWM:"))
    (OUT / "result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        (OUT / "failure.json").write_text(json.dumps({"type": type(exc).__name__, "message": str(exc), "elapsed_seconds": time.time() - START}, indent=2))
        traceback.print_exc()
        raise
    finally:
        STOP_SAMPLER.set()
