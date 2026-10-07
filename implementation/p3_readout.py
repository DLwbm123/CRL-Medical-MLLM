"""Label-free, evaluation-only option likelihoods on read-only P2 actors."""
import gc
import inspect
import json
import os
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from continual import forward_response, load_input
from state import atomic_json, inference_without_state_change

INSTRUCTION = "Select exactly one of the listed options. Return only its option identifier. Do not provide reasoning."
ASSISTANT_PREFIX = "Final answer:"


def messages(row):
    content = [{"type": "image", "image": path} for path in row["image_paths"]]
    options = "\n".join(f"{key}. {value}" for key, value in row["options"].items())
    content.append({"type": "text", "text": row["question"] + "\n" + options + "\n" + INSTRUCTION})
    return [{"role": "user", "content": content}]


def option_tokens(tokenizer, prefix, options):
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    result = {}
    for identifier in sorted(options):
        suffix = " " + identifier
        tokens = tokenizer.encode(suffix, add_special_tokens=False)
        if not tokens or tokenizer.encode(prefix + suffix, add_special_tokens=False) != prefix_ids + tokens:
            raise ValueError("Option tokenization changes the common prefix")
        result[identifier] = tokens
    return result


@torch.no_grad()
def option_log_probabilities(actor, encoded, candidates):
    """Sum all identifier-token conditional log-probabilities; no EOS or length normalization."""
    result = {}
    single = {key: tokens[0] for key, tokens in candidates.items() if len(tokens) == 1}
    if single:
        kwargs = dict(encoded, use_cache=False, return_dict=True)
        if "logits_to_keep" in inspect.signature(actor.forward).parameters:
            kwargs["logits_to_keep"] = 1
        output = actor(**kwargs)
        distribution = output.logits[0, -1].float().log_softmax(-1)
        for key, token in single.items():
            result[key] = float(distribution[token])
        del output, distribution
    for key, tokens in candidates.items():
        if len(tokens) == 1:
            continue
        response = torch.tensor(tokens, device=encoded["input_ids"].device)
        logits = forward_response(actor, encoded, response)
        selected = logits.float().log_softmax(-1).gather(-1, response[:, None]).squeeze(-1)
        result[key] = float(selected.sum())
        del logits, selected
    if any(not torch.isfinite(torch.tensor(value)) for value in result.values()):
        raise ValueError("Nonfinite legal-option likelihood")
    return result


@torch.no_grad()
def restore_read_only_actor(actor, path, revision, expected_cursor):
    saved = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if saved["model_revision"] != revision or saved["cursor"] != expected_cursor or not saved["reference_unchanged"]:
        raise ValueError("Unexpected P2 checkpoint provenance")
    parameters = dict(actor.named_parameters())
    if list(parameters) != list(saved["master_weights"]):
        raise ValueError("Checkpoint master names/order differ")
    for name, parameter in parameters.items():
        master = saved["master_weights"][name]
        if master.shape != parameter.shape or master.dtype != torch.float32:
            raise ValueError("Invalid checkpoint master schema")
        parameter.copy_(master.to(device=parameter.device, dtype=parameter.dtype))
    buffers = dict(actor.named_buffers())
    if set(buffers) != set(saved["actor_buffers"]):
        raise ValueError("Checkpoint buffer schema differs")
    for name, buffer in buffers.items():
        value = saved["actor_buffers"][name]
        if value.shape != buffer.shape:
            raise ValueError("Checkpoint buffer shape differs")
        buffer.copy_(value.to(buffer.device))
    metadata = {"cursor": saved["cursor"], "optimizer_updates": saved["optimizer_updates"],
                "training_code_commit": saved["code_commit"], "manifest_sha256": saved["manifest_sha256"],
                "model_revision": saved["model_revision"], "master_tensors": len(parameters),
                "checkpoint_bytes": path.stat().st_size, "optimizer_created": False,
                "checkpoint_opened_read_only": True}
    del saved
    gc.collect()
    return metadata


def checkpoint_path(folder, key, cursor):
    matches = list((folder / "runs" / ("main-" + key) / "checkpoints").glob(f"c{cursor:06d}-u*/state.pt"))
    if len(matches) != 1:
        raise ValueError("Expected one retained P2 checkpoint at the declared cursor")
    return matches[0]


def main():
    start = time.monotonic()
    deadline = start + min(float(os.environ.get("P3_READOUT_SECONDS", "2700")),
                           float(os.environ.get("P2_MAX_JOB_SECONDS", "2700")))
    root = Path(os.environ["P0_ROOT"])
    source = Path(os.environ["P3_P2_FOLDER"])
    destination = root / "outputs" / os.environ["P2_CAMPAIGN"] / "readout"
    destination.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((source / "runs/main-a/configuration.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    if len(manifest["probe"]) != 16 or torch.cuda.device_count() != 1 or os.environ["CUDA_VISIBLE_DEVICES"] != os.environ["P2_GPU_UUID"]:
        raise ValueError("P2 probe or authorized visible GPU differs")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    model_path = root / cfg["model_path"]
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False, use_fast=False)
    actor = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,
        trust_remote_code=False).to("cuda:0").eval().requires_grad_(False)
    actor.config.use_cache = False
    plan = [("frozen_base", None, 0), ("p2_spine_final", "d", 16),
            ("p2_ttrl_final", "c", 16), ("p2_spine_midpoint", "d", 8)]
    completed = []
    try:
        for name, key, cursor in plan:
            if time.monotonic() >= deadline:
                break
            actor_start = time.monotonic()
            provenance = {"model_revision": cfg["model_revision"], "optimizer_created": False,
                          "actor_source": "pinned original base"}
            if key:
                provenance = restore_read_only_actor(actor, checkpoint_path(source, key, cursor), cfg["model_revision"], cursor)
            torch.cuda.reset_peak_memory_stats()
            versions = {key: value._version for key, value in actor.named_parameters()}
            outputs = []
            with inference_without_state_change(actor):
                for entry in manifest["probe"]:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Readout diagnosis reached its hard budget")
                    row = load_input(source, entry)
                    prompt = processor.apply_chat_template(messages(row), tokenize=False, add_generation_prompt=True) + ASSISTANT_PREFIX
                    candidates = option_tokens(processor.tokenizer, prompt, row["options"])
                    images = []
                    for image_path in row["image_paths"]:
                        with Image.open(image_path) as image:
                            images.append(image.convert("RGB"))
                    encoded = processor(text=[prompt], images=images, return_tensors="pt").to("cuda:0")
                    if prompt.count("<|image_pad|>") != len(images) or len(encoded["image_grid_thw"]) != len(images):
                        raise ValueError("Readout image ordering/count differs")
                    scores = option_log_probabilities(actor, encoded, candidates)
                    answer = min(scores, key=lambda key: (-scores[key], key))
                    outputs.append({"id": row["id"], "answer": answer, "scores": scores, "option_token_ids": candidates})
                    atomic_json(destination / "active_readout.json", {"actor": name, "predictions": outputs})
                    print(json.dumps({"stage": "readout_case", "actor": name, "completed": len(outputs),
                                      "elapsed_seconds": time.monotonic() - start}), flush=True)
                    del encoded, images
            if versions != {key: value._version for key, value in actor.named_parameters()}:
                raise RuntimeError("Readout altered actor parameters")
            result = {"actor": name, "predictions": outputs, "provenance": provenance,
                      "protocol": {"instruction": INSTRUCTION, "assistant_prefix": ASSISTANT_PREFIX,
                                   "option_suffix": "space followed by identifier", "score": "sum conditional log-probabilities of every identifier token",
                                   "eos_scored": False, "length_normalized": False, "reasoning_context": False,
                                   "tie_rule": "lexicographically first maximum", "label_fields_read": False},
                      "elapsed_seconds": time.monotonic() - actor_start,
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                      "parameters_unchanged_by_readout": True, "rng_and_modes_preserved": True}
            atomic_json(destination / (name + ".json"), result)
            completed.append(name)
    finally:
        atomic_json(destination / "completion.json", {"completed_actors": completed, "planned_actors": [x[0] for x in plan],
                    "elapsed_seconds": time.monotonic() - start, "max_seconds": deadline - start,
                    "cpu_peak_rss_kib": next(int(line.split()[1]) for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("VmHWM:")),
                    "labels_read": False, "optimizer_created": False})
    print(json.dumps({"stage": "readout_complete", "actors": completed, "elapsed_seconds": time.monotonic() - start}), flush=True)


if __name__ == "__main__":
    main()
