"""Single-pass, label-free continual adaptation with independent fixed reference."""
import contextlib
import hashlib
import json
import math
import os
import random
import signal
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, GenerationConfig, Qwen2_5_VLForConditionalGeneration, StoppingCriteria

from core import INPUT_KEYS, consensus_rewards, extract_answer, objective, prompt_messages, select_and_band, token_statistics
from state import PersistentAdam, atomic_json, inference_without_state_change, load_checkpoint, restore_rng, rng_state, save_checkpoint


def group_rewards(completions, options, reward_target=None):
    """Preserve consensus by default; optionally reward a fixed legal pseudo-target."""
    if reward_target is not None and reward_target not in options:
        raise ValueError("Reference reward target is not a legal option")
    answers = [extract_answer(text, options) for text in completions]
    if all(answer is None for answer in answers):
        return None, None, {"answers": answers, "winner": None, "valid": 0, "counts": {},
                            "tie": False, "top_count": 0, "margin": 0,
                            "all_unparseable": True, "zero_advantage_group": False}
    rewards, advantages, detail = consensus_rewards(completions, options)
    if reward_target is not None:
        rewards = torch.tensor([float(answer == reward_target) if answer is not None else 0.0 for answer in answers])
        advantages = (rewards - rewards.mean()) / (rewards.std(correction=0) + 1e-6)
        detail.update(reward_source="frozen_legal", reward_target=reward_target)
    counts = sorted(detail["counts"].values(), reverse=True)
    detail.update(tie=len(counts) > 1 and counts[0] == counts[1], top_count=counts[0],
                  margin=counts[0] - (counts[1] if len(counts) > 1 else 0), all_unparseable=False,
                  zero_advantage_group=bool(torch.count_nonzero(advantages) == 0))
    return rewards, advantages, detail


def load_input(folder, entry):
    path = (folder / entry["input"]).resolve()
    if not path.is_relative_to((folder / "inputs").resolve()):
        raise ValueError("Input path escapes private adaptation inputs")
    row = json.loads(path.read_text())
    if set(row) != INPUT_KEYS or row["id"] != entry["id"]:
        raise ValueError("Trainer rejects labels, extra fields and mismatched sample IDs")
    return row


def validate_development_configuration(cfg, acceptance, manifest):
    """P3 varies only its declared rollout seed; P2 keeps its exact check."""
    current = {name: value for name, value in cfg.items() if name != "method"}
    accepted = acceptance["validated_configuration_without_method"]
    if "rollout_seeds" not in manifest:
        if current != accepted:
            raise RuntimeError("Configuration differs from engineering acceptance")
        return
    if manifest.get("p4_version") == 1:
        if cfg["method"] != "Frozen greedy" and cfg.get("reward_source") not in manifest["reward_sources"]:
            raise RuntimeError("Reward source is outside the locked P4 manifest")
        current = {k: v for k, v in current.items() if k != "reward_source"}
    allowed = manifest["rollout_seeds"]
    cached_base = cfg["method"] == "Frozen greedy" and cfg["seed"] == accepted["seed"]
    if not cached_base and cfg["seed"] not in allowed:
        raise RuntimeError("Rollout seed is outside the locked manifest")
    if {k: v for k, v in current.items() if k != "seed"} != {k: v for k, v in accepted.items() if k != "seed"}:
        raise RuntimeError("P3 changes more than the declared rollout seed")


def forward_response(model, encoded, response):
    length = encoded["input_ids"].shape[-1]
    inputs = dict(encoded)
    inputs["input_ids"] = torch.cat([encoded["input_ids"], response[None]], dim=-1)
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    return model(**inputs, use_cache=False, return_dict=True).logits[0, length - 1:-1]


class Deadline(StoppingCriteria):
    def __init__(self, deadline):
        self.deadline = deadline
        self.requested = False

    def check(self):
        if self.requested or time.monotonic() >= self.deadline:
            raise TimeoutError("Owned job reached its stop request or deadline")

    def __call__(self, input_ids, scores, **kwargs):
        return self.requested or time.monotonic() >= self.deadline


class Engine:
    def __init__(self, root, out, cfg, deadline):
        self.root, self.out, self.cfg, self.deadline = root, out, cfg, deadline
        self.reward_targets = None
        if cfg.get("reward_source") == "frozen_legal":
            folder = out.parent.parent
            manifest_bytes = (folder / os.environ.get("P2_MANIFEST", "manifest.json")).read_bytes()
            manifest = json.loads(manifest_bytes)
            signal = json.loads((folder / "reference/stream_targets.json").read_text())
            if signal["labels_read"] or signal["model_revision"] != cfg["model_revision"] or signal["manifest_sha256"] != hashlib.sha256(manifest_bytes).hexdigest():
                raise ValueError("Frozen reward signal provenance differs")
            if [x["id"] for x in signal["predictions"]] != [x["id"] for x in manifest["stream"]]:
                raise ValueError("Reward signal must cover only the locked stream in order")
            self.reward_targets = {}
            for prediction, entry in zip(signal["predictions"], manifest["stream"]):
                options = load_input(folder, entry)["options"]
                if prediction["winner"] not in options or set(prediction["scores"]) != set(options):
                    raise ValueError("Reference signal contains an illegal option")
                if not all(math.isfinite(v) for v in prediction["scores"].values()) or prediction["winner"] != min(prediction["scores"], key=lambda k: (-prediction["scores"][k], k)):
                    raise ValueError("Reference signal differs from finite legal-option maximum")
                self.reward_targets[prediction["id"]] = prediction["winner"]
            snapshot = out / "reference_signal_snapshot.json"
            if snapshot.exists():
                if json.loads(snapshot.read_text()) != signal:
                    raise ValueError("Frozen reward signal changed during resume")
            else:
                atomic_json(snapshot, signal)
        self.method = cfg["method"]
        self.training = self.method in {"TTRL", "SPINE"}
        self.timings = {}
        self.started = time.monotonic()
        self.stop_sampler = threading.Event()
        self.memory = {"peak_used_bytes": 0, "samples": 0}
        self.sampler = threading.Thread(target=self.sample_memory, daemon=True)
        self.sampler.start()
        with self.phase("load"):
            path = root / cfg["model_path"]
            self.processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_fast=False)
            self.actor = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                local_files_only=True, trust_remote_code=False,
            ).to("cuda:0")
            if getattr(self.actor, "is_quantized", False):
                raise ValueError("Quantization is outside this protocol")
            if self.training:
                self.reference = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                    local_files_only=True, trust_remote_code=False,
                ).to("cuda:0").eval().requires_grad_(False)
                self.reference_initial = {name: p.detach().cpu().clone() for name, p in self.reference.named_parameters()}
                if not all(torch.equal(p, dict(self.reference.named_parameters())[name]) for name, p in self.actor.named_parameters()):
                    raise RuntimeError("Fresh actor and fixed reference initialization differ")
                self.state = PersistentAdam(self.actor, cfg)
                reference_ids = {id(p) for p in self.reference.parameters()}
                optimizer_ids = {id(p) for group in self.state.optimizer.param_groups for p in group["params"]}
                if reference_ids & optimizer_ids or any(p.requires_grad for p in self.reference.parameters()):
                    raise RuntimeError("Reference entered optimization")
                self.actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                if not getattr(self.actor.visual, "gradient_checkpointing", False):
                    raise RuntimeError("Vision gradient checkpointing is not active")
                self.actor.config.use_cache = False
            else:
                self.actor.requires_grad_(False)
                self.reference = self.state = None
            if any(isinstance(module, torch.nn.Dropout) and module.p != 0 for module in self.actor.modules()):
                raise RuntimeError("Nonzero dropout requires a separate behavior-policy treatment")
            if getattr(self.actor.config, "attention_dropout", 0) != 0:
                raise RuntimeError("Attention dropout must be zero for the cached behavior pass")
        self.event("model_ready", trainable=self.training, parameters=sum(p.numel() for p in self.actor.parameters()))

    def sample_memory(self):
        while not self.stop_sampler.is_set():
            free, total = torch.cuda.mem_get_info()
            self.memory["peak_used_bytes"] = max(self.memory["peak_used_bytes"], total - free)
            self.memory["samples"] += 1
            self.stop_sampler.wait(0.25)

    @contextlib.contextmanager
    def phase(self, name):
        self.deadline.check()
        torch.cuda.synchronize()
        start = time.monotonic()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.timings[name] = self.timings.get(name, 0.0) + time.monotonic() - start

    def event(self, stage, **values):
        payload = {"stage": stage, "elapsed_seconds": time.monotonic() - self.started, **values}
        with (self.out / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(payload) + "\n")
        print(json.dumps(payload), flush=True)

    def encode(self, row):
        messages = prompt_messages(row)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if text.count("<|image_pad|>") != len(row["images"]):
            raise ValueError("Image placeholder count mismatch")
        images = []
        for path in row["image_paths"]:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        encoded = self.processor(text=[text], images=images, return_tensors="pt").to("cuda:0")
        if len(encoded["image_grid_thw"]) != len(images):
            raise ValueError("Image grid count mismatch")
        return encoded

    def generation_config(self, sample, probability_check=False):
        cfg = self.cfg
        return GenerationConfig(
            bos_token_id=self.actor.config.bos_token_id, pad_token_id=self.actor.generation_config.pad_token_id,
            eos_token_id=self.actor.generation_config.eos_token_id, do_sample=sample,
            temperature=1.0 if probability_check or not sample else cfg["temperature"],
            top_p=1.0 if probability_check or not sample else cfg["top_p"],
            top_k=0 if probability_check or not sample else cfg["top_k"],
            repetition_penalty=cfg["repetition_penalty"],
            max_new_tokens=4 if probability_check else cfg["max_new_tokens"],
            use_cache=True,
        )

    @torch.no_grad()
    def generate(self, encoded, sample):
        self.actor.eval()
        self.deadline.check()
        tokens = self.actor.generate(**encoded, generation_config=self.generation_config(sample), stopping_criteria=[self.deadline])[0]
        self.deadline.check()
        response = tokens[encoded["input_ids"].shape[-1]:]
        text = self.processor.tokenizer.decode(response, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return response, text

    def probability_check(self, row):
        with inference_without_state_change(self.actor):
            encoded = self.encode(row)
            output = self.actor.generate(**encoded, generation_config=self.generation_config(True, True),
                                         return_dict_in_generate=True, output_scores=True, stopping_criteria=[self.deadline])
            response = output.sequences[0, encoded["input_ids"].shape[-1]:]
            generated_logp = torch.stack([score[0].float().log_softmax(-1)[token] for score, token in zip(output.scores, response)])
            logits = forward_response(self.actor, encoded, response)
            recomputed_logp, _, _ = token_statistics(logits, response, chunk_tokens=self.cfg["statistics_chunk_tokens"])
            error = float((generated_logp - recomputed_logp).abs().max())
            # BF16 cached decoding and full-prefix kernels need not be bitwise equal.
            if error > self.cfg["probability_check_bf16_atol"]:
                raise RuntimeError(f"Unwarped generation/recompute log-prob mismatch: {error}")
        return {"tokens": len(response), "temperature": 1, "top_p": 1, "top_k": 0,
                "max_logp_difference": error, "atol": self.cfg["probability_check_bf16_atol"],
                "production_sampling": "temperature 0.7/top-p 0.95; loss uses raw untempered model softmax",
                "strict_on_policy_consistency_claimed": False}

    def probe(self, folder, entries, cursor):
        destination = self.out / "probe" / f"c{cursor:06d}.json"
        if destination.exists():
            return
        outputs = []
        with self.phase("probe"), inference_without_state_change(self.actor):
            for entry in entries:
                row = load_input(folder, entry)
                encoded = self.encode(row)
                response, text = self.generate(encoded, False)
                outputs.append({"id": row["id"], "text": text, "answer": extract_answer(text, row["options"]), "tokens": len(response)})
        atomic_json(destination, {"cursor": cursor, "decoding": "greedy", "predictions": outputs,
                                  "rng_and_modes_restored": True, "optimizer_updates": self.state.updates if self.state else 0})
        self.event("probe_complete", cursor=cursor, samples=len(outputs))

    def update(self, encoded, responses, advantages, drift):
        cfg, cached = self.cfg, []
        self.actor.train()
        with self.phase("behavior_forward"), torch.no_grad():
            for response in responses:
                logits = forward_response(self.actor, encoded, response)
                logp, entropy, _ = token_statistics(logits, response, chunk_tokens=cfg["statistics_chunk_tokens"])
                mask, lower, upper, threshold = select_and_band(entropy, cfg["histogram_bins"])
                if self.method == "TTRL":
                    mask = torch.ones_like(mask)
                cached.append({"old_logp": logp.detach(), "selection": mask, "lower": lower, "upper": upper,
                               "threshold": float(threshold), "mean_entropy": float(entropy.mean())})
                del logits
        total_tokens = sum(len(response) for response in responses)
        selected_tokens = sum(int(item["selection"].sum()) for item in cached)
        self.state.zero_grad()
        loss_sum = {"policy": 0.0, "band": 0.0, "kl": 0.0}
        kl_sum = sampled_ratio_sum = clip_count = 0.0
        recompute_error = 0.0
        tracked_names = ["visual.patch_embed.proj.weight", "model.layers.0.self_attn.q_proj.weight"]
        tracked_before = {name: dict(self.actor.named_parameters())[name].detach().cpu().clone() for name in tracked_names}
        for response_index, (response, item) in enumerate(zip(responses, cached)):
            self.deadline.check()
            with self.phase("reference_forward"), torch.no_grad():
                reference_logits = forward_response(self.reference, encoded, response).detach().cpu()
            with self.phase("backward"):
                logits = forward_response(self.actor, encoded, response)
                logp, entropy, kl = token_statistics(logits, response, reference_logits, cfg["statistics_chunk_tokens"])
                error = float((logp.detach() - item["old_logp"]).abs().max())
                recompute_error = max(recompute_error, error)
                if error > cfg["behavior_recompute_atol"]:
                    raise RuntimeError("Behavior policy changed before its optimizer update")
                mask, _, _, _ = select_and_band(entropy, cfg["histogram_bins"])
                if self.method == "SPINE" and not torch.equal(mask, item["selection"]):
                    raise RuntimeError("Token selection differs between statistics and gradient passes")
                ratio = (logp.detach() - item["old_logp"]).exp()
                sampled_ratio_sum += float(ratio.sum())
                clip_count += int(((ratio - 1).abs() > cfg["clip_epsilon"]).sum())
                kl_sum += float(kl.detach().sum())
                loss, components = objective(logp, item["old_logp"], entropy, kl, advantages[response_index].to(logp.device),
                                             item["selection"], item["lower"], item["upper"], total_tokens, selected_tokens, cfg)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite objective")
                loss.backward()
                for key, value in components.items():
                    loss_sum[key] += value
                del logits, reference_logits, logp, entropy, kl, loss
        with self.phase("cpu_optimizer"):
            optimization = self.state.step()
        tracked_updates = {}
        for name, before in tracked_before.items():
            after = dict(self.actor.named_parameters())[name].detach().cpu()
            tracked_updates[name] = {"changed_elements": int((after != before).sum()), "delta_l2": float((after.float() - before.float()).norm())}
        diagnostics = {
            **optimization, "loss_components": loss_sum, "response_tokens": total_tokens, "selected_tokens": selected_tokens,
            "selected_ratio": selected_tokens / total_tokens, "ratio_mean_before_step": sampled_ratio_sum / total_tokens,
            "clip_fraction_before_step": clip_count / total_tokens, "exact_full_vocabulary_kl_mean_before_step": kl_sum / total_tokens,
            "max_behavior_recompute_error": recompute_error, "tracked_actor_updates": tracked_updates,
            "selection": [{key: item[key] for key in ["threshold", "mean_entropy"]} for item in cached],
        }
        if drift:
            with self.phase("post_update_drift"), torch.no_grad():
                self.actor.eval()
                response = responses[0]
                reference_logits = forward_response(self.reference, encoded, response)
                logits = forward_response(self.actor, encoded, response)
                logp, _, kl = token_statistics(logits, response, reference_logits, cfg["statistics_chunk_tokens"])
                difference = logp - cached[0]["old_logp"]
                diagnostics["post_update_drift"] = {
                    "scope": "first rollout, same generated prefixes", "tokens": len(response),
                    "exact_full_vocabulary_kl_to_fixed_base": float(kl.mean()),
                    "sampled_token_log_ratio_to_behavior_mean": float(difference.mean()),
                    "sampled_token_log_ratio_abs_max": float(difference.abs().max()),
                    "post_update_raw_ratio_clip_fraction": float(((difference.exp() - 1).abs() > cfg["clip_epsilon"]).float().mean()),
                }
        return diagnostics

    def case(self, row, index, drift):
        start, phase_start = time.monotonic(), dict(self.timings)
        with self.phase("preprocess"):
            encoded = self.encode(row)
        record = {"index": index, "id": row["id"], "method": self.method, "images": len(row["images"]),
                  "input_tokens": encoded["input_ids"].shape[-1], "optimizer_updates_before": self.state.updates if self.state else 0}
        if self.method != "Frozen SC-8":
            with self.phase("greedy_before"):
                ids, text = self.generate(encoded, False)
            record["before"] = {"text": text, "answer": extract_answer(text, row["options"]), "tokens": len(ids)}
        atomic_json(self.out / "active_case.json", record)
        if self.method != "Frozen greedy":
            responses, completions, token_sequences = [], [], []
            with self.phase("rollout"):
                for response_index in range(self.cfg["rollouts"]):
                    response, text = self.generate(encoded, True)
                    responses.append(response)
                    completions.append(text)
                    token_sequences.append(response.tolist())
                    self.event("rollout_complete", index=index, response_index=response_index, tokens=len(response),
                               hit_length_cap=len(response) == self.cfg["max_new_tokens"])
            target = None if self.reward_targets is None else self.reward_targets[row["id"]]
            rewards, advantages, vote = group_rewards(completions, row["options"], target)
            if target is not None:
                vote.update(reward_source="frozen_legal", reward_target=target)
            record.update(vote=vote, completions=completions, response_token_ids=token_sequences,
                          response_lengths=[len(response) for response in responses],
                          hit_length_cap=[len(response) == self.cfg["max_new_tokens"] for response in responses],
                          rewards=rewards.tolist() if rewards is not None else None)
            atomic_json(self.out / "active_case.json", record)
            if self.training:
                if vote["all_unparseable"]:
                    record.update(update_applied=False, skip_reason="all_unparseable")
                else:
                    record["optimization"] = self.update(encoded, responses, advantages, drift)
                    record.update(update_applied=True, skip_reason=None,
                                  update_interpretation="band/KL or Adam momentum without group RL advantage" if vote["zero_advantage_group"] else "nonzero group RL advantages")
                with self.phase("greedy_after"):
                    ids, text = self.generate(encoded, False)
                record["after"] = {"text": text, "answer": extract_answer(text, row["options"]), "tokens": len(ids)}
        record.update(optimizer_updates_after=self.state.updates if self.state else 0,
                      elapsed_seconds=time.monotonic() - start,
                      phase_seconds={key: value - phase_start.get(key, 0.0) for key, value in self.timings.items()})
        return record

    def reference_unchanged(self):
        if not self.training:
            return True
        with self.phase("reference_integrity"):
            for name, parameter in self.reference.named_parameters():
                if parameter.requires_grad or parameter.grad is not None or not torch.equal(parameter.detach().cpu(), self.reference_initial[name]):
                    raise RuntimeError("Fixed reference parameters changed")
        return True

    def close(self):
        self.stop_sampler.set()
        self.sampler.join(timeout=2)


def recover_tail(out, cursor):
    """Preserve post-checkpoint predictions before deterministic replay after a crash."""
    abandoned = out / "interrupted" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for folder, prefix in [(out / "cases", "i"), (out / "probe", "c")]:
        for path in folder.glob(prefix + "*.json"):
            number = int(path.stem[1:])
            if (prefix == "i" and number >= cursor) or (prefix == "c" and number > cursor):
                destination = abandoned / folder.name
                destination.mkdir(parents=True, exist_ok=True)
                path.rename(destination / path.name)
    active = out / "active_case.json"
    if active.exists():
        abandoned.mkdir(parents=True, exist_ok=True)
        active.rename(abandoned / active.name)


def main():
    root = Path(os.environ["P0_ROOT"])
    folder = root / "outputs" / os.environ["P2_CAMPAIGN"]
    manifest_path = folder / os.environ.get("P2_MANIFEST", "manifest.json")
    raw = manifest_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != manifest_path.with_suffix(".sha256").read_text().strip():
        raise ValueError("Manifest checksum changed")
    manifest = json.loads(raw)
    cfg = json.loads((root / os.environ["P2_CONFIG"]).read_text())
    method = cfg["method"]
    if method not in {"Frozen greedy", "Frozen SC-8", "TTRL", "SPINE"} or cfg["rollouts"] != 8 or cfg["max_new_tokens"] != 2048 or cfg["max_updates_per_case"] != 1 or cfg["freeze_vision"] or cfg["evaluation_labels_allowed"]:
        raise ValueError("Configuration violates the locked protocol")
    if manifest["kind"] == "development":
        acceptance = json.loads((folder / "acceptance.json").read_text())
        if not acceptance["passed"]:
            raise RuntimeError("Engineering acceptance has not passed")
        validate_development_configuration(cfg, acceptance, manifest)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible not in {str(cfg["gpu_index"]), os.environ.get("P2_GPU_UUID")} or torch.cuda.device_count() != 1:
        raise RuntimeError("Only the configured authorized GPU may be visible")
    budget = json.loads((root / "metadata/campaign_budget.json").read_text())
    remaining = (datetime.fromisoformat(budget["gpu_stop_utc"]) - datetime.now(timezone.utc)).total_seconds()
    deadline = Deadline(time.monotonic() + min(remaining, float(os.environ.get("P2_MAX_JOB_SECONDS", "7200"))))
    for signum in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(signum, lambda *_: setattr(deadline, "requested", True))
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.set_num_threads(cfg["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.cuda.reset_peak_memory_stats()
    run_id = os.environ["P2_RUN"]
    out = folder / "runs" / run_id
    resume = os.environ.get("P2_RESUME") == "1"
    out.mkdir(parents=True, exist_ok=resume)
    (out / "cases").mkdir(exist_ok=True)
    (out / "probe").mkdir(exist_ok=True)
    code_commit = os.environ["P2_CODE_COMMIT"]
    if resume:
        if json.loads((out / "configuration.json").read_text()) != cfg:
            raise ValueError("Refusing to replace an existing run's configuration")
    else:
        atomic_json(out / "configuration.json", cfg)
    engine = None
    try:
        engine = Engine(root, out, cfg, deadline)
        cursor, processed = 0, []
        if resume:
            saved = load_checkpoint(out / "checkpoints", cfg, digest, method)
            if saved["code_commit"] != code_commit or saved["model_revision"] != cfg["model_revision"]:
                raise ValueError("Code/model revision changed during resume")
            cursor, processed = saved["cursor"], saved["processed_ids"]
            if processed != [entry["id"] for entry in manifest["stream"][:cursor]]:
                raise ValueError("Restored cursor does not match manifest prefix")
            if engine.state:
                engine.state.load(saved)
            restore_rng(saved["rng"])
            del saved
            recover_tail(out, cursor)
            engine.event("resumed", cursor=cursor, optimizer_updates=engine.state.updates if engine.state else 0)
        elif os.environ.get("P2_PROBABILITY_CHECK") == "1":
            result = engine.probability_check(load_input(folder, manifest["stream"][0]))
            atomic_json(out / "probability_check.json", result)
            engine.event("probability_check", **result)
        if cursor in manifest["probe_cursors"]:
            engine.probe(folder, manifest["probe"], cursor)
        stop = min(int(os.environ.get("P2_STOP_CURSOR", len(manifest["stream"]))), len(manifest["stream"]))
        if stop <= cursor:
            raise ValueError("Resume stop cursor must advance the stream")
        for index in range(cursor, stop):
            deadline.check()
            row = load_input(folder, manifest["stream"][index])
            record = engine.case(row, index, index + 1 in manifest["drift_cursors"])
            atomic_json(out / "cases" / f"i{index:06d}.json", record)
            processed.append(row["id"])
            cursor = index + 1
            engine.event("case_complete", cursor=cursor, method=method, seconds=record["elapsed_seconds"],
                         optimizer_updates=engine.state.updates if engine.state else 0,
                         valid_answers=record.get("vote", {}).get("valid"), skipped=record.get("skip_reason"))
            if cursor in manifest["probe_cursors"]:
                engine.probe(folder, manifest["probe"], cursor)
        reference_ok = engine.reference_unchanged()
        payload = {"cursor": cursor, "processed_ids": processed, "optimizer_updates": engine.state.updates if engine.state else 0,
                   "configuration": cfg, "code_commit": code_commit, "model_revision": cfg["model_revision"],
                   "manifest_sha256": digest, "method": method, "run_id": run_id, "rng": rng_state(),
                   "reference_unchanged": reference_ok}
        if engine.state:
            payload.update(engine.state.pack())
        with engine.phase("checkpoint"):
            checkpoint = save_checkpoint(out / "checkpoints", payload, keep=2)
        engine.close()
        result = {"status": "completed" if cursor == len(manifest["stream"]) else "segment_complete",
                  "method": method, "cursor": cursor, "optimizer_updates": payload["optimizer_updates"],
                  "manifest_sha256": digest, "code_commit": code_commit, "reference_unchanged": reference_ok,
                  "evaluation_labels_read": False, "elapsed_seconds": time.monotonic() - engine.started,
                  "phase_seconds": engine.timings, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "sampled_device_memory": engine.memory,
                  "cpu_peak_rss_kib": next(int(line.split()[1]) for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("VmHWM:")),
                  "checkpoint": checkpoint}
        atomic_json(out / "segments" / f"c{cursor:06d}.json", result)
        atomic_json(out / "result.json", result)
        engine.event("segment_saved", cursor=cursor, status=result["status"], checkpoint_bytes=checkpoint["state_bytes"])
    except Exception as exc:
        atomic_json(out / ("failure-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json"),
                    {"type": type(exc).__name__, "message": str(exc), "code_commit": code_commit})
        traceback.print_exc()
        raise
    finally:
        if engine:
            engine.close()


if __name__ == "__main__":
    main()
