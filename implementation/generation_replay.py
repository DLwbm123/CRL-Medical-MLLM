"""Replay the sampled prefixes with live gradients through the entire KV cache."""
import torch
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import DynamicCache
from p19_cache import cached_response


def history_cache(history):
    if history is None: return None
    frames = []
    while history is not None:
        frames.append(history[1]); history = history[0]
    frames.reverse()
    if not frames[0] or len(frames[0]) % 2 or any(len(f) != len(frames[0]) for f in frames):
        raise ValueError("Generation cache layer layout changed")
    return DynamicCache.from_legacy_cache(tuple(
        (torch.cat([f[i] for f in frames], dim=-2), torch.cat([f[i+1] for f in frames], dim=-2))
        for i in range(0, len(frames[0]), 2)))


def response_logits(model, encoded, response, check):
    if response.numel() == 0:
        raise ValueError("Cannot replay an empty response")

    def replay(tokens):
        modes = [(module, module.training) for module in model.modules()]
        # Training mode disables the model's cache; eval leaves gradients enabled.
        model.eval()
        try:
            return cached_response(model, encoded, tokens, check)
        finally:
            for module, training in modes: module.training = training

    if not torch.is_grad_enabled() or not any(p.requires_grad for p in model.parameters()):
        return replay(response)
    prefix = encoded["input_ids"]; mask = encoded["attention_mask"]
    vision = {k:v for k,v in encoded.items() if k not in {"input_ids", "attention_mask"}}
    history = None; logits = []
    for index, token in enumerate(response):
        # Bind immutable history; reusing the growing DynamicCache corrupts backward replay.
        def step(unused, previous=history, current=prefix, attention=mask, index=index):
            check(); modes = [(m, m.training) for m in model.modules()]; model.eval()
            try:
                cache = history_cache(previous)
                positions = torch.arange(current.shape[-1], device=current.device) if index == 0 else torch.tensor([current.shape[-1]-1], device=current.device)
                inputs = model.prepare_inputs_for_generation(current, past_key_values=cache,
                    attention_mask=attention, cache_position=positions, use_cache=True, **vision)
                output = model(**inputs, return_dict=True)
                if not isinstance(output.past_key_values, DynamicCache):
                    raise ValueError("Accepted DynamicCache is required")
                count = current.shape[-1] if index == 0 else 1
                # Clone slices so they do not retain each complete growing cache allocation.
                chunks = tuple(t[..., -count:, :].clone() for pair in output.past_key_values.to_legacy_cache() for t in pair)
                return (output.logits[0, -1].clone(), *chunks)
            finally:
                for module, training in modes: module.training = training
        values = checkpoint(step, token, use_reentrant=False, preserve_rng_state=True)
        logits.append(values[0]); history = (history, values[1:])
        prefix = torch.cat([prefix, token.reshape(1,1)], -1)
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], -1)
    return torch.stack(logits)
