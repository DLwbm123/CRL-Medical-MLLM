"""Preparation-only replay candidate; production Engine keeps the frozen replay."""
import torch
from torch.utils.checkpoint import checkpoint
from transformers.cache_utils import DynamicCache
from generation_replay import history_cache
from p19_cache import cached_response


BLOCK_TOKENS = 8


def response_logits(model, encoded, response, check):
    if response.numel() == 0:
        raise ValueError("Cannot replay an empty response")
    if not torch.is_grad_enabled() or not any(p.requires_grad for p in model.parameters()):
        modes = [(m, m.training) for m in model.modules()]
        model.eval()
        try:
            return cached_response(model, encoded, response, check)
        finally:
            for module, training in modes:
                module.training = training

    prefix = encoded["input_ids"]
    mask = encoded["attention_mask"]
    vision = {k: v for k, v in encoded.items() if k not in {"input_ids", "attention_mask"}}
    history = None
    logits = []
    for start in range(0, len(response), BLOCK_TOKENS):
        tokens = response[start:start + BLOCK_TOKENS]

        # Each checkpoint owns a fresh cache; only immutable new chunks cross blocks.
        def block(tokens, previous=history, current=prefix, attention=mask, start=start):
            check()
            modes = [(m, m.training) for m in model.modules()]
            model.eval()
            try:
                cache = history_cache(previous)
                outputs = []
                count = 0
                for offset, token in enumerate(tokens):
                    check()
                    first = start + offset == 0
                    positions = (torch.arange(current.shape[-1], device=current.device) if first
                                 else torch.tensor([current.shape[-1] - 1], device=current.device))
                    inputs = model.prepare_inputs_for_generation(current, past_key_values=cache,
                        attention_mask=attention, cache_position=positions, use_cache=True, **vision)
                    output = model(**inputs, return_dict=True)
                    cache = output.past_key_values
                    if not isinstance(cache, DynamicCache):
                        raise ValueError("Accepted DynamicCache is required")
                    outputs.append(output.logits[0, -1].clone())
                    count += current.shape[-1] if first else 1
                    current = torch.cat([current, token.reshape(1, 1)], -1)
                    attention = torch.cat([attention, torch.ones_like(attention[:, :1])], -1)
                chunks = tuple(t[..., -count:, :].clone()
                               for pair in cache.to_legacy_cache() for t in pair)
                return (torch.stack(outputs), *chunks)
            finally:
                for module, training in modes:
                    module.training = training

        values = checkpoint(block, tokens, use_reentrant=False, preserve_rng_state=True)
        logits.append(values[0])
        history = (history, values[1:])
        prefix = torch.cat([prefix, tokens.reshape(1, -1)], -1)
        mask = torch.cat([mask, torch.ones_like(tokens).reshape(1, -1)], -1)
    return torch.cat(logits)
