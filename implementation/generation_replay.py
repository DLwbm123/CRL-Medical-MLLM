"""Replay the sampled prefixes with live gradients through the entire KV cache."""
import torch
from torch.utils.checkpoint import checkpoint
from p19_cache import cached_response


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

    if torch.is_grad_enabled() and any(p.requires_grad for p in model.parameters()):
        return checkpoint(replay, response, use_reentrant=False, preserve_rng_state=True)
    return replay(response)
