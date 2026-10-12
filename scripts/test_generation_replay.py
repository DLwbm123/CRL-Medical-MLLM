"""A later-token loss must retain the prefill vision gradient through the cache."""
import json
from types import SimpleNamespace
import torch
from transformers.cache_utils import DynamicCache
from unittest.mock import patch
from continual import forward_response
from generation_replay import response_logits
from p24_accept import require_probability


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = torch.nn.Linear(2, 3, bias=False)
        self.embedding = torch.nn.Embedding(8, 3)
        self.language = torch.nn.Linear(3, 8, bias=False)

    def prepare_inputs_for_generation(self, prefix, past_key_values, attention_mask, cache_position, use_cache, pixel_values):
        assert use_cache and attention_mask.shape[-1] == prefix.shape[-1]
        assert cache_position.tolist() == (list(range(prefix.shape[-1])) if past_key_values is None else [prefix.shape[-1]-1])
        return dict(input_ids=prefix[:, cache_position], past_key_values=past_key_values, use_cache=True,
                    pixel_values=pixel_values if past_key_values is None else None)

    def forward(self, input_ids, past_key_values=None, pixel_values=None, **_):
        state = self.visual(pixel_values) if past_key_values is None else past_key_values.key_cache[0][:, 0, -1]
        values = []; states = []
        for token in input_ids[0]:
            state = torch.tanh(state + self.embedding(token))
            values.append(self.language(state)); states.append(state)
        cache = DynamicCache() if past_key_values is None else past_key_values
        keys = torch.stack(states, 1).unsqueeze(1)
        cache.update(keys, keys, 0)
        return SimpleNamespace(logits=torch.stack(values, 1), past_key_values=cache)


def main():
    torch.manual_seed(3)
    model = Tiny(); model.train(); model.visual.eval()
    modes = [m.training for m in model.modules()]
    encoded = dict(input_ids=torch.tensor([[1, 2, 3]]), attention_mask=torch.ones(1, 3, dtype=torch.long),
                   pixel_values=torch.tensor([[.3, .7]]))
    response = torch.tensor([4, 5, 6])
    expected = forward_response(model, encoded, response).detach()
    with torch.no_grad(): observed = response_logits(model, encoded, response, lambda: None)
    assert torch.equal(expected, observed) and [m.training for m in model.modules()] == modes
    replay = response_logits(model, encoded, response, lambda: None)
    assert torch.equal(replay.detach(), observed)
    replay[-1, 0].backward()
    replay_grads = {n: p.grad.clone() for n, p in model.named_parameters()}
    assert replay_grads['visual.weight'].norm() > 0 and replay_grads['language.weight'].norm() > 0
    model.zero_grad(); forward_response(model, encoded, response)[-1, 0].backward()
    assert all(torch.equal(p.grad, replay_grads[n]) for n, p in model.named_parameters())
    assert [m.training for m in model.modules()] == modes
    model.zero_grad(); frames = []
    def direct(fn, *args, **kwargs):
        result = fn(*args); frames.append(result[1:]); return result
    with patch("generation_replay.checkpoint", side_effect=direct):
        response_logits(model, encoded, response, lambda: None)[-1, 0].backward()
    assert all(torch.equal(p.grad, replay_grads[n]) for n, p in model.named_parameters())
    retained = sum(t.numel() for frame in frames for t in frame)
    assert retained == 2 * 3 * (encoded["input_ids"].shape[-1] + len(response) - 1)
    assert all(t.untyped_storage().nbytes() == t.numel() * t.element_size() for frame in frames for t in frame)
    def stopped(): raise TimeoutError('synthetic deadline')
    try: response_logits(model, encoded, response, stopped)
    except TimeoutError: pass
    else: raise AssertionError('Replay ignored stop')
    assert [m.training for m in model.modules()] == modes
    for value in [.100001, float('nan')]:
        try: require_probability(dict(within_tolerance=True, atol=.1, max_logp_difference=value))
        except ValueError: pass
        else: raise AssertionError('Probability guard relaxed')
    print(json.dumps(dict(late_token_vision_gradient_preserved=True, checkpoint_gradient_exact=True,
                         stats_and_grad_equal=True, modes_and_deadline_checked=True, original_point_one_guard=True, cloned_linear_KV_storage=True)))


if __name__ == '__main__': main()
