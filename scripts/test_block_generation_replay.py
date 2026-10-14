"""CPU-only candidate checks; no model download, GPU admission or experiment entry."""
import copy
import json
from pathlib import Path
from unittest.mock import patch
import torch
import block_generation_replay as candidate
from generation_replay import response_logits as original
from core import objective, select_and_band, token_statistics
from p24_contract import configurations
from state import PersistentAdam
from test_generation_replay import Tiny


def main():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.manual_seed(3)
    model = Tiny()
    model.train()
    model.visual.eval()
    encoded = dict(input_ids=torch.tensor([[1, 2, 3]]),
                   attention_mask=torch.ones(1, 3, dtype=torch.long),
                   pixel_values=torch.tensor([[.3, .7]]))
    modes = [m.training for m in model.modules()]
    cfgs = configurations(json.loads((Path(__file__).resolve().parents[1] / 'configs/p2_c.json').read_text()))
    errors = []
    for length in [1, 7, 8, 9, 17, 33]:
        response = torch.arange(length) % 8
        with torch.no_grad():
            expected = original(model, encoded, response, lambda: None)
            actual = candidate.response_logits(model, encoded, response, lambda: None)
        assert torch.equal(expected, actual)
        grads = []
        for replay in [original, candidate.response_logits]:
            model.zero_grad()
            logits = replay(model, encoded, response, lambda: None)
            assert torch.equal(logits.detach(), expected)
            logits[-1, 0].backward()
            grads.append({n: p.grad.clone() for n, p in model.named_parameters()})
            assert [m.training for m in model.modules()] == modes
        for n in grads[0]:
            torch.testing.assert_close(grads[0][n], grads[1][n], atol=1e-6, rtol=1e-6)
        assert grads[1]['visual.weight'].norm() > 0
        errors.append(max(float((grads[0][n] - grads[1][n]).abs().max()) for n in grads[0]))

    response = torch.arange(17) % 8
    # Exercise the unchanged real objectives, including full-response KL in h.
    with torch.no_grad():
        behavior = original(model, encoded, response, lambda: None)
        old_logp, entropy, _ = token_statistics(behavior, response)
        selection, lower, upper, _ = select_and_band(entropy, 16)
    for arm in ['s', 't', 'v', 'h']:
        gradients = []
        for replay in [original, candidate.response_logits]:
            model.zero_grad()
            logits = replay(model, encoded, response, lambda: None)
            logp, ent, kl = token_statistics(logits, response, behavior + .01 * torch.arange(8))
            mask = selection if arm == 's' else torch.ones_like(selection)
            loss, _ = objective(logp, old_logp, ent, kl, torch.tensor(.7), mask, lower, upper,
                                len(response), int(mask.sum()), cfgs[arm]['103'], selection if arm == 'h' else None)
            loss.backward()
            gradients.append({n: p.grad.clone() for n, p in model.named_parameters()})
        for n in gradients[0]:
            torch.testing.assert_close(gradients[0][n], gradients[1][n], atol=1e-6, rtol=1e-6)

    frames = []
    def direct(fn, *args, **kwargs):
        value = fn(*args)
        frames.append(value[1:])
        return value
    with patch.object(candidate, 'checkpoint', side_effect=direct):
        candidate.response_logits(model, encoded, response, lambda: None)[-1, 0].backward()
    assert len(frames) == 3
    assert sum(t.numel() for f in frames for t in f) == 2 * 3 * (3 + len(response) - 1)
    assert all(t.untyped_storage().nbytes() == t.numel() * t.element_size() for f in frames for t in f)
    for replay_module, expected_calls in [('generation_replay', 17), ('block_generation_replay', 3)]:
        import importlib
        module = importlib.import_module(replay_module)
        with patch.object(module, 'checkpoint', side_effect=direct), patch.object(module, 'history_cache', wraps=module.history_cache) as rebuild:
            module.response_logits(model, encoded, response, lambda: None)
            assert rebuild.call_count == expected_calls
    try:
        candidate.response_logits(model, encoded, response[:0], lambda: None)
    except ValueError:
        pass
    else:
        raise AssertionError('Empty response accepted')
    def failed_forward(*args, **kwargs):
        raise RuntimeError('synthetic model failure')
    with patch.object(model, 'forward', new=failed_forward):
        try:
            candidate.response_logits(model, encoded, response, lambda: None)
        except RuntimeError as error:
            assert str(error) == 'synthetic model failure'
        else:
            raise AssertionError('Model error swallowed')
    assert [m.training for m in model.modules()] == modes

    # A second graph must not mutate the first graph's captured cache.
    model.zero_grad()
    first = candidate.response_logits(model, encoded, response, lambda: None)
    second = candidate.response_logits(model, encoded, response.flip(0), lambda: None)
    (first[-1, 0] + second[-1, 1]).backward()
    assert model.visual.weight.grad.norm() > 0
    for backward in [False, True]:
        def stopped():
            raise TimeoutError('synthetic deadline')
        try:
            if backward:
                callback = [lambda: None]
                graph = candidate.response_logits(model, encoded, response, lambda: callback[0]())
                callback[0] = stopped
                graph[-1, 0].backward()
            else:
                candidate.response_logits(model, encoded, response, stopped)
        except TimeoutError:
            pass
        else:
            raise AssertionError('Deadline ignored')
        assert [m.training for m in model.modules()] == modes

    actor = copy.deepcopy(model)
    state = PersistentAdam(actor, cfgs['h']['103'])
    def update():
        state.zero_grad()
        candidate.response_logits(actor, encoded, response, lambda: None)[-1, 0].backward()
        state.step()
    update()
    saved = copy.deepcopy(state.pack())
    update()
    expected = copy.deepcopy(state.pack())
    state.load(saved)
    update()
    actual = state.pack()
    for n in expected['master_weights']:
        assert torch.equal(expected['master_weights'][n], actual['master_weights'][n])
    for key in expected['optimizer']['state']:
        for n in expected['optimizer']['state'][key]:
            assert torch.equal(expected['optimizer']['state'][key][n], actual['optimizer']['state'][key][n])
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(CPU_only=True, block_tokens=8, lengths=[1,7,8,9,17,33],
        max_absolute_gradient_error=max(errors), all_four_objectives_preserved=True,
        late_vision_gradient=True, linear_retained_KV=True, modes_and_deadlines=True,
        history_rebuild_calls_for_17_tokens=dict(original=17, candidate=3),
        CPU_same_process_Adam_recovery_exact=True, hardware_acceptance=False,
        GPU_speedup_measured=False)))


if __name__ == '__main__':
    main()
