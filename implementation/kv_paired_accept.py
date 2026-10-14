"""Engineering-only paired wrapper around the unchanged actual acceptance suite."""
import json
import os
import time
from pathlib import Path
import torch
import p24_accept
from continual import Engine
from state import atomic_json, rng_state, restore_rng


def compare_tensors(actual, expected, atol=1e-5):
    if set(actual) != set(expected):
        raise ValueError('Paired tensor coverage differs')
    maximum = 0.0
    for name, tensor in actual.items():
        other = expected[name]
        if tensor.shape != other.shape or tensor.dtype != other.dtype:
            raise ValueError('Paired tensor schema differs')
        error = float((tensor.float().cpu() - other.float()).abs().max())
        if not torch.isfinite(torch.tensor(error)) or error > atol:
            raise ValueError('Paired full gradient exceeds unchanged 1e-5 tolerance')
        maximum = max(maximum, error)
    return maximum


class PairedEngine(Engine):
    def __init__(self, *args, **kwargs):
        self.side = os.environ['KV_IMPLEMENTATION']
        if self.side not in ['original', 'candidate']:
            raise ValueError('Unknown paired implementation')
        self.pairs = Path(os.environ['KV_PAIR_FOLDER'])
        super().__init__(*args, **kwargs)
        self.arm = self.cfg['p10_arm']
        self.input_id = None
        self.challenge = -1
        self.update_index = 0
        self.pair_metrics = []
        self.pair_io_seconds = 0.0
        self.step_original = self.state.step
        self.state.step = self.paired_step

    def pair_path(self, suffix):
        return self.pairs / (self.arm + '-' + suffix + '.pt')

    def write_pair(self, path, value):
        start = time.monotonic()
        if path.exists():
            raise FileExistsError('Refusing to replace paired evidence')
        torch.save(value, path)
        self.pair_io_seconds += time.monotonic() - start
        self.deadline.check()

    def read_pair(self, path):
        start = time.monotonic()
        value = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        self.pair_io_seconds += time.monotonic() - start
        self.deadline.check()
        return value

    def encode(self, row):
        encoded = super().encode(row)
        if row['id'] != self.input_id:
            self.input_id = row['id']
            self.challenge += 1
            if self.challenge >= 3:
                raise ValueError('Unexpected additional engineering input')
            path = self.pair_path('initial-' + str(self.challenge))
            if self.side == 'original':
                self.write_pair(path, {'state': self.state.pack(), 'rng': rng_state()})
            else:
                value = self.read_pair(path)
                self.state.load(value['state'])
                restore_rng(value['rng'])
                del value
        return encoded

    def response_logits(self, model, encoded, response):
        if self.side == 'candidate':
            from block_generation_replay import response_logits
            return response_logits(model, encoded, response, self.deadline.check)
        return super().response_logits(model, encoded, response)

    def update(self, encoded, responses, advantages, drift):
        # Pair every optimizer call, including the original t/h recovery branches.
        key = str(self.update_index)
        before = self.pair_path('before-' + key)
        if self.side == 'original':
            self.write_pair(before, {'rng': rng_state(),
                'responses': [r.detach().cpu() for r in responses], 'advantages': advantages.cpu()})
        else:
            value = self.read_pair(before)
            if len(responses) != len(value['responses']) or any(not torch.equal(a.cpu(), b) for a, b in zip(responses, value['responses'])):
                raise ValueError('Paired original sampler responses differ')
            if not torch.equal(advantages.cpu(), value['advantages']):
                raise ValueError('Paired engineering advantages differ')
            # Preserve each candidate recovery branch's own Adam/master state.
            restore_rng(value['rng'])
            del value
        result = super().update(encoded, responses, advantages, drift)
        self.update_index += 1
        return result

    def paired_step(self):
        key = str(self.update_index)
        path = self.pair_path('gradient-' + key)
        gradients = {name: p.grad.detach().cpu() for name, p in self.state.parameters.items() if p.grad is not None}
        if len(gradients) != len(self.state.parameters):
            raise ValueError('Incomplete paired full-parameter gradient')
        if self.side == 'original':
            self.write_pair(path, gradients)
            error = 0.0
        else:
            expected = self.read_pair(path)
            error = compare_tensors(gradients, expected)
            del expected
        del gradients
        result = self.step_original()
        self.pair_metrics.append({'challenge': self.challenge, 'update_index': self.update_index,
            'full_gradient_max_abs_difference': error, 'gradient_atol': 1e-5})
        return result

    def close(self):
        atomic_json(self.out / 'paired_engineering.json', {
            'implementation': self.side, 'challenge_count': self.challenge + 1,
            'optimizer_calls': self.update_index, 'paired_updates': self.pair_metrics,
            'phase_seconds': self.timings, 'pair_evidence_io_seconds': self.pair_io_seconds,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'labels_read': False, 'temporary_weights_discarded': True})
        super().close()


def main():
    if os.environ['KV_IMPLEMENTATION'] == 'candidate':
        root = Path(os.environ['P0_ROOT'])
        campaign = os.environ['P2_CAMPAIGN']
        if not campaign.endswith('-candidate'):
            raise ValueError('Candidate campaign role differs')
        original = root / 'outputs' / (campaign[:-len('-candidate')] + '-original')
        result = json.loads((original / 'acceptance.json').read_text())
        if not result['passed'] or set(result['arms']) != {'s', 't', 'v', 'h'}:
            raise ValueError('Complete original acceptance must precede candidate')
        for value in result['arms'].values():
            if not value['passed'] or not value['memory_challenge']['passed'] or not value['memory_challenge']['largest_prefix']['passed']:
                raise ValueError('Original challenge coverage is incomplete')
    p24_accept.Engine = PairedEngine
    p24_accept.main()


if __name__ == '__main__':
    main()
