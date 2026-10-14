"""CPU trust-boundary checks; never hardware acceptance."""
import torch
from kv_paired_accept import compare_tensors


def reject(a, b):
    try:
        compare_tensors(a, b)
    except ValueError:
        return
    raise AssertionError('Invalid paired evidence accepted')


def main():
    a = {'vision': torch.tensor([1.]), 'language': torch.tensor([2.])}
    assert compare_tensors(a, a) == 0
    reject(a, {'vision': a['vision']})
    reject(a, {**a, 'vision': torch.tensor([1., 1.])})
    reject(a, {**a, 'vision': torch.tensor([1.], dtype=torch.float64)})
    reject(a, {**a, 'vision': torch.tensor([float('nan')])})
    reject(a, {**a, 'vision': torch.tensor([1.01])})
    assert 0 < compare_tensors(a, {**a, 'vision': torch.tensor([1.000001])}) < 1e-5
    print('paired gradient coverage/schema/finite/tolerance checks passed; CPU only')


if __name__ == '__main__':
    main()
