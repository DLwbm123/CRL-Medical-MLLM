"""Reuse complete-scope gates and exercise the changed differentiable replay."""
from test_p24 import main as contract

if __name__ == '__main__':
    contract()
    from test_p19 import main as cache
    from test_generation_replay import main as replay
    cache(); replay()
