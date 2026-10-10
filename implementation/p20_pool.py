"""Reward-only pools with a frozen, same-token cached probability check."""
import json
import os
from pathlib import Path
import torch
from continual import Engine
from p16_pool import main as collect
from p19_cache import measure


def validate_manifest(manifest):
    seeds = [87, 88, 89]
    if manifest.get("p20_version") != 1 or manifest["rollout_seeds"] != seeds or manifest.get("frozen_pool_probability_recompute_path") != "incremental_cache" or manifest["training_authorized_in_this_round"]:
        raise ValueError("Frozen reward-only scope differs")
    if len(manifest["stream"]) != 32 or len(manifest["probe"]) != 32 or set(manifest["configurations"]) != {f"pool-{arm}{seed}" for seed in seeds for arm in ["s", "t"]}:
        raise ValueError("Incomplete three-seed paired pools")
    for seed in seeds:
        for arm, sampler in [("s", (.7, .95)), ("t", (1., 1.))]:
            cfg = manifest["configurations"][f"pool-{arm}{seed}"]
            if cfg["seed"] != seed or cfg["method"] != "Frozen SC-8" or cfg["evaluation_labels_allowed"] or cfg.get("model_dtype", "bfloat16") != "bfloat16" or cfg["probability_check_bf16_atol"] != .1 or (cfg["temperature"], cfg["top_p"]) != sampler:
                raise ValueError("Sampling, precision or probability threshold differs")
    if manifest["gate"] != {"minimum_accepted": 16, "minimum_correct": 12, "minimum_precision": .75, "wrong_positive_drop_pp": 10}:
        raise ValueError("Reward thresholds changed")


class CachedCheckEngine(Engine):
    def probability_measurement(self, row):
        if self.training or self.state is not None or self.method != "Frozen SC-8" or self.cfg["probability_check_bf16_atol"] != .1 or any(p.requires_grad or p.dtype != torch.bfloat16 for p in self.actor.parameters()):
            raise ValueError("Cached check is restricted to frozen BF16 reward pools")
        uncached, cached = measure(self, row)
        return {**cached, "atol": .1, "temperature": 1, "top_p": 1, "top_k": 0,
            "model_dtype": "bfloat16", "recomputation_path": "incremental_cache", "paired_uncached_control": uncached,
            "strict_on_policy_consistency_claimed": False, "training_loss_validated": False}


if __name__ == "__main__":
    folder = Path(os.environ["P0_ROOT"]) / "outputs" / os.environ["P2_CAMPAIGN"]
    validate_manifest(json.loads((folder / "manifest.json").read_text()))
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    collect(CachedCheckEngine)
