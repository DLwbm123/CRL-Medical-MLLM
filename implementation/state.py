"""Persistent full-parameter Adam state and atomic, bounded checkpoints."""
import contextlib
import json
import os
import random
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import torch


def rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count changed")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@contextlib.contextmanager
def inference_without_state_change(model):
    rng, modes = rng_state(), [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            yield
    finally:
        for module, mode in modes:
            module.training = mode
        restore_rng(rng)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class PersistentAdam:
    """One FP32 CPU master and native AdamW state per trainable actor tensor."""

    def __init__(self, actor, cfg):
        self.actor = actor
        self.parameters = dict(actor.named_parameters())
        if not self.parameters or not all(p.requires_grad for p in self.parameters.values()):
            raise ValueError("All actor parameters must remain trainable")
        self.masters = {
            name: torch.nn.Parameter(parameter.detach().to(device="cpu", dtype=torch.float32).clone())
            for name, parameter in self.parameters.items()
        }
        self.optimizer = torch.optim.AdamW(
            list(self.masters.values()), lr=cfg["learning_rate"],
            betas=tuple(cfg["adam_betas"]), eps=cfg["adam_epsilon"],
            weight_decay=cfg["weight_decay"], foreach=False,
        )
        self.max_grad_norm = cfg["max_grad_norm"]
        self.updates = 0
        self.master_identity = {name: id(value) for name, value in self.masters.items()}
        self.moment_identity = {}

    def zero_grad(self):
        self.actor.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)

    @torch.no_grad()
    def sync_actor(self):
        for name, parameter in self.parameters.items():
            parameter.copy_(self.masters[name].to(device=parameter.device, dtype=parameter.dtype))

    def step(self):
        squared = {"vision": 0.0, "language": 0.0}
        for name, parameter in self.parameters.items():
            if parameter.grad is None:
                raise RuntimeError(f"Missing actor gradient: {name}")
            master = self.masters[name]
            if id(master) != self.master_identity[name]:
                raise RuntimeError("Persistent master was replaced")
            master.grad = parameter.grad.detach().to(device="cpu", dtype=torch.float32)
            if not torch.isfinite(master.grad).all():
                raise RuntimeError(f"Nonfinite gradient: {name}")
            role = "vision" if name.startswith("visual.") else "language"
            squared[role] += float(master.grad.norm()) ** 2
            parameter.grad = None
        norm = torch.nn.utils.clip_grad_norm_(
            list(self.masters.values()), self.max_grad_norm, error_if_nonfinite=True,
        )
        self.optimizer.step()
        self.updates += 1
        for name, master in self.masters.items():
            item = self.optimizer.state[master]
            if int(item["step"]) != self.updates:
                raise RuntimeError("Adam step counter did not persist across cases")
            identities = (id(item["exp_avg"]), id(item["exp_avg_sq"]))
            if name in self.moment_identity and self.moment_identity[name] != identities:
                raise RuntimeError("Adam moments were replaced between updates")
            self.moment_identity[name] = identities
        self.sync_actor()
        self.optimizer.zero_grad(set_to_none=True)
        return {
            "gradient_norms": {name: value ** 0.5 for name, value in squared.items()},
            "pre_clip_global_grad_norm": float(norm), "optimizer_updates": self.updates,
            "masters_persisted": True, "moments_persisted": True,
        }

    def pack(self):
        return {
            "master_weights": {name: value.detach() for name, value in self.masters.items()},
            "optimizer": self.optimizer.state_dict(), "optimizer_updates": self.updates,
            "actor_buffers": {name: value.detach().cpu() for name, value in self.actor.named_buffers()},
            "actor_reconstruction": "cast persistent FP32 masters to original actor parameter dtypes",
        }

    def load(self, packed):
        if list(packed["master_weights"]) != list(self.masters):
            raise ValueError("Checkpoint parameter names/order changed")
        with torch.no_grad():
            for name, master in self.masters.items():
                saved = packed["master_weights"][name]
                if saved.shape != master.shape or saved.dtype != torch.float32:
                    raise ValueError(f"Invalid master tensor: {name}")
                master.copy_(saved)
            buffers = dict(self.actor.named_buffers())
            if set(buffers) != set(packed["actor_buffers"]):
                raise ValueError("Actor buffer schema changed")
            for name, buffer in buffers.items():
                buffer.copy_(packed["actor_buffers"][name].to(buffer.device))
        self.optimizer.load_state_dict(packed["optimizer"])
        self.updates = packed["optimizer_updates"]
        self.moment_identity = {}
        for name, master in self.masters.items():
            if self.updates:
                item = self.optimizer.state[master]
                if int(item["step"]) != self.updates:
                    raise ValueError("Invalid restored Adam step")
                if item["exp_avg"].dtype != torch.float32 or item["exp_avg_sq"].dtype != torch.float32:
                    raise ValueError("Restored Adam moments must be FP32")
                self.moment_identity[name] = (id(item["exp_avg"]), id(item["exp_avg_sq"]))
        self.sync_actor()
        self.zero_grad()


def save_checkpoint(folder, state, keep=2):
    """Retain only this run's last two complete checkpoints; preserve partial failures."""
    start = time.monotonic()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    name = f"c{state['cursor']:06d}-u{state['optimizer_updates']:06d}"
    target = folder / name
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {name}")
    temporary = folder / (".pending-" + uuid.uuid4().hex)
    temporary.mkdir()
    with (temporary / "state.pt").open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    metadata = {key: state[key] for key in [
        "cursor", "processed_ids", "optimizer_updates", "configuration", "code_commit",
        "model_revision", "manifest_sha256", "method", "run_id",
    ]}
    metadata.update(state_bytes=(temporary / "state.pt").stat().st_size,
                    checkpoint_seconds=time.monotonic() - start)
    atomic_json(temporary / "metadata.json", metadata)
    os.replace(temporary, target)
    atomic_json(folder / "latest.json", {"directory": name, **metadata})
    complete = sorted(path for path in folder.glob("c*-u*") if path.is_dir())
    for old in complete[:-keep]:
        shutil.rmtree(old)
    return metadata


def load_checkpoint(folder, configuration, manifest_sha256, method):
    folder = Path(folder)
    latest = json.loads((folder / "latest.json").read_text())
    checkpoint = folder / latest["directory"] / "state.pt"
    # Only load checkpoints created by this run. They include Python/NumPy RNG objects.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    if state["configuration"] != configuration or state["manifest_sha256"] != manifest_sha256 or state["method"] != method:
        raise ValueError("Resume configuration, manifest or method changed")
    if state["cursor"] != len(state["processed_ids"]):
        raise ValueError("Checkpoint cursor disagrees with processed IDs")
    return state
