"""Explicit policy loading and metadata helpers for sim2sim audits."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.sim2sim import g0_sim2sim_config as cfg
except ModuleNotFoundError:
    import g0_sim2sim_config as cfg


def require_absolute_path(path: str | Path, label: str) -> Path:
    """Return an existing absolute path or raise a checkpoint-identity error."""

    resolved = Path(path)
    if not resolved.is_absolute():
        raise ValueError(f"{label} must be an absolute path. Refusing ambiguous path: {path}")
    if not resolved.exists():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_run_folder(path: str | Path) -> str:
    path = Path(path)
    if path.parent.name == "exported":
        return str(path.parent.parent)
    return str(path.parent)


def policy_metadata(
    policy_path: str | Path | None,
    *,
    task: str,
    command: np.ndarray | list[float] | tuple[float, ...],
    steps: int,
) -> dict[str, np.ndarray]:
    """Build metadata arrays that every rollout artifact should carry."""

    command_array = np.asarray(command, dtype=np.float64)
    metadata: dict[str, np.ndarray] = {
        "task": np.asarray(task),
        "command": command_array,
        "steps": np.asarray(int(steps)),
        "action_dim": np.asarray(cfg.get_action_dim()),
        "obs_dim": np.asarray(cfg.get_policy_observation_dim()),
        "joint_names": np.asarray(cfg.get_joint_names()),
        "action_scale": np.asarray(cfg.ACTION_SCALE),
    }
    if policy_path is None:
        metadata.update(
            {
                "policy_path": np.asarray(""),
                "policy_filename": np.asarray(""),
                "policy_sha256": np.asarray(""),
                "checkpoint_run_folder": np.asarray(""),
            }
        )
        return metadata

    absolute = require_absolute_path(policy_path, "policy/checkpoint path")
    metadata.update(
        {
            "policy_path": np.asarray(str(absolute)),
            "policy_filename": np.asarray(absolute.name),
            "policy_sha256": np.asarray(sha256_file(absolute)),
            "checkpoint_run_folder": np.asarray(checkpoint_run_folder(absolute)),
        }
    )
    return metadata


def metadata_from_npz(data: np.lib.npyio.NpzFile) -> dict[str, np.ndarray]:
    keys = [
        "policy_path",
        "policy_filename",
        "policy_sha256",
        "checkpoint_run_folder",
        "task",
        "command",
        "steps",
        "action_dim",
        "obs_dim",
        "joint_names",
        "action_scale",
    ]
    return {key: np.asarray(data[key]) for key in keys if key in data.files}


class RslRlActorCheckpoint:
    """Minimal deterministic actor loader for RSL-RL checkpoints.

    The current G0 checkpoints store `actor_state_dict` with an MLP under keys
    like `mlp.0.weight`, `mlp.2.weight`, ..., and `distribution.std_param`.
    For deployment inference we use the actor mean output, matching the usual
    deterministic RSL-RL inference path.
    """

    def __init__(self, checkpoint_path: str | Path, device: str):
        import torch
        from torch import nn

        self.torch = torch
        self.device = device
        checkpoint = torch.load(str(checkpoint_path), map_location=device)
        if not isinstance(checkpoint, dict) or "actor_state_dict" not in checkpoint:
            raise ValueError(f"Checkpoint does not contain actor_state_dict: {checkpoint_path}")
        state = checkpoint["actor_state_dict"]
        linear_keys = sorted(
            key for key in state.keys() if key.startswith("mlp.") and key.endswith(".weight")
        )
        if not linear_keys:
            raise ValueError(f"Could not infer actor MLP from checkpoint: {checkpoint_path}")
        layers: list[nn.Module] = []
        stripped_state: dict[str, Any] = {}
        for layer_index, key in enumerate(linear_keys):
            original_index = key.split(".")[1]
            weight = state[f"mlp.{original_index}.weight"]
            bias = state[f"mlp.{original_index}.bias"]
            linear = nn.Linear(int(weight.shape[1]), int(weight.shape[0]))
            layers.append(linear)
            stripped_state[f"{len(layers) - 1}.weight"] = weight
            stripped_state[f"{len(layers) - 1}.bias"] = bias
            if layer_index != len(linear_keys) - 1:
                layers.append(nn.ELU())
        self.model = nn.Sequential(*layers).to(device)
        self.model.load_state_dict(stripped_state, strict=True)
        self.model.eval()

    def __call__(self, obs):
        with self.torch.no_grad():
            return self.model(obs)


def load_policy(policy_path: str | Path, device: str):
    """Load an explicit absolute policy path as TorchScript or raw RSL-RL actor checkpoint."""

    absolute = require_absolute_path(policy_path, "policy/checkpoint path")
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Torch is required for policy inference.") from exc

    try:
        policy = torch.jit.load(str(absolute), map_location=device)
        policy.eval()
        return policy, "torchscript"
    except Exception:
        return RslRlActorCheckpoint(absolute, device), "rsl_rl_actor_checkpoint"
