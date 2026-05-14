from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_joints_by_group_offsets(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    joint_position_ranges: dict[str, tuple[float, float]],
    joint_velocity_ranges: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset G0 joints with small additive group-wise perturbations.

    Isaac Lab's built-in ``reset_joints_by_scale`` is awkward for G0 because many
    default joint positions are exactly zero, so multiplying by a range leaves
    those joints unperturbed. This event uses additive offsets while still
    clamping to the articulation soft limits.
    """

    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    for pattern, (low, high) in joint_position_ranges.items():
        joint_ids, _ = asset.find_joints(pattern)
        if len(joint_ids) == 0:
            continue
        noise = torch.empty((len(env_ids), len(joint_ids)), device=env.device).uniform_(low, high)
        joint_pos[:, joint_ids] += noise

    for pattern, (low, high) in joint_velocity_ranges.items():
        joint_ids, _ = asset.find_joints(pattern)
        if len(joint_ids) == 0:
            continue
        noise = torch.empty((len(env_ids), len(joint_ids)), device=env.device).uniform_(low, high)
        joint_vel[:, joint_ids] += noise

    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
