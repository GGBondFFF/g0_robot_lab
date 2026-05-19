from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch


def _to_tensor(value: Any) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach()
    if hasattr(value, "detach") and hasattr(value, "shape"):
        return value.detach()
    try:
        return torch.as_tensor(value)
    except Exception:
        return None


def _scalar_bool(value: Any) -> bool:
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def _safe_term(termination_manager: Any, name: str, mask: torch.Tensor | None = None) -> int:
    if termination_manager is None:
        return 0
    try:
        value = termination_manager.get_term(name)
    except Exception:
        return 0
    tensor = _to_tensor(value)
    if tensor is None:
        return int(bool(value))
    if tensor.ndim == 0:
        return int(_scalar_bool(tensor))
    if mask is not None:
        mask_tensor = mask.to(dtype=torch.bool, device=tensor.device)
        tensor = tensor[mask_tensor]
        if tensor.numel() == 0:
            return 0
    return int(tensor.to(dtype=torch.int64).sum().item())


def quat_wxyz_to_euler_deg(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quat = quat_wxyz.detach().to(dtype=torch.float64)
    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    scale = 180.0 / math.pi
    return roll * scale, pitch * scale, yaw * scale


@dataclass
class TensorStats:
    min: float | None = None
    max: float | None = None
    sum: float = 0.0
    sum_sq: float = 0.0
    count: int = 0

    def update(self, value: Any) -> None:
        tensor = _to_tensor(value)
        if tensor is None:
            return
        flat = tensor.to(dtype=torch.float64).reshape(-1)
        if flat.numel() == 0:
            return
        current_min = float(flat.min().item())
        current_max = float(flat.max().item())
        self.min = current_min if self.min is None else min(self.min, current_min)
        self.max = current_max if self.max is None else max(self.max, current_max)
        self.sum += float(flat.sum().item())
        self.sum_sq += float((flat * flat).sum().item())
        self.count += int(flat.numel())

    def as_dict(self) -> dict[str, float | int | None]:
        mean = self.sum / self.count if self.count else None
        variance = (self.sum_sq / self.count - mean * mean) if self.count and mean is not None else None
        std = math.sqrt(max(variance, 0.0)) if variance is not None else None
        return {
            "min": self.min,
            "max": self.max,
            "mean": mean,
            "std": std,
            "count": self.count,
        }


@dataclass
class RolloutMetrics:
    expected_action_dim: int
    num_envs: int
    effort_ratio_threshold: float = 0.9
    raw_action_stats: TensorStats = field(default_factory=TensorStats)
    clipped_action_stats: TensorStats = field(default_factory=TensorStats)
    target_joint_pos_stats: TensorStats = field(default_factory=TensorStats)
    target_delta_stats: TensorStats = field(default_factory=TensorStats)
    root_z_stats: TensorStats = field(default_factory=TensorStats)
    joint_pos_stats: TensorStats = field(default_factory=TensorStats)
    joint_vel_stats: TensorStats = field(default_factory=TensorStats)
    torque_stats: TensorStats = field(default_factory=TensorStats)
    effort_ratio_stats: TensorStats = field(default_factory=TensorStats)
    roll_stats: TensorStats = field(default_factory=TensorStats)
    pitch_stats: TensorStats = field(default_factory=TensorStats)
    yaw_stats: TensorStats = field(default_factory=TensorStats)
    total_steps: int = 0
    action_out_of_range_count: int = 0
    reset_count: int = 0
    base_height_count: int = 0
    bad_orientation_count: int = 0
    time_out_count: int = 0
    obs_non_finite_step: int | None = None
    actions_non_finite_step: int | None = None
    action_shape_error: dict[str, Any] | None = None
    target_delta_available: bool = False
    target_joint_pos_available: bool = False
    torque_available: bool = False
    effort_ratio_available: bool = False
    effort_ratio_steps_above_threshold: int = 0
    joint_limit_margin_min: float | None = None

    def record_obs(self, step: int, obs_tensor: torch.Tensor) -> None:
        if not torch.isfinite(obs_tensor).all() and self.obs_non_finite_step is None:
            self.obs_non_finite_step = step

    def record_actions(self, step: int, raw_actions: torch.Tensor, clipped_actions: torch.Tensor | None = None) -> None:
        if tuple(raw_actions.shape) != (self.num_envs, self.expected_action_dim) and self.action_shape_error is None:
            self.action_shape_error = {
                "step": step,
                "expected": [self.num_envs, self.expected_action_dim],
                "actual": list(raw_actions.shape),
            }
        if not torch.isfinite(raw_actions).all() and self.actions_non_finite_step is None:
            self.actions_non_finite_step = step
        self.raw_action_stats.update(raw_actions)
        self.action_out_of_range_count += int(((raw_actions < -1.0) | (raw_actions > 1.0)).sum().item())
        if clipped_actions is not None:
            self.clipped_action_stats.update(clipped_actions)

    def record_step(
        self,
        *,
        step: int,
        robot: Any,
        dones: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        termination_manager: Any,
        target_joint_pos: torch.Tensor | None,
        target_delta: torch.Tensor | None,
        torque: torch.Tensor | None,
        effort_ratio: torch.Tensor | None,
        joint_limit_margin: torch.Tensor | None,
    ) -> None:
        del step, terminated
        data = robot.data
        done_mask = dones.to(dtype=torch.bool)
        self.total_steps += 1
        self.reset_count += int(done_mask.any().item())
        self.base_height_count += _safe_term(termination_manager, "base_height", done_mask)
        self.bad_orientation_count += _safe_term(termination_manager, "bad_orientation", done_mask)
        self.time_out_count += max(
            int(truncated.to(dtype=torch.bool).any().item()),
            _safe_term(termination_manager, "time_out", done_mask),
        )

        self.root_z_stats.update(data.root_pos_w[:, 2])
        self.joint_pos_stats.update(getattr(data, "joint_pos", None))
        self.joint_vel_stats.update(getattr(data, "joint_vel", None))

        root_quat = _to_tensor(getattr(data, "root_quat_w", None))
        if root_quat is not None and root_quat.ndim == 2 and root_quat.shape[1] >= 4:
            roll, pitch, yaw = quat_wxyz_to_euler_deg(root_quat[:, :4])
            self.roll_stats.update(roll)
            self.pitch_stats.update(pitch)
            self.yaw_stats.update(yaw)

        if target_joint_pos is not None:
            self.target_joint_pos_available = True
            self.target_joint_pos_stats.update(target_joint_pos)
        if target_delta is not None:
            self.target_delta_available = True
            self.target_delta_stats.update(target_delta.abs())
        if torque is not None:
            self.torque_available = True
            self.torque_stats.update(torque)
        if effort_ratio is not None:
            self.effort_ratio_available = True
            self.effort_ratio_stats.update(effort_ratio)
            per_env_peak = effort_ratio.max(dim=1).values if effort_ratio.ndim > 1 else effort_ratio.reshape(-1)
            self.effort_ratio_steps_above_threshold += int((per_env_peak > self.effort_ratio_threshold).any().item())
        if joint_limit_margin is not None:
            current_min = float(joint_limit_margin.min().item())
            self.joint_limit_margin_min = (
                current_min if self.joint_limit_margin_min is None else min(self.joint_limit_margin_min, current_min)
            )

    def contract_errors(self) -> list[str]:
        errors: list[str] = []
        if self.obs_non_finite_step is not None:
            errors.append(f"non-finite observations at step {self.obs_non_finite_step}")
        if self.actions_non_finite_step is not None:
            errors.append(f"non-finite actions at step {self.actions_non_finite_step}")
        if self.action_shape_error is not None:
            errors.append(
                "action shape mismatch: "
                f"expected {self.action_shape_error['expected']}, got {self.action_shape_error['actual']}"
            )
        return errors

    def warnings(self) -> list[str]:
        warnings: list[str] = []
        if self.action_out_of_range_count:
            warnings.append(f"raw policy produced {self.action_out_of_range_count} actions outside [-1, 1]")
            warnings.append("deployment must apply equivalent action clipping before any LowCmd path")
        if self.reset_count:
            warnings.append(f"environment reset count: {self.reset_count}")
        if self.base_height_count:
            warnings.append(f"base_height terminations: {self.base_height_count}")
        if self.bad_orientation_count:
            warnings.append(f"bad_orientation terminations: {self.bad_orientation_count}")
        if self.time_out_count:
            warnings.append(f"time_out / truncated count: {self.time_out_count}")
        if self.effort_ratio_available:
            ratio_max = self.effort_ratio_stats.as_dict()["max"]
            if ratio_max is not None and ratio_max > self.effort_ratio_threshold:
                warnings.append(
                    f"effort ratio max {ratio_max:.4f} exceeded threshold {self.effort_ratio_threshold:.2f}"
                )
        return warnings

    def summary(self) -> dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "raw_policy_action": {
                **self.raw_action_stats.as_dict(),
                "shape": [self.num_envs, self.expected_action_dim],
                "out_of_range_count": self.action_out_of_range_count,
            },
            "clipped_action": {
                **self.clipped_action_stats.as_dict(),
                "shape": [self.num_envs, self.expected_action_dim],
                "clip_range": [-1.0, 1.0],
            },
            "target_joint_pos": {
                **self.target_joint_pos_stats.as_dict(),
                "available": self.target_joint_pos_available,
            },
            "target_delta_abs": {
                **self.target_delta_stats.as_dict(),
                "available": self.target_delta_available,
            },
            "root_z": self.root_z_stats.as_dict(),
            "joint_pos": self.joint_pos_stats.as_dict(),
            "joint_vel": self.joint_vel_stats.as_dict(),
            "torque": {**self.torque_stats.as_dict(), "available": self.torque_available},
            "effort": {
                **self.effort_ratio_stats.as_dict(),
                "available": self.effort_ratio_available,
                "steps_above_threshold": self.effort_ratio_steps_above_threshold,
                "threshold": self.effort_ratio_threshold,
            },
            "joint_limit_margin_min": self.joint_limit_margin_min,
            "base_euler_deg": {
                "roll": self.roll_stats.as_dict(),
                "pitch": self.pitch_stats.as_dict(),
                "yaw": self.yaw_stats.as_dict(),
            },
            "terminations": {
                "base_height": self.base_height_count,
                "bad_orientation": self.bad_orientation_count,
                "time_out": self.time_out_count,
            },
            "resets": self.reset_count,
        }
