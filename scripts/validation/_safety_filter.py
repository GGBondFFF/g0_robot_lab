from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence


class SafetyFilterError(ValueError):
    """Raised when a dry-run command fails a safety precondition."""


@dataclass(frozen=True)
class SafetyLimits:
    pos_lower: Mapping[str, float]
    pos_upper: Mapping[str, float]
    vel_limit: Mapping[str, float]
    effort_limit: Mapping[str, float]
    max_target_delta: float
    max_action_delta: float
    max_command_age_s: float


@dataclass
class SafetyState:
    prev_target: dict[str, float] = field(default_factory=dict)
    prev_action: dict[str, float] | None = None
    last_obs_time: float | None = None


@dataclass(frozen=True)
class SafetyFilterResult:
    joint_order: tuple[str, ...]
    raw_action_by_joint: dict[str, float]
    clipped_action_by_joint: dict[str, float]
    target_by_joint: dict[str, float]
    dq_by_joint: dict[str, float]
    tau_ff_by_joint: dict[str, float]
    safe_mode: str
    diagnostics: tuple[str, ...] = ()


def _is_finite_scalar(value: float) -> bool:
    return math.isfinite(float(value))


def _require_finite_mapping(values: Mapping[str, float], label: str) -> None:
    for joint_name, value in values.items():
        if not _is_finite_scalar(value):
            raise SafetyFilterError(f"{label} is non-finite for joint {joint_name}: {value!r}")


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


class SafetyFilter:
    """Pure-Python safety filtering for LowCmd-style dry-run generation."""

    def apply(
        self,
        *,
        raw_action: Sequence[float],
        joint_order: Sequence[str],
        default_joint_pos: Mapping[str, float],
        action_scale: float,
        limits: SafetyLimits,
        state: SafetyState,
        now: float,
        emergency_stop: bool,
        hardware_enabled: bool,
    ) -> SafetyFilterResult:
        if hardware_enabled:
            raise SafetyFilterError("hardware_enabled=True is forbidden for dry-run mapping.")
        if not _is_finite_scalar(action_scale):
            raise SafetyFilterError(f"action_scale must be finite, got {action_scale!r}")

        joint_names = tuple(str(name) for name in joint_order)
        if len(raw_action) != len(joint_names):
            raise SafetyFilterError(f"Expected {len(joint_names)} actions, got {len(raw_action)}.")

        missing_defaults = [joint_name for joint_name in joint_names if joint_name not in default_joint_pos]
        if missing_defaults:
            raise SafetyFilterError(f"Missing default joint positions for: {missing_defaults}")

        diagnostics: list[str] = []
        dq_by_joint = {joint_name: 0.0 for joint_name in joint_names}
        tau_ff_by_joint = {
            joint_name: _clamp(0.0, -abs(float(limits.effort_limit[joint_name])), abs(float(limits.effort_limit[joint_name])))
            for joint_name in joint_names
        }

        if emergency_stop:
            target_by_joint: dict[str, float] = {}
            for joint_name in joint_names:
                hold_target = state.prev_target.get(joint_name, float(default_joint_pos[joint_name]))
                lower = float(limits.pos_lower[joint_name])
                upper = float(limits.pos_upper[joint_name])
                target_by_joint[joint_name] = _clamp(float(hold_target), lower, upper)
            raw_action_by_joint = {
                joint_name: float(raw_action[index]) if _is_finite_scalar(float(raw_action[index])) else 0.0
                for index, joint_name in enumerate(joint_names)
            }
            clipped_action_by_joint = {
                joint_name: _clamp(
                    float(state.prev_action.get(joint_name, 0.0)) if state.prev_action is not None else 0.0,
                    -1.0,
                    1.0,
                )
                for joint_name in joint_names
            }
            _require_finite_mapping(target_by_joint, "emergency-stop target")
            _require_finite_mapping(dq_by_joint, "emergency-stop dq")
            _require_finite_mapping(tau_ff_by_joint, "emergency-stop tau_ff")
            return SafetyFilterResult(
                joint_order=joint_names,
                raw_action_by_joint=raw_action_by_joint,
                clipped_action_by_joint=clipped_action_by_joint,
                target_by_joint=target_by_joint,
                dq_by_joint=dq_by_joint,
                tau_ff_by_joint=tau_ff_by_joint,
                safe_mode="emergency_stop_hold",
                diagnostics=("emergency_stop_hold",),
            )

        if state.last_obs_time is not None and (now - state.last_obs_time) > limits.max_command_age_s:
            raise SafetyFilterError(
                f"Stale observation: age={now - state.last_obs_time:.6f}s exceeds {limits.max_command_age_s:.6f}s"
            )

        raw_action_by_joint: dict[str, float] = {}
        for index, joint_name in enumerate(joint_names):
            action_value = float(raw_action[index])
            if not _is_finite_scalar(action_value):
                raise SafetyFilterError(f"raw_action is non-finite for joint {joint_name}: {action_value!r}")
            raw_action_by_joint[joint_name] = action_value

        clipped_action_by_joint = {
            joint_name: _clamp(raw_action_by_joint[joint_name], -1.0, 1.0) for joint_name in joint_names
        }

        if state.prev_action is not None:
            for joint_name in joint_names:
                prev_action = float(state.prev_action.get(joint_name, 0.0))
                bounded = _clamp(
                    clipped_action_by_joint[joint_name],
                    prev_action - limits.max_action_delta,
                    prev_action + limits.max_action_delta,
                )
                if bounded != clipped_action_by_joint[joint_name]:
                    diagnostics.append(f"action_delta_clamped:{joint_name}")
                clipped_action_by_joint[joint_name] = _clamp(bounded, -1.0, 1.0)

        target_by_joint = {
            joint_name: float(default_joint_pos[joint_name]) + action_scale * clipped_action_by_joint[joint_name]
            for joint_name in joint_names
        }

        for joint_name in joint_names:
            lower = float(limits.pos_lower[joint_name])
            upper = float(limits.pos_upper[joint_name])
            bounded = _clamp(target_by_joint[joint_name], lower, upper)
            if bounded != target_by_joint[joint_name]:
                diagnostics.append(f"position_clamped:{joint_name}")
            target_by_joint[joint_name] = bounded

        if state.prev_target:
            for joint_name in joint_names:
                if joint_name not in state.prev_target:
                    continue
                prev_target = float(state.prev_target[joint_name])
                bounded = _clamp(
                    target_by_joint[joint_name],
                    prev_target - limits.max_target_delta,
                    prev_target + limits.max_target_delta,
                )
                if bounded != target_by_joint[joint_name]:
                    diagnostics.append(f"target_delta_clamped:{joint_name}")
                lower = float(limits.pos_lower[joint_name])
                upper = float(limits.pos_upper[joint_name])
                target_by_joint[joint_name] = _clamp(bounded, lower, upper)

        _require_finite_mapping(target_by_joint, "target")
        _require_finite_mapping(dq_by_joint, "dq")
        _require_finite_mapping(tau_ff_by_joint, "tau_ff")

        return SafetyFilterResult(
            joint_order=joint_names,
            raw_action_by_joint=raw_action_by_joint,
            clipped_action_by_joint=clipped_action_by_joint,
            target_by_joint=target_by_joint,
            dq_by_joint=dq_by_joint,
            tau_ff_by_joint=tau_ff_by_joint,
            safe_mode="normal",
            diagnostics=tuple(diagnostics),
        )
