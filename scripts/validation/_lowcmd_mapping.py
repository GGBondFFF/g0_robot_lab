from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from ._joint_contract_tables import load_joint_contract_tables
from ._safety_filter import SafetyFilter, SafetyFilterError, SafetyLimits, SafetyState


@dataclass(frozen=True)
class FakeMotorCommand:
    motor_id: int
    joint_name: str
    q: float
    dq: float
    kp: float
    kd: float
    tau_ff: float
    effort_limit: float
    source_action_index: int
    source_raw_action: float
    source_clipped_action: float

    @property
    def target_position(self) -> float:
        return self.q

    @property
    def torque_ff(self) -> float:
        return self.tau_ff

    @property
    def source_action_value(self) -> float:
        return self.source_raw_action


@dataclass(frozen=True)
class FakeLowCmd:
    dry_run: bool
    control_dt: float
    motors: tuple[FakeMotorCommand, ...]


def build_default_safety_limits(
    joint_order: Sequence[str],
    *,
    max_target_delta: float = 0.25,
    max_action_delta: float = 1.0,
    max_command_age_s: float = 0.1,
) -> SafetyLimits:
    tables = load_joint_contract_tables(joint_order)
    return SafetyLimits(
        pos_lower=dict(tables.pos_lower),
        pos_upper=dict(tables.pos_upper),
        vel_limit=dict(tables.vel_limit),
        effort_limit=dict(tables.effort_limit),
        max_target_delta=float(max_target_delta),
        max_action_delta=float(max_action_delta),
        max_command_age_s=float(max_command_age_s),
    )


def _ensure_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise SafetyFilterError(f"{name} must be finite, got {value!r}")
    return value


def map_policy_to_lowcmd_dry_run(
    raw_action: Sequence[float],
    *,
    joint_order: Sequence[str],
    default_joint_pos: Mapping[str, float],
    action_scale: float = 0.12,
    safety: SafetyFilter | None = None,
    limits: SafetyLimits | None = None,
    state: SafetyState | None = None,
    kp_by_joint: Mapping[str, float] | None = None,
    kd_by_joint: Mapping[str, float] | None = None,
    control_dt: float = 0.02,
    now: float | None = None,
    emergency_stop: bool = False,
    hardware_enabled: bool = False,
) -> FakeLowCmd:
    if hardware_enabled:
        raise SafetyFilterError("hardware_enabled=True is forbidden for dry-run mapping.")
    if control_dt <= 0.0:
        raise SafetyFilterError(f"control_dt must be positive, got {control_dt!r}")

    joint_names = tuple(str(name) for name in joint_order)
    limits = limits or build_default_safety_limits(joint_names)
    safety = safety or SafetyFilter()
    state = state or SafetyState()
    kp_by_joint = kp_by_joint or {}
    kd_by_joint = kd_by_joint or {}
    now = float(now if now is not None else (state.last_obs_time if state.last_obs_time is not None else 0.0))

    filtered = safety.apply(
        raw_action=raw_action,
        joint_order=joint_names,
        default_joint_pos=default_joint_pos,
        action_scale=float(action_scale),
        limits=limits,
        state=state,
        now=now,
        emergency_stop=emergency_stop,
        hardware_enabled=False,
    )

    motors: list[FakeMotorCommand] = []
    for index, joint_name in enumerate(joint_names):
        q = _ensure_finite(f"{joint_name}.q", filtered.target_by_joint[joint_name])
        dq = _ensure_finite(f"{joint_name}.dq", filtered.dq_by_joint[joint_name])
        kp = _ensure_finite(f"{joint_name}.kp", kp_by_joint.get(joint_name, 0.0))
        kd = _ensure_finite(f"{joint_name}.kd", kd_by_joint.get(joint_name, 0.0))
        effort_limit = abs(_ensure_finite(f"{joint_name}.effort_limit", limits.effort_limit[joint_name]))
        tau_ff = _ensure_finite(
            f"{joint_name}.tau_ff",
            max(-effort_limit, min(effort_limit, filtered.tau_ff_by_joint[joint_name])),
        )
        source_raw_action = _ensure_finite(
            f"{joint_name}.source_raw_action", filtered.raw_action_by_joint[joint_name]
        )
        source_clipped_action = _ensure_finite(
            f"{joint_name}.source_clipped_action", filtered.clipped_action_by_joint[joint_name]
        )
        motors.append(
            FakeMotorCommand(
                motor_id=index,
                joint_name=joint_name,
                q=q,
                dq=dq,
                kp=kp,
                kd=kd,
                tau_ff=tau_ff,
                effort_limit=effort_limit,
                source_action_index=index,
                source_raw_action=source_raw_action,
                source_clipped_action=source_clipped_action,
            )
        )

    if len(motors) != len(joint_names):
        raise SafetyFilterError(f"Expected {len(joint_names)} motors, got {len(motors)}.")

    return FakeLowCmd(dry_run=True, control_dt=float(control_dt), motors=tuple(motors))
