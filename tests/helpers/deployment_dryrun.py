from __future__ import annotations

import math
from typing import Mapping, Sequence

from scripts.validation._lowcmd_mapping import (
    FakeLowCmd,
    FakeMotorCommand,
    build_default_safety_limits,
    map_policy_to_lowcmd_dry_run,
)
from scripts.validation._safety_filter import SafetyLimits, SafetyState


class FakeLowCmdTransport:
    def __init__(self) -> None:
        self.sent: list[FakeLowCmd] = []

    def send(self, command: FakeLowCmd) -> None:
        if not command.dry_run:
            raise AssertionError("FakeLowCmdTransport only accepts dry-run commands.")
        self.sent.append(command)


def build_fake_lowcmd(
    policy_action: Sequence[float],
    default_joint_pos: Mapping[str, float],
    joint_order: Sequence[str],
    *,
    action_scale: float,
    kp_by_joint: Mapping[str, float] | None = None,
    kd_by_joint: Mapping[str, float] | None = None,
    limits: SafetyLimits | None = None,
    state: SafetyState | None = None,
    control_dt: float = 0.02,
    dry_run: bool = True,
) -> FakeLowCmd:
    if not dry_run:
        raise AssertionError("Tests must not build non-dry-run LowCmd objects.")
    joint_names = tuple(str(name) for name in joint_order)
    if limits is None:
        baseline_limits = build_default_safety_limits(joint_names)
        resolved_limits = SafetyLimits(
            pos_lower={joint_name: -math.inf for joint_name in joint_names},
            pos_upper={joint_name: math.inf for joint_name in joint_names},
            vel_limit=dict(baseline_limits.vel_limit),
            effort_limit=dict(baseline_limits.effort_limit),
            max_target_delta=math.inf,
            max_action_delta=math.inf,
            max_command_age_s=math.inf,
        )
    else:
        resolved_limits = limits
    resolved_state = state or SafetyState()
    return map_policy_to_lowcmd_dry_run(
        policy_action,
        joint_order=joint_names,
        default_joint_pos=default_joint_pos,
        action_scale=action_scale,
        limits=resolved_limits,
        state=resolved_state,
        kp_by_joint=kp_by_joint,
        kd_by_joint=kd_by_joint,
        control_dt=control_dt,
        now=resolved_state.last_obs_time if resolved_state.last_obs_time is not None else 0.0,
        emergency_stop=False,
        hardware_enabled=False,
    )
