from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class FakeMotorCommand:
    joint_name: str
    target_position: float
    kp: float
    kd: float
    torque_ff: float
    source_action_index: int
    source_action_value: float


@dataclass(frozen=True)
class FakeLowCmd:
    dry_run: bool
    control_dt: float
    motors: tuple[FakeMotorCommand, ...]


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
    control_dt: float = 0.02,
    dry_run: bool = True,
) -> FakeLowCmd:
    if not dry_run:
        raise AssertionError("Tests must not build non-dry-run LowCmd objects.")
    if len(policy_action) != len(joint_order):
        raise ValueError(f"Expected {len(joint_order)} actions, got {len(policy_action)}.")

    kp_by_joint = kp_by_joint or {}
    kd_by_joint = kd_by_joint or {}
    motors = []
    for index, joint_name in enumerate(joint_order):
        action_value = float(policy_action[index])
        clipped = max(-1.0, min(1.0, action_value))
        motors.append(
            FakeMotorCommand(
                joint_name=joint_name,
                target_position=float(default_joint_pos[joint_name]) + action_scale * clipped,
                kp=float(kp_by_joint.get(joint_name, 0.0)),
                kd=float(kd_by_joint.get(joint_name, 0.0)),
                torque_ff=0.0,
                source_action_index=index,
                source_action_value=action_value,
            )
        )
    return FakeLowCmd(dry_run=True, control_dt=control_dt, motors=tuple(motors))
