from __future__ import annotations

import math

import pytest

from scripts.validation._lowcmd_mapping import build_default_safety_limits, map_policy_to_lowcmd_dry_run
from scripts.validation._safety_filter import SafetyFilterError, SafetyState
from tests.conftest import G0_SOURCE
from tests.helpers.deployment_dryrun import FakeLowCmdTransport, build_fake_lowcmd
from tests.helpers.static_contract import literal_assignment, load_module_ast


pytestmark = [pytest.mark.deployment_dryrun, pytest.mark.hardware_forbidden]


def _contract_values():
    module = load_module_ast(G0_SOURCE)
    joint_order = literal_assignment(module, "G0_JOINT_SDK_NAMES")
    default_pose = literal_assignment(module, "G0_DEFAULT_JOINT_POS")
    return joint_order, default_pose


def _limits(joint_order):
    return build_default_safety_limits(joint_order)


def test_lowcmd_mapping_builds_22_finite_dry_run_commands(dryrun_required):
    joint_order, default_pose = _contract_values()
    command = map_policy_to_lowcmd_dry_run(
        [0.0] * 22,
        joint_order=joint_order,
        default_joint_pos=default_pose,
        action_scale=0.12,
        limits=_limits(joint_order),
        state=SafetyState(last_obs_time=1.0),
        now=1.0,
    )
    assert command.dry_run is True
    assert len(command.motors) == 22
    assert [motor.motor_id for motor in command.motors] == list(range(22))
    assert [motor.joint_name for motor in command.motors] == joint_order
    for motor in command.motors:
        assert math.isfinite(motor.q)
        assert math.isfinite(motor.dq)
        assert math.isfinite(motor.kp)
        assert math.isfinite(motor.kd)
        assert math.isfinite(motor.tau_ff)
        assert math.isfinite(motor.effort_limit)
        assert math.isfinite(motor.source_raw_action)
        assert math.isfinite(motor.source_clipped_action)
        assert motor.q == pytest.approx(default_pose[motor.joint_name])
        assert motor.dq == 0.0
        assert motor.kp == 0.0
        assert motor.kd == 0.0
        assert motor.tau_ff == 0.0


def test_plus_and_minus_one_map_to_default_plus_or_minus_scale(dryrun_required):
    joint_order, default_pose = _contract_values()
    plus = map_policy_to_lowcmd_dry_run(
        [1.0] + [0.0] * 21,
        joint_order=joint_order,
        default_joint_pos=default_pose,
        action_scale=0.12,
        limits=_limits(joint_order),
        state=SafetyState(last_obs_time=1.0),
        now=1.0,
    )
    minus = map_policy_to_lowcmd_dry_run(
        [-1.0] + [0.0] * 21,
        joint_order=joint_order,
        default_joint_pos=default_pose,
        action_scale=0.12,
        limits=_limits(joint_order),
        state=SafetyState(last_obs_time=1.0),
        now=1.0,
    )
    first_joint = joint_order[0]
    assert plus.motors[0].q == pytest.approx(default_pose[first_joint] + 0.12)
    assert minus.motors[0].q == pytest.approx(default_pose[first_joint] - 0.12)


def test_raw_overflow_action_is_not_used_directly(dryrun_required):
    joint_order, default_pose = _contract_values()
    command = map_policy_to_lowcmd_dry_run(
        [10.0] + [0.0] * 21,
        joint_order=joint_order,
        default_joint_pos=default_pose,
        action_scale=0.12,
        limits=_limits(joint_order),
        state=SafetyState(last_obs_time=1.0),
        now=1.0,
    )
    first_joint = joint_order[0]
    assert command.motors[0].source_raw_action == 10.0
    assert command.motors[0].source_clipped_action == 1.0
    assert command.motors[0].q == pytest.approx(default_pose[first_joint] + 0.12)
    assert command.motors[0].q != pytest.approx(default_pose[first_joint] + 1.2)


def test_mapping_rejects_nonfinite_and_hardware_enabled(dryrun_required):
    joint_order, default_pose = _contract_values()
    with pytest.raises(SafetyFilterError, match="non-finite"):
        map_policy_to_lowcmd_dry_run(
            [math.nan] + [0.0] * 21,
            joint_order=joint_order,
            default_joint_pos=default_pose,
            limits=_limits(joint_order),
            state=SafetyState(last_obs_time=1.0),
            now=1.0,
        )
    with pytest.raises(SafetyFilterError, match="hardware_enabled=True"):
        map_policy_to_lowcmd_dry_run(
            [0.0] * 22,
            joint_order=joint_order,
            default_joint_pos=default_pose,
            limits=_limits(joint_order),
            state=SafetyState(last_obs_time=1.0),
            now=1.0,
            hardware_enabled=True,
        )


def test_mapping_emergency_stop_produces_safe_hold_command(dryrun_required):
    joint_order, default_pose = _contract_values()
    limits = _limits(joint_order)
    prev_target = {joint_name: float(default_pose[joint_name]) + 0.02 for joint_name in joint_order}
    command = map_policy_to_lowcmd_dry_run(
        [1.0] * 22,
        joint_order=joint_order,
        default_joint_pos=default_pose,
        limits=limits,
        state=SafetyState(
            prev_action={joint_name: 0.5 for joint_name in joint_order},
            prev_target=prev_target,
            last_obs_time=0.0,
        ),
        now=100.0,
        emergency_stop=True,
    )
    for motor in command.motors:
        expected = max(limits.pos_lower[motor.joint_name], min(limits.pos_upper[motor.joint_name], prev_target[motor.joint_name]))
        assert motor.q == pytest.approx(expected)


def test_build_fake_lowcmd_wrapper_remains_compatible(dryrun_required):
    joint_order, default_pose = _contract_values()
    command = build_fake_lowcmd([2.5] + [0.0] * 21, default_pose, joint_order, action_scale=0.12)
    assert command.dry_run is True
    assert command.motors[0].motor_id == 0
    assert command.motors[0].source_action_value == 2.5
    assert command.motors[0].target_position == pytest.approx(default_pose[joint_order[0]] + 0.12)
    transport = FakeLowCmdTransport()
    transport.send(command)
    assert transport.sent == [command]
