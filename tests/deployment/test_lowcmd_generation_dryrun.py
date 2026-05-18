from __future__ import annotations

import pytest

from tests.conftest import G0_SOURCE
from tests.helpers.deployment_dryrun import FakeLowCmdTransport, build_fake_lowcmd
from tests.helpers.static_contract import literal_assignment, load_module_ast


pytestmark = [pytest.mark.deployment_dryrun, pytest.mark.hardware_forbidden]


def _contract_values():
    module = load_module_ast(G0_SOURCE)
    return literal_assignment(module, "G0_JOINT_SDK_NAMES"), literal_assignment(module, "G0_DEFAULT_JOINT_POS")


def test_lowcmd_generation_zero_action_matches_default_pose(dryrun_required):
    joint_order, default_pose = _contract_values()
    command = build_fake_lowcmd([0.0] * 22, default_pose, joint_order, action_scale=0.12)
    assert command.dry_run is True
    assert command.control_dt == pytest.approx(0.02)
    assert [motor.joint_name for motor in command.motors] == joint_order
    for motor in command.motors:
        assert motor.target_position == pytest.approx(default_pose[motor.joint_name])
        assert motor.torque_ff == 0.0


def test_lowcmd_generation_one_hot_action_changes_matching_joint_only(dryrun_required):
    joint_order, default_pose = _contract_values()
    for index, joint_name in enumerate(joint_order):
        action = [0.0] * 22
        action[index] = 1.0
        command = build_fake_lowcmd(action, default_pose, joint_order, action_scale=0.12)
        for motor in command.motors:
            expected = default_pose[motor.joint_name] + (0.12 if motor.joint_name == joint_name else 0.0)
            assert motor.target_position == pytest.approx(expected)
            assert motor.source_action_index == joint_order.index(motor.joint_name)


def test_lowcmd_generation_clips_policy_action_for_dryrun(dryrun_required):
    joint_order, default_pose = _contract_values()
    action = [2.5] + [0.0] * 21
    command = build_fake_lowcmd(action, default_pose, joint_order, action_scale=0.12)
    assert command.motors[0].source_action_value == 2.5
    assert command.motors[0].target_position == pytest.approx(default_pose[joint_order[0]] + 0.12)


def test_fake_transport_captures_only_dryrun_commands(dryrun_required):
    joint_order, default_pose = _contract_values()
    command = build_fake_lowcmd([0.0] * 22, default_pose, joint_order, action_scale=0.12)
    transport = FakeLowCmdTransport()
    transport.send(command)
    assert transport.sent == [command]
