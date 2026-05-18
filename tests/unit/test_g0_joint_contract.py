from __future__ import annotations

import pytest

from tests.conftest import G0_SOURCE
from tests.helpers.static_contract import literal_assignment, load_module_ast


pytestmark = pytest.mark.unit


EXPECTED_SDK_ORDER = [
    "l_hip_pitch_joint",
    "l_hip_roll_joint",
    "l_hip_yaw_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_hip_pitch_joint",
    "r_hip_roll_joint",
    "r_hip_yaw_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "l_shoulder_pitch_joint",
    "l_shoulder_roll_joint",
    "l_shoulder_yaw_joint",
    "l_elbow_pitch_joint",
    "r_shoulder_pitch_joint",
    "r_shoulder_roll_joint",
    "r_shoulder_yaw_joint",
    "r_elbow_pitch_joint",
]


EXPECTED_RIGHT_ANGLE = [
    "l_elbow_pitch_joint",
    "r_elbow_pitch_joint",
    "l_knee_pitch_joint",
    "r_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "r_ankle_pitch_joint",
]


def _g0_module():
    return load_module_ast(G0_SOURCE)


def test_g0_sdk_joint_order_is_deployment_contract():
    sdk_names = literal_assignment(_g0_module(), "G0_JOINT_SDK_NAMES")
    assert sdk_names == EXPECTED_SDK_ORDER
    assert len(sdk_names) == 22
    assert len(set(sdk_names)) == 22


def test_g0_default_pose_covers_sdk_joints_exactly():
    module = _g0_module()
    sdk_names = literal_assignment(module, "G0_JOINT_SDK_NAMES")
    default_pose = literal_assignment(module, "G0_DEFAULT_JOINT_POS")
    assert set(default_pose) == set(sdk_names)
    assert len(default_pose) == 22


def test_g0_default_pose_keeps_mirrored_pitch_signs():
    default_pose = literal_assignment(_g0_module(), "G0_DEFAULT_JOINT_POS")
    assert default_pose["l_hip_pitch_joint"] == -0.20
    assert default_pose["r_hip_pitch_joint"] == 0.20
    assert default_pose["l_knee_pitch_joint"] == -0.34
    assert default_pose["r_knee_pitch_joint"] == 0.34
    assert default_pose["l_ankle_pitch_joint"] == 0.14
    assert default_pose["r_ankle_pitch_joint"] == -0.14
    assert default_pose["l_elbow_pitch_joint"] == 0.97
    assert default_pose["r_elbow_pitch_joint"] == -0.97


def test_servo_groups_partition_sdk_joint_set():
    module = _g0_module()
    sdk_names = literal_assignment(module, "G0_JOINT_SDK_NAMES")
    right_angle = literal_assignment(module, "G0_RIGHT_ANGLE_SERVO_JOINT_NAMES")
    assert right_angle == EXPECTED_RIGHT_ANGLE

    standard = [name for name in sdk_names if name not in right_angle]
    assert len(standard) == 16
    assert len(right_angle) == 6
    assert set(standard).isdisjoint(right_angle)
    assert set(standard) | set(right_angle) == set(sdk_names)
