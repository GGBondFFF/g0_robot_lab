from __future__ import annotations

from scripts.sim2sim import g0_sim2sim_config as cfg


def test_joint_order_is_sdk_policy_order() -> None:
    assert cfg.get_joint_names() == [
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


def test_joint_order_has_no_duplicates() -> None:
    names = cfg.get_joint_names()
    assert len(names) == len(set(names))
