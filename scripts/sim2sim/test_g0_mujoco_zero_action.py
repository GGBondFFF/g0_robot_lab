import numpy as np
import pytest

from g0_mujoco_zero_action import DEFAULT_Q, MUJOCO_JOINT_ORDER, get_target_q


def test_zero_pose_target_is_all_zero():
    target_q = get_target_q("zero_pose")

    assert target_q.dtype == np.float64
    assert target_q.tolist() == [0.0] * len(MUJOCO_JOINT_ORDER)


def test_default_stand_target_uses_default_joint_pose():
    target_q = get_target_q("default_stand")

    assert target_q.tolist() == [DEFAULT_Q[name] for name in MUJOCO_JOINT_ORDER]


def test_unknown_target_pose_raises():
    with pytest.raises(ValueError):
        get_target_q("bad_pose")
