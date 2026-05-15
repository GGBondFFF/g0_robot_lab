from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.sim2sim import g0_sim2sim_config as cfg
from scripts.sim2sim.g0_mujoco_interface import G0MuJoCoInterface, import_mujoco


def test_observation_frame_shape_without_mujoco() -> None:
    frame = G0MuJoCoInterface.build_observation_frame(
        joint_pos=cfg.get_default_joint_pos_array(),
        joint_vel=np.zeros(cfg.get_action_dim()),
        last_action=np.zeros(cfg.get_action_dim()),
        command=np.zeros(3),
        sim_time=0.0,
    )
    assert frame.shape == (cfg.get_single_frame_observation_dim(),)


def test_expected_observation_dim_without_mujoco() -> None:
    assert G0MuJoCoInterface.expected_observation_dim() == cfg.get_policy_observation_dim()


def test_projected_gravity_for_identity_quaternion() -> None:
    rot = G0MuJoCoInterface.quat_wxyz_to_matrix(np.asarray([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(rot.T @ np.asarray([0.0, 0.0, -1.0]), np.asarray([0.0, 0.0, -1.0]))


def test_mujoco_model_has_all_g0_joints_if_available() -> None:
    try:
        import_mujoco()
    except ModuleNotFoundError:
        pytest.skip("mujoco is not installed")

    model_path = Path("mujoco/g0.xml")
    if not model_path.exists():
        pytest.skip("mujoco/g0.xml does not exist")

    interface = G0MuJoCoInterface(model_path)
    assert list(interface.joint_indices.keys()) == cfg.get_joint_names()
    assert interface.get_joint_pos().shape == (cfg.get_action_dim(),)
    assert interface.get_joint_vel().shape == (cfg.get_action_dim(),)
