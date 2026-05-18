from __future__ import annotations

from pathlib import Path

import pytest

from scripts.sim2sim import g0_sim2sim_config as cfg
from scripts.sim2sim.g0_mujoco_interface import G0MuJoCoInterface, import_mujoco


def test_mujoco_model_has_all_policy_joints_and_actuators() -> None:
    try:
        import_mujoco()
    except ModuleNotFoundError:
        pytest.skip("mujoco is not installed")

    model_path = Path("mujoco/g0.xml")
    interface = G0MuJoCoInterface(model_path)
    assert list(interface.joint_indices) == cfg.get_joint_names()
    assert all(index.qpos >= 0 and index.qvel >= 0 and index.actuator >= 0 for index in interface.joint_indices.values())
