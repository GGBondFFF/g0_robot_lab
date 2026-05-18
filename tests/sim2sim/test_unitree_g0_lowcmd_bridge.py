from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scripts.sim2sim import g0_sim2sim_config as cfg
from scripts.unitree_mujoco_g0.g0_lowcmd_bridge import G0UnitreeMujocoBridge, import_mujoco
from scripts.unitree_mujoco_g0.generate_g0_unitree_mjcf import generate


def test_unitree_style_generated_model_loads_and_matches_policy_order(tmp_path: Path) -> None:
    try:
        import_mujoco()
    except ModuleNotFoundError:
        pytest.skip("mujoco is not installed")

    model_path, scene_path = generate(Path("mujoco/g0.xml"), tmp_path)
    assert model_path.exists()
    assert scene_path.exists()

    bridge = G0UnitreeMujocoBridge(model_path)
    assert bridge.model.nu == cfg.get_action_dim()
    assert list(bridge.joint_indices) == cfg.get_joint_names()
    np.testing.assert_allclose(bridge.get_joint_pos(), cfg.get_default_joint_pos_array())


def test_lowcmd_fields_follow_isaac_action_bridge(tmp_path: Path) -> None:
    try:
        import_mujoco()
    except ModuleNotFoundError:
        pytest.skip("mujoco is not installed")

    model_path, _ = generate(Path("mujoco/g0.xml"), tmp_path)
    bridge = G0UnitreeMujocoBridge(model_path)
    action = np.ones(cfg.get_action_dim())
    _processed_action, target = bridge.policy_action_to_target(action)
    lowcmd = bridge.build_lowcmd(target)

    assert len(lowcmd.motor_cmd) == cfg.get_action_dim()
    for index, motor_cmd in enumerate(lowcmd.motor_cmd):
        assert np.isclose(motor_cmd.q, target[index])
        assert motor_cmd.dq == 0.0
        assert motor_cmd.tau == 0.0
        assert np.isclose(motor_cmd.kp, bridge.kp[index])
        assert np.isclose(motor_cmd.kd, bridge.kd[index])
