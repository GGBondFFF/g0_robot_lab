from __future__ import annotations

import pytest

from tests.helpers.isaaclab_runtime import get_robot_joint_names, resolve_action_joint_order


pytestmark = pytest.mark.isaaclab


def test_g0_runtime_smoke_headless(isaac_sim_app):
    import gymnasium as gym
    import torch

    import g0_robot_lab.tasks  # noqa: F401
    from g0_robot_lab.assets.robots.g0 import G0_JOINT_SDK_NAMES
    from g0_robot_lab.tasks.locomotion.robots.g0.velocity_env_cfg import G0RobotLabEnvCfg

    env_cfg = G0RobotLabEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.physics_material = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.reset_base = None
    env_cfg.events.reset_robot_joints = None
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "add_base_mass"):
        env_cfg.events.add_base_mass = None

    env = gym.make("G0-Velocity-v0", cfg=env_cfg)
    try:
        obs, _ = env.reset()
        assert env.action_space.shape[-1] == 22
        assert obs is not None
        if isinstance(obs, dict):
            for value in obs.values():
                if hasattr(value, "isfinite"):
                    assert torch.isfinite(value).all()

        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        robot_joint_names = get_robot_joint_names(robot)
        action_joint_order = resolve_action_joint_order(base_env.action_manager, robot_joint_names)
        assert action_joint_order == list(G0_JOINT_SDK_NAMES)
        assert base_env.action_manager.total_action_dim == 22

        action = torch.zeros((1, base_env.action_manager.total_action_dim), device=base_env.device)
        for _ in range(10):
            env.step(action)

        assert torch.isfinite(robot.data.root_pos_w).all()
        assert torch.isfinite(robot.data.root_quat_w).all()
        assert torch.isfinite(robot.data.joint_pos).all()
        assert torch.isfinite(robot.data.joint_vel).all()
    finally:
        env.close()
