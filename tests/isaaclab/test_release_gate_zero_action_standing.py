from __future__ import annotations

import pytest


pytestmark = [pytest.mark.release_gate, pytest.mark.slow]


def test_zero_action_standing_release_gate_500_steps(isaac_sim_app):
    import gymnasium as gym
    import torch

    import g0_robot_lab.tasks  # noqa: F401
    from g0_robot_lab.tasks.locomotion.robots.g0.velocity_env_cfg import G0RobotLabEnvCfg

    env_cfg = G0RobotLabEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.events.physics_material = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.reset_base = None
    env_cfg.events.reset_robot_joints = None
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "add_base_mass"):
        env_cfg.events.add_base_mass = None
    env_cfg.commands.base_velocity.rel_standing_envs = 1.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.observations.policy.enable_corruption = False

    env = gym.make("G0-Velocity-v0", cfg=env_cfg)
    try:
        env.reset()
        base_env = env.unwrapped
        action = torch.zeros((1, base_env.action_manager.total_action_dim), device=base_env.device)
        first_done_step = None
        for step in range(500):
            out = env.step(action)
            done = out[2] if len(out) == 4 else torch.logical_or(out[2], out[3])
            if bool(done[0].item()):
                first_done_step = step
                break
        assert first_done_step is None, f"Zero-action standing terminated at step {first_done_step}."
    finally:
        env.close()
