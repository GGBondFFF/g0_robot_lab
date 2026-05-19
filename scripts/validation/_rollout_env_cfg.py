from __future__ import annotations

from typing import Any


def apply_fixed_policy_rollout_env_cfg(env_cfg: Any, *, num_envs: int = 1, root_z: float = 0.233) -> Any:
    env_cfg.scene.num_envs = num_envs
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, root_z)
    env_cfg.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    env_cfg.scene.robot.init_state.lin_vel = (0.0, 0.0, 0.0)
    env_cfg.scene.robot.init_state.ang_vel = (0.0, 0.0, 0.0)
    env_cfg.scene.robot.init_state.joint_vel = {".*": 0.0}

    env_cfg.events.physics_material = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.reset_base = None
    env_cfg.events.reset_robot_joints = None
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "add_base_mass"):
        env_cfg.events.add_base_mass = None

    if hasattr(env_cfg.curriculum, "lin_vel_cmd_levels"):
        env_cfg.curriculum.lin_vel_cmd_levels = None
    if hasattr(env_cfg.curriculum, "ang_vel_cmd_levels"):
        env_cfg.curriculum.ang_vel_cmd_levels = None

    env_cfg.commands.base_velocity.rel_standing_envs = 1.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    if hasattr(env_cfg.commands.base_velocity, "limit_ranges"):
        env_cfg.commands.base_velocity.limit_ranges.lin_vel_x = (0.0, 0.0)
        env_cfg.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.limit_ranges.ang_vel_z = (0.0, 0.0)

    env_cfg.observations.policy.enable_corruption = False

    if hasattr(env_cfg.rewards, "undesired_contacts"):
        env_cfg.rewards.undesired_contacts = None

    return env_cfg
