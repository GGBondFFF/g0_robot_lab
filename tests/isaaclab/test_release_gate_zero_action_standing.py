from __future__ import annotations

import pytest


pytestmark = [pytest.mark.release_gate, pytest.mark.slow]


def test_zero_action_standing_release_gate_500_steps(isaac_sim_app):
    import gymnasium as gym
    import torch

    import g0_robot_lab.tasks  # noqa: F401
    from g0_robot_lab.tasks.locomotion.robots.g0.velocity_env_cfg import G0RobotLabEnvCfg

    env_cfg = G0RobotLabEnvCfg()

    # Fixed zero-action release-gate condition.
    env_cfg.scene.num_envs = 1
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.233)
    env_cfg.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    env_cfg.scene.robot.init_state.lin_vel = (0.0, 0.0, 0.0)
    env_cfg.scene.robot.init_state.ang_vel = (0.0, 0.0, 0.0)
    env_cfg.scene.robot.init_state.joint_vel = {".*": 0.0}

    # Disable randomization / disturbance events.
    env_cfg.events.physics_material = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.reset_base = None
    env_cfg.events.reset_robot_joints = None
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "add_base_mass"):
        env_cfg.events.add_base_mass = None

    # Keep the release gate deterministic and independent from curriculum updates.
    if hasattr(env_cfg.curriculum, "lin_vel_cmd_levels"):
        env_cfg.curriculum.lin_vel_cmd_levels = None
    if hasattr(env_cfg.curriculum, "ang_vel_cmd_levels"):
        env_cfg.curriculum.ang_vel_cmd_levels = None

    # Fixed standing command.
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

    # Remove reward-only contact diagnostics from this fixed standing gate.
    # Physical terminations remain active below.
    if hasattr(env_cfg.rewards, "undesired_contacts"):
        env_cfg.rewards.undesired_contacts = None

    env = gym.make("G0-Velocity-v0", cfg=env_cfg)
    try:
        env.reset()
        base_env = env.unwrapped
        action = torch.zeros((1, base_env.action_manager.total_action_dim), device=base_env.device)

        failure_report = None
        for step in range(500):
            out = env.step(action)

            if len(out) == 4:
                terminated = out[2]
                truncated = torch.zeros_like(terminated, dtype=torch.bool)
                info = out[3]
            else:
                terminated = out[2]
                truncated = out[3]
                info = out[4]

            done = torch.logical_or(terminated, truncated)
            if bool(done[0].item()):
                robot = base_env.scene["robot"]
                root_z = float(robot.data.root_pos_w[0, 2].item())

                term_report = {}
                termination_manager = getattr(base_env, "termination_manager", None)
                if termination_manager is not None:
                    for name in ("time_out", "base_height", "bad_orientation"):
                        try:
                            value = termination_manager.get_term(name)
                            if hasattr(value, "detach"):
                                value = value.detach()
                            if hasattr(value, "__getitem__"):
                                value = value[0]
                            if hasattr(value, "item"):
                                value = value.item()
                            term_report[name] = bool(value)
                        except Exception as exc:
                            term_report[name] = f"unavailable: {type(exc).__name__}: {exc}"

                failure_report = {
                    "step": step,
                    "terminated": bool(terminated[0].item()),
                    "truncated": bool(truncated[0].item()),
                    "done": bool(done[0].item()),
                    "root_z": root_z,
                    "termination_terms": term_report,
                    "info_keys": list(info.keys()) if isinstance(info, dict) else str(type(info)),
                }
                break

        assert failure_report is None, f"Zero-action standing failed: {failure_report}"
    finally:
        env.close()
