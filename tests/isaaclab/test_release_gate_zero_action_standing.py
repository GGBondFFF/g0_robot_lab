from __future__ import annotations

import pytest


pytestmark = [pytest.mark.release_gate, pytest.mark.slow]


def test_zero_action_standing_release_gate_500_steps(isaac_sim_app):
    import gymnasium as gym
    import torch

    print("[ZERO-GATE] test entered", flush=True)

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

    # Match fixed standing-command condition.
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

    # Keep the physical terminations intact. This is only to avoid reward-side
    # contact diagnostics affecting the fixed zero-action release-gate readout.
    if hasattr(env_cfg.rewards, "undesired_contacts"):
        env_cfg.rewards.undesired_contacts = None

    env = None
    try:
        print("[ZERO-GATE] before gym.make", flush=True)
        env = gym.make("G0-Velocity-v0", cfg=env_cfg)
        print("[ZERO-GATE] after gym.make", flush=True)

        print("[ZERO-GATE] before env.reset", flush=True)
        env.reset()
        print("[ZERO-GATE] after env.reset", flush=True)

        base_env = env.unwrapped
        action_dim = base_env.action_manager.total_action_dim
        action = torch.zeros((1, action_dim), device=base_env.device)

        print(f"[ZERO-GATE] action_dim={action_dim}", flush=True)
        print("[ZERO-GATE] before 500-step loop", flush=True)

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

            robot = base_env.scene["robot"]
            root_z = float(robot.data.root_pos_w[0, 2].item())

            if step in (0, 1, 2, 10, 50, 100, 200, 300, 400, 499):
                print(
                    "[ZERO-GATE] "
                    f"step={step} "
                    f"terminated={bool(terminated[0].item())} "
                    f"truncated={bool(truncated[0].item())} "
                    f"done={bool(done[0].item())} "
                    f"root_z={root_z:.6f}",
                    flush=True,
                )

            if bool(done[0].item()):
                term_report = {}
                tm = getattr(base_env, "termination_manager", None)
                if tm is not None:
                    for name in ("time_out", "base_height", "bad_orientation"):
                        try:
                            value = tm.get_term(name)
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
                    "info_log": info.get("log", None) if isinstance(info, dict) else None,
                }
                print(f"[ZERO-GATE] FAILURE_REPORT={failure_report}", flush=True)
                break

        print("[ZERO-GATE] loop finished", flush=True)

        assert failure_report is None, f"Zero-action standing failed: {failure_report}"

        print("[ZERO-GATE] assertion passed", flush=True)

    finally:
        print("[ZERO-GATE] entering finally", flush=True)
        if env is not None:
            print("[ZERO-GATE] before env.close", flush=True)
            env.close()
            print("[ZERO-GATE] after env.close", flush=True)
        print("[ZERO-GATE] finally done", flush=True)
