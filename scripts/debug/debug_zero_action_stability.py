#!/usr/bin/env python3
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

import torch

from isaaclab.app import AppLauncher

LOG_PATH = Path("logs/debug_zero_action_stability.txt")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.write_text("", encoding="utf-8")


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()


parser = argparse.ArgumentParser(description="Run zero-action stability test for G0.")
parser.add_argument("--task", type=str, default="G0-Velocity-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=500)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

log("[ZERO-DBG] before AppLauncher")
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
log("[ZERO-DBG] after AppLauncher")


def _reset_env(env):
    """Handle different Isaac Lab / Gym reset return styles."""
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def _step_env(env, action):
    """Handle different Isaac Lab / Gym step return styles."""
    out = env.step(action)

    if not isinstance(out, tuple):
        raise RuntimeError(f"env.step(action) returned non-tuple: {type(out)}")

    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = torch.logical_or(terminated, truncated)
        return obs, reward, done, terminated, truncated, info

    if len(out) == 4:
        obs, reward, done, info = out
        terminated = done
        truncated = torch.zeros_like(done, dtype=torch.bool)
        return obs, reward, done, terminated, truncated, info

    raise RuntimeError(f"Unsupported env.step(action) return length: {len(out)}")


def main() -> None:
    log("[ZERO-DBG] enter main")

    import gymnasium as gym
    import g0_robot_lab.tasks  # noqa: F401

    from g0_robot_lab.tasks.locomotion.robots.g0.velocity_env_cfg import G0RobotLabEnvCfg

    log("[ZERO-DBG] before env_cfg")
    env_cfg = G0RobotLabEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    if getattr(args_cli, "device", None) is not None:
        env_cfg.sim.device = args_cli.device

    log("[ZERO-DBG] before gym.make")
    env = gym.make(args_cli.task, cfg=env_cfg)
    log("[ZERO-DBG] after gym.make")

    try:
        log("[ZERO-DBG] before env.reset")
        obs = _reset_env(env)
        log("[ZERO-DBG] after env.reset")

        base_env = env.unwrapped
        robot = base_env.scene["robot"]

        log("")
        log("=" * 100)
        log("ZERO ACTION STABILITY TEST")
        log("=" * 100)
        action_dim = base_env.action_manager.total_action_dim

        log(f"num_envs    = {base_env.num_envs}")
        log(f"num_actions = {action_dim}")
        log(f"steps       = {args_cli.steps}")
        log(f"device      = {base_env.device}")

        zero_action = torch.zeros(
            (base_env.num_envs, action_dim),
            device=base_env.device,
            dtype=torch.float32,
        )

        log("[ZERO-DBG] before first env.step")

        for step in range(args_cli.steps):
            obs, reward, done, terminated, truncated, info = _step_env(env, zero_action)

            root_pos = robot.data.root_pos_w[0].detach().cpu().tolist()
            root_quat = robot.data.root_quat_w[0].detach().cpu().tolist()

            if step % 25 == 0 or bool(torch.any(done)):
                log(
                    f"step={step:04d} "
                    f"root_z={root_pos[2]: .4f} "
                    f"root_pos={root_pos} "
                    f"root_quat={root_quat} "
                    f"reward0={reward[0].item(): .4f} "
                    f"done0={bool(done[0].item())} "
                    f"terminated0={bool(terminated[0].item())} "
                    f"truncated0={bool(truncated[0].item())}"
                )

            if bool(torch.any(done)):
                log("[STOP] Environment done during zero-action test.")
                break

        log("[ZERO-DBG] finished loop")

    finally:
        log("[ZERO-DBG] before env.close")
        env.close()
        log("[ZERO-DBG] after env.close")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("")
        log("=" * 100)
        log("[ZERO-DBG] ERROR")
        log("=" * 100)
        log(traceback.format_exc())
    finally:
        log("[ZERO-DBG] before simulation_app.close")
        simulation_app.close()
        log("[ZERO-DBG] after simulation_app.close")