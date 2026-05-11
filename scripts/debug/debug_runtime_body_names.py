#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

LOG_PATH = Path("logs/debug_runtime_body_names.txt")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


parser = argparse.ArgumentParser(description="Debug G0 runtime body names and root state.")
parser.add_argument("--task", type=str, default="G0-Velocity-v0")
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    import gymnasium as gym
    import g0_robot_lab.tasks  # noqa: F401

    from g0_robot_lab.tasks.locomotion.robots.g0.velocity_env_cfg import G0RobotLabEnvCfg

    env_cfg = G0RobotLabEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    if getattr(args_cli, "device", None) is not None:
        env_cfg.sim.device = args_cli.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset()

    base_env = env.unwrapped
    robot = base_env.scene["robot"]

    log("")
    log("=" * 100)
    log("BODY NAMES")
    log("=" * 100)
    for i, name in enumerate(robot.data.body_names):
        log(f"[{i:02d}] {name}")

    log("")
    log("=" * 100)
    log("JOINT NAMES")
    log("=" * 100)
    for i, name in enumerate(robot.data.joint_names):
        log(f"[{i:02d}] {name}")

    log("")
    log("=" * 100)
    log("ROOT STATE")
    log("=" * 100)
    root_pos = robot.data.root_pos_w[0].detach().cpu().tolist()
    root_quat = robot.data.root_quat_w[0].detach().cpu().tolist()
    log(f"root_pos_w[0]  = {root_pos}")
    log(f"root_quat_w[0] = {root_quat}")

    log("")
    log("=" * 100)
    log("DEFAULT JOINT POS")
    log("=" * 100)
    default_joint_pos = robot.data.default_joint_pos[0].detach().cpu().tolist()
    for i, name in enumerate(robot.data.joint_names):
        log(f"[{i:02d}] {name:32s} default={default_joint_pos[i]: .6f}")

    log("")
    log("=" * 100)
    log("FOOT / ANKLE BODY CANDIDATES")
    log("=" * 100)
    keywords = ("foot", "toe", "ankle", "sole", "heel")
    for i, name in enumerate(robot.data.body_names):
        if any(k in name.lower() for k in keywords):
            log(f"[candidate] [{i:02d}] {name}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()