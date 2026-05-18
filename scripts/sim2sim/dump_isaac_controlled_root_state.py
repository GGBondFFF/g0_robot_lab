#!/usr/bin/env python3
"""Dump Isaac Lab root-frame observations under controlled root states."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="G0-Velocity-v0", help="Isaac Lab task id.")
    parser.add_argument("--output", default="logs/sim2sim/root_frame/isaac_controlled_root_state.npz", help="Output .npz path.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless.")
    return parser.parse_args()


def tensor_to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim > 1:
        value = value[0]
    return np.asarray(value, dtype=np.float64)


def call_with_optional_env_ids(func: Any, *args: Any, env_ids: Any) -> None:
    try:
        func(*args, env_ids=env_ids)
    except TypeError:
        func(*args)


def main() -> int:
    args = parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher({"headless": args.headless})
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    import g0_robot_lab  # noqa: F401
    import g0_robot_lab.tasks  # noqa: F401
    from g0_robot_lab.tasks.locomotion.robots.g0.velocity_env_cfg import G0RobotLabEnvCfg
    from scripts.sim2sim.root_frame_samples import controlled_samples, quat_wxyz_from_rpy

    env = None
    try:
        env_cfg = G0RobotLabEnvCfg()
        env_cfg.scene.num_envs = 1
        env = gym.make(args.task, cfg=env_cfg)
        env.reset()
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        device = base_env.device
        env_ids = torch.tensor([0], dtype=torch.long, device=device)
        samples = controlled_samples()

        rows: dict[str, list[Any]] = {
            "sample_name": [],
            "commanded_rpy": [],
            "commanded_ang_vel": [],
            "root_quat_w": [],
            "projected_gravity": [],
            "base_ang_vel": [],
            "root_pos": [],
            "root_height": [],
        }

        for sample in samples:
            quat = quat_wxyz_from_rpy(*sample.rpy)
            root_pose = robot.data.default_root_state[:, :7].clone()
            root_pose[:, 0:3] = robot.data.default_root_state[:, 0:3]
            root_pose[:, 3:7] = torch.tensor(quat, device=device, dtype=root_pose.dtype).unsqueeze(0)
            root_velocity = torch.zeros_like(robot.data.default_root_state[:, 7:])
            root_velocity[:, 3:6] = torch.tensor(sample.ang_vel, device=device, dtype=root_velocity.dtype).unsqueeze(0)
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = torch.zeros_like(robot.data.default_joint_vel)

            if hasattr(base_env.scene, "reset"):
                base_env.scene.reset(env_ids)
            call_with_optional_env_ids(robot.write_root_pose_to_sim, root_pose, env_ids=env_ids)
            call_with_optional_env_ids(robot.write_root_velocity_to_sim, root_velocity, env_ids=env_ids)
            call_with_optional_env_ids(robot.write_joint_state_to_sim, joint_pos, joint_vel, env_ids=env_ids)
            call_with_optional_env_ids(robot.set_joint_position_target, joint_pos, env_ids=env_ids)
            base_env.scene.write_data_to_sim()
            if hasattr(base_env, "sim"):
                base_env.sim.forward()
            robot.update(float(base_env.cfg.sim.dt))

            rows["sample_name"].append(sample.name)
            rows["commanded_rpy"].append(np.asarray(sample.rpy, dtype=np.float64))
            rows["commanded_ang_vel"].append(np.asarray(sample.ang_vel, dtype=np.float64))
            rows["root_quat_w"].append(tensor_to_numpy(robot.data.root_quat_w))
            rows["projected_gravity"].append(tensor_to_numpy(robot.data.projected_gravity_b))
            rows["base_ang_vel"].append(tensor_to_numpy(robot.data.root_ang_vel_b))
            root_pos = tensor_to_numpy(robot.data.root_pos_w)
            rows["root_pos"].append(root_pos)
            rows["root_height"].append(float(root_pos[2]))

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, **{key: np.asarray(value) for key, value in rows.items()})
        print(f"Saved Isaac controlled root state: {output}")
        return 0
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
