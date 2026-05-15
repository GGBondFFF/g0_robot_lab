#!/usr/bin/env python3
"""Dump Isaac Lab golden observation/action/state data for MuJoCo alignment.

This script must be launched with Isaac Lab, for example:

```
/home/lz/IsaacLab/isaaclab.sh -p scripts/sim2sim/dump_isaac_golden_io.py --zero-action --headless
```
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="G0-Velocity-v0", help="Isaac Lab task id.")
    parser.add_argument("--checkpoint", default=None, help="Exported TorchScript policy.pt path. RSL-RL checkpoints are TODO.")
    parser.add_argument("--steps", type=int, default=100, help="Number of control steps to dump.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of Isaac environments. Use 1 for golden I/O.")
    parser.add_argument("--output", default="logs/sim2sim/isaac_golden_io.npz", help="Output .npz path.")
    parser.add_argument("--zero-action", action="store_true", help="Use zero policy actions instead of loading a policy.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless.")
    parser.add_argument("--deterministic-zero", action="store_true", help="Temporarily freeze reset randomization and commands for zero-action diagnostics.")
    parser.add_argument("--command", nargs=3, type=float, default=[0.0, 0.0, 0.0], help="Fixed command for --deterministic-zero: lin_x lin_y yaw.")
    parser.add_argument("--root-pos", nargs=3, type=float, default=[0.0, 0.0, 0.23], help="Root position used for deterministic reset diagnostics.")
    parser.add_argument("--root-yaw", type=float, default=0.0, help="Root yaw used for deterministic reset diagnostics.")
    parser.add_argument("--zero-joint-velocity", action="store_true", help="Force reset joint velocity range to zero.")
    parser.add_argument("--seed", type=int, default=0, help="Seed used when --deterministic-zero is enabled.")
    return parser.parse_args()


def warn_missing(name: str, exc: Exception) -> None:
    warnings.warn(f"Could not collect {name}: {exc}", stacklevel=2)


def tensor_to_numpy(value: Any, env_index: int = 0) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim > 0 and value.shape[0] > env_index:
        value = value[env_index]
    return np.asarray(value)


def extract_policy_obs(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        if "policy" in obs:
            return tensor_to_numpy(obs["policy"])
        first_key = next(iter(obs))
        return tensor_to_numpy(obs[first_key])
    return tensor_to_numpy(obs)


def load_torchscript_policy(path: str | None, device: str):
    if path is None:
        return None
    policy_path = Path(path)
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy/checkpoint does not exist: {policy_path}")
    try:
        import torch

        policy = torch.jit.load(str(policy_path), map_location=device)
        policy.eval()
        return policy
    except Exception as exc:
        raise RuntimeError(
            "Only exported TorchScript policy.pt is supported here. "
            "Raw RSL-RL training checkpoints should first be exported by scripts/rsl_rl/play.py."
        ) from exc


def safe_get_joint_indices(robot, joint_names: list[str]) -> list[int] | None:
    try:
        result = robot.find_joints(joint_names, preserve_order=True)
        if isinstance(result, tuple):
            return list(result[0])
        return list(result)
    except Exception as exc:
        warn_missing("joint indices", exc)
        return None


def safe_get_body_indices(robot, body_names: list[str]) -> list[int] | None:
    try:
        result = robot.find_bodies(body_names, preserve_order=True)
        if isinstance(result, tuple):
            return list(result[0])
        return list(result)
    except Exception as exc:
        warn_missing("body indices", exc)
        return None


def configure_deterministic_zero(env_cfg: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Apply in-memory deterministic overrides to an Isaac env cfg instance."""

    command = tuple(float(value) for value in args.command)
    root_pos = tuple(float(value) for value in args.root_pos)
    root_yaw = float(args.root_yaw)
    overrides: dict[str, Any] = {
        "deterministic_zero": True,
        "command": command,
        "root_pos": root_pos,
        "root_yaw": root_yaw,
        "zero_joint_velocity": bool(args.deterministic_zero or args.zero_joint_velocity),
        "reset_base_pose_range": {"x": (root_pos[0], root_pos[0]), "y": (root_pos[1], root_pos[1]), "yaw": (root_yaw, root_yaw)},
        "reset_base_velocity_range": "all_zero",
        "command_ranges": "fixed",
        "physics_material_randomization": "fixed_or_disabled",
        "push_robot": "disabled",
    }

    try:
        env_cfg.seed = int(args.seed)
    except Exception as exc:
        warn_missing("deterministic seed override", exc)

    try:
        env_cfg.scene.robot.init_state.pos = root_pos
    except Exception as exc:
        warn_missing("deterministic root init_state.pos override", exc)

    try:
        pose_range = env_cfg.events.reset_base.params["pose_range"]
        pose_range["x"] = (root_pos[0], root_pos[0])
        pose_range["y"] = (root_pos[1], root_pos[1])
        pose_range["yaw"] = (root_yaw, root_yaw)
        velocity_range = env_cfg.events.reset_base.params["velocity_range"]
        for key in ("x", "y", "z", "roll", "pitch", "yaw"):
            velocity_range[key] = (0.0, 0.0)
    except Exception as exc:
        warn_missing("deterministic reset_base override", exc)

    if args.deterministic_zero or args.zero_joint_velocity:
        try:
            env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        except Exception as exc:
            warn_missing("deterministic reset_robot_joints velocity override", exc)

    try:
        ranges = env_cfg.commands.base_velocity.ranges
        ranges.lin_vel_x = (command[0], command[0])
        ranges.lin_vel_y = (command[1], command[1])
        ranges.ang_vel_z = (command[2], command[2])
        limit_ranges = env_cfg.commands.base_velocity.limit_ranges
        limit_ranges.lin_vel_x = (command[0], command[0])
        limit_ranges.lin_vel_y = (command[1], command[1])
        limit_ranges.ang_vel_z = (command[2], command[2])
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
        env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    except Exception as exc:
        warn_missing("deterministic command override", exc)

    try:
        env_cfg.events.physics_material.params["static_friction_range"] = (1.0, 1.0)
        env_cfg.events.physics_material.params["dynamic_friction_range"] = (1.0, 1.0)
        env_cfg.events.physics_material.params["restitution_range"] = (0.0, 0.0)
        env_cfg.events.physics_material.params["num_buckets"] = 1
    except Exception as exc:
        warn_missing("deterministic physics material override", exc)

    try:
        env_cfg.events.add_base_mass.params["mass_distribution_params"] = (0.0, 0.0)
    except Exception as exc:
        warn_missing("deterministic base mass override", exc)

    try:
        env_cfg.events.push_robot = None
    except Exception as exc:
        warn_missing("deterministic push_robot disable", exc)

    try:
        env_cfg.observations.policy.enable_corruption = False
    except Exception as exc:
        warn_missing("deterministic policy observation corruption disable", exc)
    try:
        env_cfg.observations.critic.enable_corruption = False
    except Exception:
        pass

    return overrides


def extract_foot_contact_forces(contact_sensor: Any, foot_body_indices: list[int] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return all, both-foot, left-foot, and right-foot contact force arrays."""

    contact_force = tensor_to_numpy(contact_sensor.data.net_forces_w)
    sensor_body_names = list(getattr(contact_sensor, "body_names", []) or [])
    if "l_foot_link" in sensor_body_names and "r_foot_link" in sensor_body_names:
        left_index = sensor_body_names.index("l_foot_link")
        right_index = sensor_body_names.index("r_foot_link")
    elif foot_body_indices is not None and len(foot_body_indices) >= 2:
        left_index, right_index = foot_body_indices[:2]
    else:
        raise RuntimeError("foot body indices are unavailable")
    left_force = tensor_to_numpy(contact_sensor.data.net_forces_w[:, left_index : left_index + 1, :])
    right_force = tensor_to_numpy(contact_sensor.data.net_forces_w[:, right_index : right_index + 1, :])
    foot_force = np.concatenate([left_force.reshape(1, 3), right_force.reshape(1, 3)], axis=0)
    return contact_force, foot_force, left_force.reshape(3), right_force.reshape(3)


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        print("ERROR: --steps must be positive.", file=sys.stderr)
        return 2
    if args.checkpoint is None and not args.zero_action:
        print("ERROR: provide --checkpoint or use --zero-action.", file=sys.stderr)
        return 2

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher({"headless": args.headless})
    simulation_app = app_launcher.app

    import torch
    import gymnasium as gym

    import g0_robot_lab  # noqa: F401
    import g0_robot_lab.tasks  # noqa: F401
    from g0_robot_lab.tasks.locomotion.robots.g0.velocity_env_cfg import G0RobotLabEnvCfg
    from scripts.sim2sim import g0_sim2sim_config as cfg

    env = None
    try:
        env_cfg = G0RobotLabEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        deterministic_overrides: dict[str, Any] = {"deterministic_zero": False}
        if args.deterministic_zero:
            deterministic_overrides = configure_deterministic_zero(env_cfg, args)
        env = gym.make(args.task, cfg=env_cfg)
        device = getattr(env.unwrapped, "device", "cpu")
        policy = load_torchscript_policy(args.checkpoint, str(device))

        obs, _ = env.reset()
        robot = env.unwrapped.scene["robot"]
        joint_indices = safe_get_joint_indices(robot, cfg.get_joint_names())
        foot_body_indices = safe_get_body_indices(robot, ["l_foot_link", "r_foot_link"])
        rows: dict[str, list[np.ndarray]] = {
            "obs": [],
            "action": [],
            "target_joint_pos": [],
            "joint_pos": [],
            "joint_vel": [],
            "root_pos": [],
            "root_height": [],
            "root_quat": [],
            "base_ang_vel": [],
            "projected_gravity": [],
            "command": [],
            "joint_acc": [],
            "contact_force": [],
            "foot_contact_force": [],
            "left_foot_contact_force": [],
            "right_foot_contact_force": [],
            "foot_contact_force_norm": [],
            "left_foot_contact_force_norm": [],
            "right_foot_contact_force_norm": [],
        }

        for _ in range(args.steps):
            obs_np = extract_policy_obs(obs)
            if policy is None:
                action_np = np.zeros(cfg.get_action_dim(), dtype=np.float32)
            else:
                with torch.no_grad():
                    action_tensor = policy(torch.as_tensor(obs_np, dtype=torch.float32, device=device).unsqueeze(0))
                action_np = action_tensor.squeeze(0).detach().cpu().numpy()

            action = torch.as_tensor(action_np, dtype=torch.float32, device=device).repeat(args.num_envs, 1)
            target_joint_pos = cfg.compute_target_joint_pos(action_np)
            obs, _, _, _, _ = env.step(action)

            rows["obs"].append(obs_np)
            rows["action"].append(np.clip(action_np, -1.0, 1.0))
            rows["target_joint_pos"].append(target_joint_pos)

            try:
                joint_pos = robot.data.joint_pos
                rows["joint_pos"].append(tensor_to_numpy(joint_pos[:, joint_indices] if joint_indices is not None else joint_pos))
            except Exception as exc:
                warn_missing("joint_pos", exc)
            try:
                joint_vel = robot.data.joint_vel
                rows["joint_vel"].append(tensor_to_numpy(joint_vel[:, joint_indices] if joint_indices is not None else joint_vel))
            except Exception as exc:
                warn_missing("joint_vel", exc)
            try:
                root_pos = tensor_to_numpy(robot.data.root_pos_w)
                rows["root_pos"].append(root_pos)
                rows["root_height"].append(np.asarray(root_pos[2], dtype=np.float64))
            except Exception as exc:
                warn_missing("root_pos", exc)
            try:
                rows["root_quat"].append(tensor_to_numpy(robot.data.root_quat_w))
            except Exception as exc:
                warn_missing("root_quat", exc)
            try:
                rows["base_ang_vel"].append(tensor_to_numpy(robot.data.root_ang_vel_b))
            except Exception as exc:
                warn_missing("base_ang_vel", exc)
            try:
                rows["projected_gravity"].append(tensor_to_numpy(robot.data.projected_gravity_b))
            except Exception as exc:
                warn_missing("projected_gravity", exc)
            try:
                rows["command"].append(tensor_to_numpy(env.unwrapped.command_manager.get_command("base_velocity")))
            except Exception as exc:
                warn_missing("command", exc)
            try:
                joint_acc = robot.data.joint_acc
                rows["joint_acc"].append(tensor_to_numpy(joint_acc[:, joint_indices] if joint_indices is not None else joint_acc))
            except Exception as exc:
                warn_missing("joint_acc", exc)
            try:
                contact_sensor = env.unwrapped.scene.sensors.get("contact_forces")
                contact_force, foot_force, left_force, right_force = extract_foot_contact_forces(contact_sensor, foot_body_indices)
                rows["contact_force"].append(contact_force)
                rows["foot_contact_force"].append(foot_force)
                rows["left_foot_contact_force"].append(left_force)
                rows["right_foot_contact_force"].append(right_force)
                rows["left_foot_contact_force_norm"].append(np.asarray(np.linalg.norm(left_force), dtype=np.float64))
                rows["right_foot_contact_force_norm"].append(np.asarray(np.linalg.norm(right_force), dtype=np.float64))
                rows["foot_contact_force_norm"].append(np.asarray(np.linalg.norm(foot_force), dtype=np.float64))
            except Exception as exc:
                warn_missing("contact_force", exc)

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        arrays = {key: np.asarray(value) for key, value in rows.items() if value}
        np.savez(
            output,
            **arrays,
            default_joint_pos=cfg.get_default_joint_pos_array(),
            joint_names=np.asarray(cfg.get_joint_names()),
            action_scale=np.asarray(cfg.ACTION_SCALE),
            sim_dt=np.asarray(cfg.ISAAC_SIM_DT),
            decimation=np.asarray(cfg.ISAAC_DECIMATION),
            control_dt=np.asarray(cfg.CONTROL_DT),
            deterministic_zero=np.asarray(bool(args.deterministic_zero)),
            deterministic_overrides=np.asarray(str(deterministic_overrides)),
        )
        print(f"Saved Isaac golden I/O: {output}")
        return 0
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
