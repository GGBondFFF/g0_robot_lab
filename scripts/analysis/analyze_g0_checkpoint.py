"""Record and plot a single-env G0 checkpoint rollout."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Analyze a G0 locomotion checkpoint.")
parser.add_argument("--task", type=str, default="G0-Velocity-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--duration", type=float, default=12.0)
parser.add_argument("--warmup", type=float, default=2.0)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--video", action="store_true", default=True)
parser.add_argument("--video_length", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import importlib.metadata as metadata

import isaaclab_tasks  # noqa: F401
import g0_robot_lab.tasks  # noqa: F401
from g0_robot_lab.assets.robots.g0 import (
    G0_ARM_JOINT_NAMES,
    G0_LEFT_LEG_JOINT_NAMES,
    G0_RIGHT_LEG_JOINT_NAMES,
    G0_WAIST_JOINT_NAMES,
)


FOOT_NAMES = ["l_foot_link", "r_foot_link"]


def _as_list(value) -> list[str]:
    return [str(v) for v in value] if value is not None else []


def _migrate_scalar_std_checkpoint_for_log_std(loaded_dict: dict) -> bool:
    actor_state_dict = loaded_dict.get("actor_state_dict", {})
    scalar_key = "distribution.std_param"
    log_key = "distribution.log_std_param"
    if scalar_key not in actor_state_dict or log_key in actor_state_dict:
        return False
    std = actor_state_dict.pop(scalar_key).detach().clone().float().clamp_min(1.0e-4)
    actor_state_dict[log_key] = torch.log(std)
    return True


def _load_checkpoint(runner: OnPolicyRunner, checkpoint: str) -> None:
    try:
        runner.load(checkpoint)
        return
    except RuntimeError as exc:
        print("[G0 analysis] strict load failed; trying actor/critic migration.")
        print(f"[G0 analysis] original load error: {exc}")
    loaded = torch.load(checkpoint, weights_only=False, map_location=runner.device)
    migrated = _migrate_scalar_std_checkpoint_for_log_std(loaded)
    runner.alg.load(
        loaded,
        load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": True, "rnd": False},
        strict=False,
    )
    if "iter" in loaded:
        runner.current_learning_iteration = loaded["iter"]
    print(f"[G0 analysis] loaded actor/critic, migrated_scalar_std={migrated}")


def _plot_group(time_s, data, names, title, path, ylabel):
    if len(names) == 0:
        return
    cols = [names.index(name) for name in names if name in names]
    plt.figure(figsize=(12, 6))
    for idx, name in zip(cols, names):
        plt.plot(time_s, data[:, idx], label=name, linewidth=1.1)
    plt.title(title)
    plt.xlabel("time [s]")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def _plot_named(time_s, values, names, title, path, ylabel):
    plt.figure(figsize=(12, 6))
    for i, name in enumerate(names):
        plt.plot(time_s, values[:, i], label=name, linewidth=1.2)
    plt.title(title)
    plt.xlabel("time [s]")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    checkpoint = Path(args_cli.checkpoint).resolve()
    output_dir = Path(args_cli.output_dir) if args_cli.output_dir else checkpoint.parent / "analysis" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    env_cfg.log_dir = str(output_dir.resolve())

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    dt = env.unwrapped.step_dt
    total_steps = int((args_cli.warmup + args_cli.duration) / dt)
    video_length = args_cli.video_length or total_steps
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=str(output_dir / "videos"),
            step_trigger=lambda step: step == 0,
            video_length=video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    _load_checkpoint(runner, str(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    contact_sensor = base_env.scene.sensors["contact_forces"]
    joint_names = _as_list(getattr(robot.data, "joint_names", getattr(robot, "joint_names", [])))
    body_names = _as_list(getattr(robot.data, "body_names", getattr(robot, "body_names", [])))
    foot_ids = [body_names.index(name) for name in FOOT_NAMES]
    sensor_foot_ids, _ = contact_sensor.find_bodies(FOOT_NAMES, preserve_order=True)

    obs = env.get_observations()
    rows = []
    arrays = {
        "time": [],
        "joint_pos": [],
        "joint_vel": [],
        "action": [],
        "target_joint_pos": [],
        "applied_torque": [],
        "root_pos": [],
        "root_rpy_or_projected_gravity": [],
        "foot_contact": [],
        "foot_force": [],
        "foot_clearance": [],
        "foot_slide": [],
        "terminations": [],
    }

    action_term = base_env.action_manager.get_term("joint_pos")
    for step in range(total_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        t = step * dt
        joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()
        joint_vel = robot.data.joint_vel[0].detach().cpu().numpy()
        raw_action = base_env.action_manager.action[0].detach().cpu().numpy()
        target_joint_pos = action_term.processed_actions[0].detach().cpu().numpy()
        applied_torque = robot.data.applied_torque[0].detach().cpu().numpy()
        root_pos = robot.data.root_pos_w[0].detach().cpu().numpy()
        projected_gravity = robot.data.projected_gravity_b[0].detach().cpu().numpy()
        forces = contact_sensor.data.net_forces_w[0, sensor_foot_ids].detach().cpu().numpy()
        foot_force = np.linalg.norm(forces, axis=1)
        foot_contact = (foot_force > 1.0).astype(np.float32)
        foot_pos = robot.data.body_pos_w[0, foot_ids].detach().cpu().numpy()
        foot_vel = robot.data.body_lin_vel_w[0, foot_ids, :2].detach().cpu().numpy()
        foot_clearance = foot_pos[:, 2]
        foot_slide = np.linalg.norm(foot_vel, axis=1) * foot_contact
        terminations = np.array(
            [
                float(base_env.termination_manager.get_term("time_out")[0].item()),
                float(base_env.termination_manager.get_term("base_height")[0].item()),
                float(base_env.termination_manager.get_term("bad_orientation")[0].item()),
            ],
            dtype=np.float32,
        )

        arrays["time"].append(t)
        arrays["joint_pos"].append(joint_pos)
        arrays["joint_vel"].append(joint_vel)
        arrays["action"].append(raw_action)
        arrays["target_joint_pos"].append(target_joint_pos)
        arrays["applied_torque"].append(applied_torque)
        arrays["root_pos"].append(root_pos)
        arrays["root_rpy_or_projected_gravity"].append(projected_gravity)
        arrays["foot_contact"].append(foot_contact)
        arrays["foot_force"].append(foot_force)
        arrays["foot_clearance"].append(foot_clearance)
        arrays["foot_slide"].append(foot_slide)
        arrays["terminations"].append(terminations)

        row = {"step": step, "time": t}
        for name, value in zip(joint_names, joint_pos):
            row[f"joint_pos/{name}"] = value
        for name, value in zip(joint_names, applied_torque):
            row[f"applied_torque/{name}"] = value
        row.update(
            {
                "root_x": root_pos[0],
                "root_y": root_pos[1],
                "root_z": root_pos[2],
                "projected_gravity_x": projected_gravity[0],
                "projected_gravity_y": projected_gravity[1],
                "projected_gravity_z": projected_gravity[2],
                "left_foot_contact": foot_contact[0],
                "right_foot_contact": foot_contact[1],
                "left_foot_force": foot_force[0],
                "right_foot_force": foot_force[1],
                "left_foot_slide": foot_slide[0],
                "right_foot_slide": foot_slide[1],
            }
        )
        rows.append(row)

    np_arrays = {key: np.asarray(value) for key, value in arrays.items()}
    np_arrays["joint_names"] = np.asarray(joint_names)
    np_arrays["foot_names"] = np.asarray(FOOT_NAMES)
    npz_path = output_dir / "rollout_data.npz"
    np.savez(npz_path, **np_arrays)

    csv_path = output_dir / "rollout_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    time_s = np_arrays["time"]
    analysis_mask = (time_s >= args_cli.warmup) & (time_s <= args_cli.warmup + args_cli.duration)
    analysis_time_s = time_s[analysis_mask]
    groups = {
        "left_leg": G0_LEFT_LEG_JOINT_NAMES,
        "right_leg": G0_RIGHT_LEG_JOINT_NAMES,
        "waist": G0_WAIST_JOINT_NAMES,
        "left_arm": [name for name in G0_ARM_JOINT_NAMES if name.startswith("l_")],
        "right_arm": [name for name in G0_ARM_JOINT_NAMES if name.startswith("r_")],
    }
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    for group_name, group_joints in groups.items():
        indices = [name_to_idx[name] for name in group_joints if name in name_to_idx]
        names = [name for name in group_joints if name in name_to_idx]
        if indices:
            _plot_named(analysis_time_s, np_arrays["applied_torque"][analysis_mask][:, indices], names, f"{group_name} applied torque", output_dir / f"torque_{group_name}.png", "Nm")
            _plot_named(analysis_time_s, np_arrays["joint_pos"][analysis_mask][:, indices], names, f"{group_name} joint position", output_dir / f"joint_pos_{group_name}.png", "rad")

    _plot_named(analysis_time_s, np_arrays["root_pos"][analysis_mask], ["x", "y", "z"], "root/base position approximation", output_dir / "root_position_approximation.png", "m")
    _plot_named(analysis_time_s, np_arrays["root_rpy_or_projected_gravity"][analysis_mask], ["gx", "gy", "gz"], "projected gravity body frame", output_dir / "projected_gravity.png", "")
    _plot_named(analysis_time_s, np_arrays["foot_contact"][analysis_mask], FOOT_NAMES, "foot contact state", output_dir / "foot_contact.png", "contact")
    _plot_named(analysis_time_s, np_arrays["foot_force"][analysis_mask], FOOT_NAMES, "foot contact force", output_dir / "foot_contact_force.png", "N")
    _plot_named(analysis_time_s, np_arrays["foot_slide"][analysis_mask], FOOT_NAMES, "feet slide metric", output_dir / "feet_slide.png", "m/s while contact")
    _plot_named(analysis_time_s, np_arrays["foot_clearance"][analysis_mask], FOOT_NAMES, "feet clearance", output_dir / "feet_clearance.png", "m")

    video_dir = output_dir / "videos"
    print(f"[G0 analysis] checkpoint: {checkpoint}")
    print(f"[G0 analysis] npz: {npz_path}")
    print(f"[G0 analysis] csv: {csv_path}")
    print(f"[G0 analysis] plots: {output_dir}")
    print(f"[G0 analysis] video_dir: {video_dir if args_cli.video else 'not recorded'}")
    print(f"[G0 analysis] analyzed segment: t={args_cli.warmup:.2f}s to t={args_cli.warmup + args_cli.duration:.2f}s")
    print(
        "[G0 analysis] analyzed steps: "
        f"{int(np.nonzero(analysis_mask)[0][0])} to {int(np.nonzero(analysis_mask)[0][-1])}"
    )
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
