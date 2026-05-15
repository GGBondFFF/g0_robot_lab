#!/usr/bin/env python3
"""Export a Unitree-style deploy.yaml for the G0 velocity task.

This script must be launched through Isaac Lab so the task registry, managers,
and robot articulation data are available.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="G0-Velocity-v0", help="Isaac Lab task id.")
    parser.add_argument("--output", default="logs/sim2sim/g0_deploy/params/deploy.yaml", help="Output deploy.yaml path.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless.")
    return parser.parse_args()


def to_builtin(value: Any) -> Any:
    """Convert tensors, numpy values, tuples, and config objects to YAML-safe values."""

    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [to_builtin(item) for item in value]
    if isinstance(value, list):
        return [to_builtin(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if hasattr(value, "to_dict"):
        return to_builtin(value.to_dict())
    return value


def warn(message: str) -> None:
    warnings.warn(message, stacklevel=2)


def git_branch() -> str:
    try:
        return subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def tensor_row(value: Any, indices: list[int] | None = None) -> list[float]:
    array = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim >= 2:
        array = array[0]
    if indices is not None:
        array = array[indices]
    return [float(x) for x in np.asarray(array, dtype=np.float64).reshape(-1)]


def resolve_joint_indices(robot: Any, joint_names: list[str]) -> list[int]:
    try:
        result = robot.find_joints(joint_names, preserve_order=True)
        return list(result[0] if isinstance(result, tuple) else result)
    except Exception as exc:
        raise RuntimeError(f"Could not resolve deployment joint order from Isaac robot: {exc}") from exc


def extract_command_cfg(env_cfg: Any) -> dict[str, Any]:
    commands: dict[str, Any] = {}
    try:
        base_velocity = env_cfg.commands.base_velocity
    except Exception as exc:
        warn(f"Could not export base_velocity command config: {exc}")
        return commands

    ranges = getattr(base_velocity, "limit_ranges", None) or getattr(base_velocity, "ranges", None)
    commands["base_velocity"] = {
        "ranges": to_builtin(ranges),
        "deterministic_command": None,
    }
    return commands


def action_class_name(action_term: Any) -> str:
    cfg_obj = getattr(action_term, "cfg", None)
    class_type = getattr(cfg_obj, "class_type", None)
    if class_type is not None:
        return getattr(class_type, "__name__", str(class_type).split(".")[-1])
    return type(action_term).__name__


def extract_actions(env: Any, joint_names: list[str]) -> dict[str, Any]:
    manager = env.unwrapped.action_manager
    actions: dict[str, Any] = {}
    for term_name, action_term in zip(manager.active_terms, manager._terms.values(), strict=True):
        name = action_class_name(action_term)
        action_dim = int(getattr(action_term, "action_dim", len(joint_names)))
        scale = getattr(action_term, "_scale", None)
        if scale is None:
            scale = getattr(getattr(action_term, "cfg", None), "scale", None)
        if scale is None:
            raise RuntimeError(f"Action scale is missing for action term {term_name!r}")
        scale_values = tensor_row(scale)
        if len(scale_values) == 1:
            scale_values = scale_values * action_dim
        offset = getattr(action_term, "_offset", None)
        if offset is None:
            offset_values = [0.0] * action_dim
        else:
            offset_values = tensor_row(offset)
        joint_ids = getattr(action_term, "_joint_ids", None)
        if isinstance(joint_ids, slice):
            joint_ids_value = None
        elif joint_ids is None:
            joint_ids_value = list(range(action_dim))
        else:
            joint_ids_value = [int(index) for index in joint_ids]
        clip = getattr(action_term, "_clip", None)
        clip_value = None if clip is None else to_builtin(clip)

        actions[name] = {
            "source_term_name": str(term_name),
            "action_dim": action_dim,
            "scale": scale_values,
            "offset": offset_values,
            "joint_ids": joint_ids_value,
            "clip": clip_value,
            "clip_policy_action": [-1.0, 1.0],
            "formula": "target_joint_pos = offset + scale * clipped_action",
        }
    if "JointPositionAction" not in actions:
        raise RuntimeError(f"Expected a JointPositionAction action term, got {list(actions)}")
    return actions


def obs_scale(obs_cfg: Any, dim: int) -> list[float]:
    scale = getattr(obs_cfg, "scale", None)
    if scale is None:
        return [1.0] * dim
    values = to_builtin(scale)
    if isinstance(values, (float, int)):
        return [float(values)] * dim
    values = list(np.asarray(values, dtype=np.float64).reshape(-1))
    if len(values) == 1:
        return [float(values[0])] * dim
    return [float(value) for value in values]


def extract_observations(env: Any, policy_obs_dim: int) -> dict[str, Any]:
    manager = env.unwrapped.observation_manager
    term_names = list(manager.active_terms["policy"])
    term_cfgs = list(manager._group_obs_term_cfgs["policy"])
    group_cfg = env.unwrapped.cfg.observations.policy
    group_history = int(getattr(group_cfg, "history_length", 1) or 1)
    flatten_history_dim = bool(getattr(group_cfg, "flatten_history_dim", True))
    terms: list[dict[str, Any]] = []
    for name, obs_cfg in zip(term_names, term_cfgs, strict=True):
        try:
            obs = obs_cfg.func(env.unwrapped, **obs_cfg.params)
            dim = int(obs.shape[-1])
        except Exception as exc:
            warn(f"Could not evaluate observation term {name!r}; falling back to dim=0: {exc}")
            dim = 0
        history_length = int(getattr(obs_cfg, "history_length", 0) or group_history)
        terms.append(
            {
                "name": str(name),
                "dim": dim,
                "scale": obs_scale(obs_cfg, dim),
                "history_length": history_length,
                "clip": to_builtin(getattr(obs_cfg, "clip", None)),
                "params": to_builtin(getattr(obs_cfg, "params", {})),
            }
        )
    return {
        "terms": terms,
        "policy_obs_dim": int(policy_obs_dim),
        "history_length": group_history,
        "flatten_history_dim": flatten_history_dim,
    }


def extract_optional_limits(robot: Any, joint_indices: list[int], joint_names: list[str]) -> tuple[list[float], list[float], list[float]]:
    try:
        from scripts.sim2sim import g0_sim2sim_config as local_cfg

        specs = local_cfg.get_isaac_actuator_specs()
        effort = [float(specs[name].effort_limit_sim) for name in joint_names]
        velocity = [float(specs[name].velocity_limit_sim) for name in joint_names]
        armature = [float(specs[name].armature) for name in joint_names]
        return effort, velocity, armature
    except Exception as exc:
        warn(f"Could not read local actuator specs; trying Isaac articulation data: {exc}")

    effort = tensor_row(getattr(robot.data, "soft_joint_effort_limits"), joint_indices)
    velocity = tensor_row(getattr(robot.data, "soft_joint_vel_limits"), joint_indices)
    armature = [0.0] * len(joint_names)
    return effort, velocity, armature


def extract_policy_obs(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        obs = obs.get("policy", next(iter(obs.values())))
    if hasattr(obs, "detach"):
        obs = obs.detach().cpu().numpy()
    obs = np.asarray(obs)
    return obs[0] if obs.ndim > 1 else obs


def main() -> int:
    args = parse_args()

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher({"headless": args.headless})
    simulation_app = app_launcher.app

    import gymnasium as gym

    import g0_robot_lab  # noqa: F401
    import g0_robot_lab.tasks  # noqa: F401
    from g0_robot_lab.tasks.locomotion.robots.g0.velocity_env_cfg import G0RobotLabEnvCfg
    from scripts.sim2sim import g0_sim2sim_config as local_cfg

    env = None
    try:
        env_cfg = G0RobotLabEnvCfg()
        env_cfg.scene.num_envs = 1
        env = gym.make(args.task, cfg=env_cfg)
        obs, _ = env.reset()
        robot = env.unwrapped.scene["robot"]
        joint_names = local_cfg.get_joint_names()
        if not joint_names:
            raise RuntimeError("joint_names is empty")
        joint_ids_map = resolve_joint_indices(robot, joint_names)
        default_joint_pos = tensor_row(robot.data.default_joint_pos, joint_ids_map)
        if not default_joint_pos:
            raise RuntimeError("default_joint_pos is empty")
        stiffness = tensor_row(robot.data.default_joint_stiffness, joint_ids_map)
        damping = tensor_row(robot.data.default_joint_damping, joint_ids_map)
        effort, velocity, armature = extract_optional_limits(robot, joint_ids_map, joint_names)
        policy_obs_dim = int(extract_policy_obs(obs).shape[-1])
        if policy_obs_dim <= 0:
            raise RuntimeError("policy_obs_dim is missing or zero")

        cfg = {
            "task": args.task,
            "joint_names": joint_names,
            "joint_ids_map": [int(index) for index in joint_ids_map],
            "step_dt": float(env.unwrapped.step_dt),
            "sim_dt": float(env.unwrapped.cfg.sim.dt),
            "decimation": int(env.unwrapped.cfg.decimation),
            "default_joint_pos": default_joint_pos,
            "stiffness": stiffness,
            "damping": damping,
            "effort_limit_sim": effort,
            "velocity_limit_sim": velocity,
            "armature": armature,
            "actions": extract_actions(env, joint_names),
            "observations": extract_observations(env, policy_obs_dim),
            "commands": extract_command_cfg(env_cfg),
            "metadata": {
                "generated_time": dt.datetime.now(dt.timezone.utc).isoformat(),
                "branch": git_branch(),
                "source": "Isaac Lab env managers",
            },
        }

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(yaml.safe_dump(to_builtin(cfg), sort_keys=False), encoding="utf-8")
        print(f"Saved G0 deploy config: {output}")
        return 0
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
