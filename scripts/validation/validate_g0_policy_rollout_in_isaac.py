#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import os
import sys
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SOURCE_ROOT = REPO_ROOT / "source" / "g0_robot_lab"
RSL_RL_SCRIPT_DIR = REPO_ROOT / "scripts" / "rsl_rl"
for path in (REPO_ROOT, SOURCE_ROOT, RSL_RL_SCRIPT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Validate G0 policy rollout safety inside Isaac Lab.")
parser.add_argument("--task", type=str, default="G0-Velocity-v0")
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--root-z", type=float, default=0.233)
parser.add_argument("--json-out", type=Path, default=None)
parser.add_argument("--no-json", action="store_true", default=False)
parser.add_argument("--effort-ratio-threshold", type=float, default=0.9)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(
    checkpoint=str(REPO_ROOT / "logs" / "rsl_rl" / "g0_velocity" / "2026-05-14_18-29-19" / "model_9999.pt")
)
args_cli = parser.parse_args()

os.environ["G0_ALLOW_HARDWARE"] = "0"

EXPECTED_CHECKPOINT_SHA256 = "1dc0c434a4b991eaaa435a21b9d4265e0267eb781b69b132bd75a0b5883928cd"
EXPECTED_ACTION_DIM = 22
EXIT_CONTRACT_FAILURE = 2


def _iter_obs_tensors(obs: Any):
    if isinstance(obs, dict):
        if "policy" in obs:
            yield from _iter_obs_tensors(obs["policy"])
            return
        for value in obs.values():
            yield from _iter_obs_tensors(value)
        return
    if hasattr(obs, "items") and callable(obs.items):
        items = list(obs.items())
        if any(key == "policy" for key, _value in items):
            for key, value in items:
                if key == "policy":
                    yield from _iter_obs_tensors(value)
                    return
        for _key, value in items:
            yield from _iter_obs_tensors(value)
        return
    if isinstance(obs, tuple):
        for value in obs:
            yield from _iter_obs_tensors(value)
        return
    if hasattr(obs, "detach") and hasattr(obs, "shape"):
        yield obs.detach()


def _resolve_checkpoint_path(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _reset_env(env: Any) -> Any:
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def _step_env(env: Any, actions: Any) -> tuple[Any, Any, Any, Any, Any]:
    out = env.step(actions)
    if not isinstance(out, tuple):
        raise RuntimeError(f"env.step(actions) returned non-tuple: {type(out)}")
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        dones = terminated | truncated
        return obs, reward, dones, terminated, truncated, info
    if len(out) == 4:
        obs, reward, dones, info = out
        import torch

        terminated = dones
        truncated = torch.zeros_like(dones, dtype=torch.bool)
        return obs, reward, dones, terminated, truncated, info
    raise RuntimeError(f"Unsupported env.step return length: {len(out)}")


def _compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint missing: {path}")
    digest = _compute_sha256(path)
    if digest != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(f"SHA256 mismatch for {path}: got {digest}, expected {EXPECTED_CHECKPOINT_SHA256}")
    return digest


def _load_policy(task: str, checkpoint: Path):
    import gymnasium as gym
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    import g0_robot_lab.tasks  # noqa: F401
    from g0_robot_lab.tasks.locomotion.robots.g0.velocity_env_cfg import G0RobotLabEnvCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    from _rollout_env_cfg import apply_fixed_policy_rollout_env_cfg

    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = cli_args.parse_rsl_rl_cfg(task, args_cli)
    try:
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    except TypeError:
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg)
    env_cfg = G0RobotLabEnvCfg()
    env_cfg.seed = agent_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    apply_fixed_policy_rollout_env_cfg(env_cfg, num_envs=args_cli.num_envs, root_z=args_cli.root_z)

    env = gym.make(task, cfg=env_cfg)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(wrapped_env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=wrapped_env.unwrapped.device)
    if hasattr(policy, "reset"):
        policy_reset = policy.reset
    else:
        policy_nn = getattr(runner.alg, "policy", None)
        if policy_nn is None:
            policy_nn = getattr(runner.alg, "actor_critic", None)
        if policy_nn is None or not hasattr(policy_nn, "reset"):
            raise RuntimeError("Loaded inference policy does not provide a reset path.")
        policy_reset = policy_nn.reset
    return wrapped_env, policy, policy_reset, env_cfg, agent_cfg


def _compute_target_and_limits(robot: Any, actions, action_joint_order: list[str], default_joint_pos: dict[str, float]):
    import torch

    data = robot.data
    joint_names = [str(name) for name in getattr(data, "joint_names", getattr(robot, "joint_names", []))]
    joint_pos = getattr(data, "joint_pos", None)
    if joint_pos is None:
        return None, None, None, None, None, None, None, None, None

    joint_targets = getattr(data, "joint_pos_target", None)
    if joint_targets is None and joint_names:
        joint_targets = joint_pos.clone()
        name_to_index = {name: index for index, name in enumerate(joint_names)}
        clipped_actions = actions.clamp(-1.0, 1.0)
        for action_index, joint_name in enumerate(action_joint_order):
            joint_index = name_to_index.get(joint_name)
            if joint_index is None:
                continue
            default = default_joint_pos.get(joint_name)
            if default is None:
                continue
            joint_targets[:, joint_index] = default + 0.12 * clipped_actions[:, action_index]

    target_delta = None
    if joint_targets is not None:
        target_delta = joint_targets - joint_pos

    applied_torque = getattr(data, "applied_torque", None)
    computed_torque = getattr(data, "computed_torque", None)
    torque = applied_torque
    if torque is None:
        torque = computed_torque
    elif computed_torque is not None and torch.count_nonzero(torque).item() == 0 and torch.count_nonzero(computed_torque).item() > 0:
        torque = computed_torque

    effort_limit = getattr(data, "joint_effort_limits", None)
    if effort_limit is None:
        effort_limit = getattr(data, "default_joint_effort_limits", None)
    effort_ratio = None
    if torque is not None and effort_limit is not None:
        safe_limit = torch.where(torch.abs(effort_limit) > 0.0, torch.abs(effort_limit), torch.ones_like(effort_limit))
        effort_ratio = torch.abs(torque) / safe_limit

    joint_limits = getattr(data, "soft_joint_pos_limits", None)
    if joint_limits is None:
        joint_limits = getattr(data, "joint_pos_limits", None)
    joint_limit_margin = None
    joint_lower_limit = None
    joint_upper_limit = None
    if joint_limits is not None:
        joint_lower_limit = joint_limits[..., 0]
        joint_upper_limit = joint_limits[..., 1]
        lower_margin = joint_pos - joint_lower_limit
        upper_margin = joint_upper_limit - joint_pos
        joint_limit_margin = torch.minimum(lower_margin, upper_margin)

    return (
        joint_targets,
        target_delta,
        torque,
        effort_limit,
        effort_ratio,
        joint_pos,
        joint_lower_limit,
        joint_upper_limit,
        joint_limit_margin,
    )


def _build_payload(
    *,
    command: list[str],
    checkpoint: Path,
    sha256: str,
    metrics: Any,
    errors: list[str],
    json_path: Path | None,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "command": command,
        "checkpoint": {"path": str(checkpoint), "sha256": sha256},
        "config": {
            "task": args_cli.task,
            "steps": args_cli.steps,
            "num_envs": args_cli.num_envs,
            "seed": args_cli.seed,
            "device": args_cli.device,
            "root_z": args_cli.root_z,
            "effort_ratio_threshold": args_cli.effort_ratio_threshold,
        },
        "phase": 1,
        "result": {
            "contract_pass": not errors,
            "exit_code": 0 if not errors else EXIT_CONTRACT_FAILURE,
            "errors": errors,
        },
        "metrics": metrics.summary(),
        "warnings": metrics.warnings(),
        "json_out": str(json_path) if json_path is not None else None,
    }


def main() -> int:
    from _rollout_io import default_json_path, print_summary, write_json
    from _rollout_metrics import RolloutMetrics

    metrics = RolloutMetrics(
        expected_action_dim=EXPECTED_ACTION_DIM,
        num_envs=args_cli.num_envs,
        effort_ratio_threshold=args_cli.effort_ratio_threshold,
    )
    json_path = None if args_cli.no_json else (args_cli.json_out.resolve() if args_cli.json_out else default_json_path(REPO_ROOT))
    checkpoint_path = _resolve_checkpoint_path(args_cli.checkpoint)
    command = sys.argv[:]
    sha256 = "unverified"
    errors: list[str] = []
    simulation_app = None
    env = None
    try:
        sha256 = verify_checkpoint(checkpoint_path)
        app_launcher = AppLauncher(args_cli)
        simulation_app = app_launcher.app
        from g0_robot_lab.assets.robots.g0.g0 import G0_DEFAULT_JOINT_POS
        from tests.helpers.isaaclab_runtime import get_robot_joint_names, resolve_action_joint_order

        env, policy, policy_reset, env_cfg, agent_cfg = _load_policy(args_cli.task, checkpoint_path)
        del env_cfg, agent_cfg

        _reset_env(env)
        obs = env.get_observations()
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        robot_joint_names = get_robot_joint_names(robot)
        action_joint_order = resolve_action_joint_order(base_env.action_manager, robot_joint_names)
        metrics.action_joint_names = action_joint_order
        metrics.robot_joint_names = robot_joint_names

        for step in range(args_cli.steps):
            obs_tensors = list(_iter_obs_tensors(obs))
            if not obs_tensors:
                raise RuntimeError("Could not resolve policy observation tensor for finiteness check.")
            for obs_tensor in obs_tensors:
                metrics.record_obs(step, obs_tensor)
            if metrics.obs_non_finite_step is not None:
                break

            with torch.inference_mode():
                actions = policy(obs)
            clipped_actions = actions.clamp(-1.0, 1.0)
            metrics.record_actions(step, actions, clipped_actions)
            if metrics.actions_non_finite_step is not None or metrics.action_shape_error is not None:
                break

            obs, _reward, dones, terminated, truncated, _info = _step_env(env, actions)
            (
                target_joint_pos,
                target_delta,
                torque,
                effort_limit,
                effort_ratio,
                joint_pos,
                joint_lower_limit,
                joint_upper_limit,
                joint_limit_margin,
            ) = _compute_target_and_limits(robot, actions, action_joint_order, G0_DEFAULT_JOINT_POS)
            metrics.record_step(
                step=step,
                robot=robot,
                dones=dones,
                terminated=terminated,
                truncated=truncated,
                termination_manager=getattr(base_env, "termination_manager", None),
                target_joint_pos=target_joint_pos,
                target_delta=target_delta,
                torque=torque,
                effort_limit=effort_limit,
                effort_ratio=effort_ratio,
                joint_pos=joint_pos,
                joint_lower_limit=joint_lower_limit,
                joint_upper_limit=joint_upper_limit,
                joint_limit_margin=joint_limit_margin,
            )
            policy_reset(dones)
        errors = metrics.contract_errors()
    except Exception as exc:
        errors = [f"{type(exc).__name__}: {exc}"]
    finally:
        if env is not None:
            env.close()

    payload = _build_payload(
        command=command,
        checkpoint=checkpoint_path,
        sha256=sha256,
        metrics=metrics,
        errors=errors,
        json_path=json_path,
    )
    print_summary(payload)
    if json_path is not None:
        write_json(json_path, payload)
    if simulation_app is not None:
        simulation_app.close()
    return payload["result"]["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
