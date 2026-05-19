from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata as metadata
import os
import sys
from pathlib import Path
from typing import Any

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


DEFAULT_TASK = "G0-Velocity-v0"
DEFAULT_CHECKPOINT = REPO_ROOT / "logs" / "rsl_rl" / "g0_velocity" / "2026-05-14_18-29-19" / "model_9999.pt"
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


def _make_rsl_rl_args(*, checkpoint: Path, seed: int) -> argparse.Namespace:
    return argparse.Namespace(
        seed=seed,
        resume=False,
        load_run=None,
        checkpoint=str(checkpoint),
        experiment_name=None,
        run_name=None,
        logger=None,
        log_project_name=None,
    )


def _load_policy(
    *,
    task: str,
    checkpoint: Path,
    seed: int,
    num_envs: int,
    root_z: float,
    device: str | None,
):
    import gymnasium as gym
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    import g0_robot_lab.tasks  # noqa: F401
    from g0_robot_lab.tasks.locomotion.robots.g0.velocity_env_cfg import G0RobotLabEnvCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

    try:
        from ._rollout_env_cfg import apply_fixed_policy_rollout_env_cfg
    except ImportError:
        from _rollout_env_cfg import apply_fixed_policy_rollout_env_cfg

    installed_version = metadata.version("rsl-rl-lib")
    agent_args = _make_rsl_rl_args(checkpoint=checkpoint, seed=seed)
    agent_cfg = cli_args.parse_rsl_rl_cfg(task, agent_args)
    try:
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    except TypeError:
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg)

    env_cfg = G0RobotLabEnvCfg()
    env_cfg.seed = agent_cfg.seed = seed
    env_cfg.sim.device = device if device is not None else env_cfg.sim.device
    apply_fixed_policy_rollout_env_cfg(env_cfg, num_envs=num_envs, root_z=root_z)

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
    return wrapped_env, policy, policy_reset, runner, env_cfg, agent_cfg


def _compute_target_and_limits(robot: Any, actions, action_joint_order: list[str], default_joint_pos: dict[str, float]):
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
    task: str,
    steps: int,
    num_envs: int,
    seed: int,
    device: str | None,
    root_z: float,
    effort_ratio_threshold: float,
    metrics: Any,
    errors: list[str],
    json_path: Path | None,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "command": command,
        "checkpoint": {"path": str(checkpoint), "sha256": sha256},
        "config": {
            "task": task,
            "steps": steps,
            "num_envs": num_envs,
            "seed": seed,
            "device": device,
            "root_z": root_z,
            "effort_ratio_threshold": effort_ratio_threshold,
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


def run_policy_rollout_validation(
    *,
    task: str = DEFAULT_TASK,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    steps: int = 500,
    num_envs: int = 1,
    seed: int = 42,
    root_z: float = 0.233,
    effort_ratio_threshold: float = 0.9,
    json_out: Path | None = None,
    write_json: bool = False,
    device: str | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    from g0_robot_lab.assets.robots.g0.g0 import G0_DEFAULT_JOINT_POS
    from tests.helpers.isaaclab_runtime import get_robot_joint_names, resolve_action_joint_order

    try:
        from ._rollout_io import default_json_path, write_json as write_payload_json
        from ._rollout_metrics import RolloutMetrics
    except ImportError:
        from _rollout_io import default_json_path, write_json as write_payload_json
        from _rollout_metrics import RolloutMetrics

    os.environ["G0_ALLOW_HARDWARE"] = "0"

    metrics = RolloutMetrics(
        expected_action_dim=EXPECTED_ACTION_DIM,
        num_envs=num_envs,
        effort_ratio_threshold=effort_ratio_threshold,
    )
    checkpoint_path = _resolve_checkpoint_path(checkpoint)
    resolved_json_path = None
    if write_json:
        resolved_json_path = json_out.resolve() if json_out is not None else default_json_path(REPO_ROOT)
    command = [] if command is None else command
    sha256 = "unverified"
    errors: list[str] = []
    env = None
    policy = None
    policy_reset = None
    runner = None
    base_env = None
    robot = None
    try:
        sha256 = verify_checkpoint(checkpoint_path)
        env, policy, policy_reset, runner, env_cfg, agent_cfg = _load_policy(
            task=task,
            checkpoint=checkpoint_path,
            seed=seed,
            num_envs=num_envs,
            root_z=root_z,
            device=device,
        )
        resolved_device = str(env.unwrapped.device)
        del env_cfg, agent_cfg

        _reset_env(env)
        obs = env.get_observations()
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        robot_joint_names = get_robot_joint_names(robot)
        action_joint_order = resolve_action_joint_order(base_env.action_manager, robot_joint_names)
        metrics.action_joint_names = action_joint_order
        metrics.robot_joint_names = robot_joint_names

        for step in range(steps):
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
        resolved_device = device
    finally:
        if env is not None:
            env.close()
        del robot
        del base_env
        del policy_reset
        del policy
        del runner
        gc.collect()

    payload = _build_payload(
        command=command,
        checkpoint=checkpoint_path,
        sha256=sha256,
        task=task,
        steps=steps,
        num_envs=num_envs,
        seed=seed,
        device=resolved_device,
        root_z=root_z,
        effort_ratio_threshold=effort_ratio_threshold,
        metrics=metrics,
        errors=errors,
        json_path=resolved_json_path,
    )
    if write_json and resolved_json_path is not None:
        write_payload_json(resolved_json_path, payload)
    return payload
