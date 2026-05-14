"""Run a short G0 PPO training pass with explicit finite-value diagnostics."""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Check G0 training numerics.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default="G0-Velocity-v0")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--iterations", type=int, default=5)
parser.add_argument("--steps_per_env", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from tensordict import TensorDict

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import importlib.metadata as metadata

import isaaclab_tasks  # noqa: F401
import g0_robot_lab.tasks  # noqa: F401


def _iter_tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, TensorDict):
        for value in obj.values():
            yield from _iter_tensors(value)
    elif isinstance(obj, dict):
        for value in obj.values():
            yield from _iter_tensors(value)


def _finite(name: str, obj) -> bool:
    ok = True
    for tensor in _iter_tensors(obj):
        if not torch.isfinite(tensor).all():
            print(f"[G0 numerics] non-finite {name}: shape={tuple(tensor.shape)}")
            ok = False
    return ok


def _tensor_stats(name: str, tensor: torch.Tensor) -> str:
    tensor = tensor.detach().float()
    finite = torch.isfinite(tensor)
    if not finite.any():
        return f"{name}: all non-finite shape={tuple(tensor.shape)}"
    valid = tensor[finite]
    return (
        f"{name}: mean={valid.mean().item():.6g}, min={valid.min().item():.6g}, "
        f"max={valid.max().item():.6g}, finite={finite.float().mean().item():.3f}"
    )


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
        print("[G0 numerics] strict load failed; trying actor/critic migration.")
        print(f"[G0 numerics] original load error: {exc}")
    loaded = torch.load(checkpoint, weights_only=False, map_location=runner.device)
    migrated = _migrate_scalar_std_checkpoint_for_log_std(loaded)
    runner.alg.load(
        loaded,
        load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": True, "rnd": False},
        strict=False,
    )
    if "iter" in loaded:
        runner.current_learning_iteration = loaded["iter"]
    print(f"[G0 numerics] loaded actor/critic, optimizer reset, migrated_scalar_std={migrated}")


def _dump_debug(iteration: int, env_step: int, runner: OnPolicyRunner, env) -> None:
    base_env = env.unwrapped
    print(f"\n[G0 numerics] dump iteration={iteration} env_step={env_step}")
    action = getattr(env, "actions", None)
    if action is None and hasattr(base_env, "action_manager"):
        action = base_env.action_manager.action
    if action is not None:
        print(_tensor_stats("action", action))
    obs = env.get_observations()
    for key, value in obs.items():
        for idx, tensor in enumerate(_iter_tensors(value)):
            print(_tensor_stats(f"obs/{key}/{idx}", tensor))
    if hasattr(runner.alg.actor, "output_distribution_params"):
        params = runner.alg.actor.output_distribution_params
        if len(params) >= 2:
            print(_tensor_stats("actor_mean", params[0]))
            print(_tensor_stats("actor_std", params[1]))
    if getattr(runner.alg, "transition", None) is not None and runner.alg.transition.values is not None:
        print(_tensor_stats("critic_value", runner.alg.transition.values))
    if hasattr(base_env, "reward_manager"):
        for name, sums in base_env.reward_manager._episode_sums.items():
            print(_tensor_stats(f"episode_reward_sum/{name}", sums))
    if hasattr(base_env, "termination_manager"):
        for name in base_env.termination_manager.active_terms:
            term = base_env.termination_manager.get_term(name).float()
            print(f"termination/{name}: ratio={term.mean().item():.6g}")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    if args_cli.steps_per_env is not None:
        agent_cfg.num_steps_per_env = args_cli.steps_per_env
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_dir = Path("logs/rsl_rl") / agent_cfg.experiment_name / (
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_g0_numerics"
    )
    env_cfg.log_dir = str(log_dir.resolve())

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    if args_cli.checkpoint is not None:
        _load_checkpoint(runner, args_cli.checkpoint)

    obs = env.get_observations().to(runner.device)
    runner.alg.train_mode()
    runner.logger.init_logging_writer()

    start_wall_time = time.time()
    start_iteration = runner.current_learning_iteration
    for local_it in range(args_cli.iterations):
        iteration = start_iteration + local_it
        try:
            with torch.inference_mode():
                for env_step in range(agent_cfg.num_steps_per_env):
                    actions = runner.alg.act(obs)
                    if not _finite("actions", actions):
                        _dump_debug(iteration, env_step, runner, env)
                        raise RuntimeError("non-finite actions")
                    params = runner.alg.actor.output_distribution_params
                    if not _finite("actor_mean", params[0]) or not _finite("actor_std", params[1]):
                        _dump_debug(iteration, env_step, runner, env)
                        raise RuntimeError("non-finite actor distribution")
                    if not _finite("critic_value", runner.alg.transition.values):
                        _dump_debug(iteration, env_step, runner, env)
                        raise RuntimeError("non-finite critic value")
                    obs, rewards, dones, extras = env.step(actions.to(env.device))
                    if not _finite("obs", obs) or not _finite("rewards", rewards):
                        _dump_debug(iteration, env_step, runner, env)
                        raise RuntimeError("non-finite env output")
                    obs, rewards, dones = obs.to(runner.device), rewards.to(runner.device), dones.to(runner.device)
                    runner.alg.process_env_step(obs, rewards, dones, extras)
                    runner.logger.process_env_step(rewards, dones, extras, None)
                runner.alg.compute_returns(obs)
            loss_dict = runner.alg.update()
            for name, value in loss_dict.items():
                if not torch.isfinite(torch.tensor(value)):
                    _dump_debug(iteration, agent_cfg.num_steps_per_env, runner, env)
                    raise RuntimeError(f"non-finite loss_dict[{name}]={value}")
            print(
                f"[G0 numerics] iteration={iteration} ok "
                f"loss={loss_dict} std_mean={runner.alg.get_policy().output_std.mean().item():.6g}"
            )
        except Exception:
            _dump_debug(iteration, env_step if "env_step" in locals() else -1, runner, env)
            raise

    print(f"[G0 numerics] completed {args_cli.iterations} iterations in {time.time() - start_wall_time:.2f}s")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
