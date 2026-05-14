from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _episode_reward_per_second(env: ManagerBasedRLEnv, env_ids: Sequence[int], reward_term_name: str) -> torch.Tensor:
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s
    return reward / max(abs(float(reward_term.weight)), 1.0e-6)


def _termination_rate(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str) -> torch.Tensor:
    if not hasattr(env.termination_manager, "_last_episode_dones"):
        return torch.zeros((), device=env.device)
    try:
        term_idx = env.termination_manager._term_name_to_term_idx[term_name]
    except KeyError:
        return torch.zeros((), device=env.device)
    return env.termination_manager._last_episode_dones[env_ids, term_idx].float().mean()


def _stable_enough(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    min_tracking_reward: float,
    min_episode_length_ratio: float,
    max_bad_orientation_rate: float,
    max_base_height_rate: float,
) -> bool:
    tracking_reward = _episode_reward_per_second(env, env_ids, reward_term_name)
    episode_length_ratio = torch.mean(env.episode_length_buf[env_ids].float()) / float(env.max_episode_length)
    bad_orientation_rate = _termination_rate(env, env_ids, "bad_orientation")
    base_height_rate = _termination_rate(env, env_ids, "base_height")
    return bool(
        tracking_reward > min_tracking_reward
        and episode_length_ratio > min_episode_length_ratio
        and bad_orientation_rate < max_bad_orientation_rate
        and base_height_rate < max_base_height_rate
    )


def _log_once(env: ManagerBasedRLEnv, key: str, message: str) -> None:
    last_value = getattr(env, "_g0_curriculum_log_cache", {})
    if last_value.get(key) == message:
        return
    print(message)
    last_value[key] = message
    env._g0_curriculum_log_cache = last_value


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    min_tracking_reward: float = 0.70,
    min_episode_length_ratio: float = 0.70,
    max_bad_orientation_rate: float = 0.08,
    max_base_height_rate: float = 0.02,
) -> torch.Tensor:
    """Expand linear velocity command range based on tracking performance.

    This curriculum reads the current command range from the `base_velocity`
    command term. If the linear velocity tracking reward is good enough, it
    gradually expands lin_vel_x and lin_vel_y until reaching `limit_ranges`.

    The command term must be configured with UniformLevelVelocityCommandCfg,
    because this function requires both:
    - cfg.ranges
    - cfg.limit_ranges
    """

    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    # Only update once per full episode horizon.
    if env.common_step_counter % env.max_episode_length == 0:
        if _stable_enough(
            env,
            env_ids,
            reward_term_name,
            min_tracking_reward,
            min_episode_length_ratio,
            max_bad_orientation_rate,
            max_base_height_rate,
        ):
            old_x = tuple(ranges.lin_vel_x)
            old_y = tuple(ranges.lin_vel_y)
            delta_command = torch.tensor([-0.05, 0.05], device=env.device)

            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()

            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()
            _log_once(
                env,
                "lin_vel_cmd_levels",
                "[G0 curriculum] lin/y command range "
                f"x {old_x} -> {tuple(ranges.lin_vel_x)}, y {old_y} -> {tuple(ranges.lin_vel_y)}",
            )

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
    min_tracking_reward: float = 0.68,
    min_episode_length_ratio: float = 0.70,
    max_bad_orientation_rate: float = 0.08,
    max_base_height_rate: float = 0.02,
) -> torch.Tensor:
    """Expand angular velocity command range based on tracking performance.

    This curriculum gradually expands the yaw velocity command range when
    angular velocity tracking is good enough.
    """

    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    # Only update once per full episode horizon.
    if env.common_step_counter % env.max_episode_length == 0:
        if _stable_enough(
            env,
            env_ids,
            reward_term_name,
            min_tracking_reward,
            min_episode_length_ratio,
            max_bad_orientation_rate,
            max_base_height_rate,
        ):
            old_z = tuple(ranges.ang_vel_z)
            delta_command = torch.tensor([-0.025, 0.025], device=env.device)

            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()
            _log_once(
                env,
                "ang_vel_cmd_levels",
                f"[G0 curriculum] yaw command range {old_z} -> {tuple(ranges.ang_vel_z)}",
            )

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def event_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
    max_level: int = 3,
    min_tracking_reward: float = 0.72,
    min_episode_length_ratio: float = 0.75,
    max_bad_orientation_rate: float = 0.05,
    max_base_height_rate: float = 0.01,
) -> dict[str, float]:
    """Stage event difficulty once the policy is stable enough."""

    level = int(getattr(env, "_g0_event_curriculum_level", 0))
    if env.common_step_counter % env.max_episode_length != 0:
        return {"level": float(level)}

    if level < max_level and _stable_enough(
        env,
        env_ids,
        reward_term_name,
        min_tracking_reward,
        min_episode_length_ratio,
        max_bad_orientation_rate,
        max_base_height_rate,
    ):
        level += 1
        env._g0_event_curriculum_level = level

    friction_by_level = {
        0: ((0.8, 1.0), (0.8, 1.0)),
        1: ((0.7, 1.1), (0.7, 1.1)),
        2: ((0.6, 1.15), (0.6, 1.15)),
        3: ((0.5, 1.2), (0.5, 1.2)),
    }
    push_by_level = {
        0: (-0.0, 0.0),
        1: (-0.05, 0.05),
        2: (-0.10, 0.10),
        3: (-0.20, 0.20),
    }
    reset_pos_by_level = {
        0: (0.0, 0.0),
        1: (-0.02, 0.02),
        2: (-0.03, 0.03),
        3: (-0.04, 0.04),
    }
    reset_vel_by_level = {
        0: (0.0, 0.0),
        1: (-0.20, 0.20),
        2: (-0.30, 0.30),
        3: (-0.40, 0.40),
    }

    static_range, dynamic_range = friction_by_level[level]
    try:
        friction_cfg = env.event_manager.get_term_cfg("physics_material")
        friction_cfg.params["static_friction_range"] = static_range
        friction_cfg.params["dynamic_friction_range"] = dynamic_range
        friction_term = friction_cfg.func
        if hasattr(friction_term, "material_buckets"):
            ranges = torch.tensor([static_range, dynamic_range, (0.0, 0.0)], device="cpu")
            buckets = torch.empty_like(friction_term.material_buckets)
            buckets[:, 0] = torch.empty(buckets.shape[0]).uniform_(ranges[0, 0], ranges[0, 1])
            buckets[:, 1] = torch.empty(buckets.shape[0]).uniform_(ranges[1, 0], ranges[1, 1])
            buckets[:, 2] = 0.0
            buckets[:, 1] = torch.min(buckets[:, 0], buckets[:, 1])
            friction_term.material_buckets = buckets
    except ValueError:
        pass

    try:
        push_cfg = env.event_manager.get_term_cfg("push_robot")
        push_range = push_by_level[level]
        push_cfg.params["velocity_range"]["x"] = push_range
        push_cfg.params["velocity_range"]["y"] = push_range
    except ValueError:
        pass

    try:
        reset_cfg = env.event_manager.get_term_cfg("reset_robot_joints")
        reset_cfg.params["joint_position_ranges"][".*"] = reset_pos_by_level[level]
        reset_cfg.params["joint_velocity_ranges"][".*"] = reset_vel_by_level[level]
    except ValueError:
        pass

    _log_once(
        env,
        "event_levels",
        "[G0 curriculum] event level "
        f"{level}: friction={static_range}, push_xy={push_by_level[level]}, "
        f"joint_pos={reset_pos_by_level[level]}, joint_vel={reset_vel_by_level[level]}",
    )
    return {
        "level": float(level),
        "friction_min": float(static_range[0]),
        "friction_max": float(static_range[1]),
        "push_xy_max": float(push_by_level[level][1]),
        "reset_joint_pos_max": float(reset_pos_by_level[level][1]),
    }
