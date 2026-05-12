#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import math
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from isaaclab.app import AppLauncher

LOG_PATH = Path("logs/sweep_zero_action_standing_pose.txt")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.write_text("", encoding="utf-8")


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()


parser = argparse.ArgumentParser(description="Sweep G0 zero-action standing poses under fixed debug conditions.")
parser.add_argument("--task", type=str, default="G0-Velocity-v0")
parser.add_argument("--short_steps", type=int, default=120)
parser.add_argument("--long_steps", type=int, default=500)
parser.add_argument("--top_k", type=int, default=5)
parser.add_argument("--root_z", type=float, default=0.235)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


HIP_VALUES = [0.200, 0.205, 0.210, 0.215, 0.220]
KNEE_VALUES = [0.340, 0.345, 0.350, 0.355, 0.360]
ANKLE_VALUES = [0.145, 0.150, 0.155]

PITCH_JOINT_GROUPS = {
    "hip": ["l_hip_pitch_joint", "r_hip_pitch_joint"],
    "knee": ["l_knee_pitch_joint", "r_knee_pitch_joint"],
    "ankle": ["l_ankle_pitch_joint", "r_ankle_pitch_joint"],
}


@dataclass(frozen=True)
class Candidate:
    hip: float
    knee: float
    ankle: float


@dataclass
class CandidateResult:
    candidate: Candidate
    survive_steps: int
    done_step: int | None
    done_reason: str
    max_abs_root_pitch_deg: float
    final_root_pitch_deg: float
    pitch_slope_deg_per_step: float
    min_root_z: float
    max_ankle_torque_ratio: float | None
    max_knee_torque_ratio: float | None
    max_hip_torque_ratio: float | None
    max_foot_force_imbalance: float | None
    initial_root_yaw_deg: float


def _reset_env(env: Any) -> Any:
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def _step_env(env: Any, action: torch.Tensor):
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


def _quat_wxyz_to_euler_deg(quat_wxyz: torch.Tensor) -> tuple[float, float, float]:
    quat = quat_wxyz.detach().cpu().tolist()
    w, x, y, z = [float(v) for v in quat]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def _as_env0(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "ndim") and value.ndim > 0:
        value = value[0]
    if hasattr(value, "cpu"):
        value = value.cpu()
    return value


def _get_env0_joint_scalar(value: Any, joint_id: int) -> float | None:
    value = _as_env0(value)
    if value is None:
        return None
    value = value[joint_id]
    if hasattr(value, "ndim") and value.ndim > 0:
        return None
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _get_robot_joint_names(robot: Any) -> list[str]:
    joint_names = getattr(robot.data, "joint_names", None)
    if joint_names is None:
        joint_names = getattr(robot, "joint_names", None)
    if joint_names is None:
        raise RuntimeError("Cannot find robot joint names.")
    return [str(name) for name in joint_names]


def _resolve_joint_ids(joint_names: list[str], requested_names: list[str]) -> dict[str, int]:
    joint_ids: dict[str, int] = {}
    for requested_name in requested_names:
        joint_ids[requested_name] = joint_names.index(requested_name)
    return joint_ids


def _get_term_diagnostics(base_env: Any, env_index: int = 0) -> list[tuple[str, bool]]:
    termination_manager = getattr(base_env, "termination_manager", None)
    if termination_manager is None:
        return [("termination_manager_unavailable", False)]

    if hasattr(termination_manager, "get_active_iterable_terms"):
        try:
            terms = termination_manager.get_active_iterable_terms(env_index)
            return [(str(name), bool(float(values[0]))) for name, values in terms]
        except Exception as exc:
            return [(f"get_active_iterable_terms_failed:{exc.__class__.__name__}", False)]

    active_terms = getattr(termination_manager, "active_terms", None)
    if active_terms is None:
        active_terms = getattr(termination_manager, "_term_names", None)
    if active_terms is None:
        return [("termination_terms_unavailable", False)]

    diagnostics: list[tuple[str, bool]] = []
    for term_name in active_terms:
        try:
            value = termination_manager.get_term(term_name)
            diagnostics.append((str(term_name), bool(value[env_index].item())))
        except Exception as exc:
            diagnostics.append((f"{term_name}:unavailable:{exc.__class__.__name__}", False))
    return diagnostics


def _done_reason(base_env: Any, terminated0: bool, truncated0: bool) -> str:
    active = [name for name, value in _get_term_diagnostics(base_env, env_index=0) if value]
    if active:
        return "+".join(active)
    if truncated0:
        return "truncated"
    if terminated0:
        return "terminated"
    return "alive"


def _safe_ratio(value: float | None, limit: float | None) -> float | None:
    if value is None or limit is None or abs(limit) <= 0.0:
        return None
    return abs(value) / abs(limit)


def _get_torque_ratios(robot: Any, group_joint_ids: dict[str, dict[str, int]]) -> dict[str, float | None]:
    data = robot.data
    applied_torque = getattr(data, "applied_torque", None)
    computed_torque = getattr(data, "computed_torque", None)
    effort_limit = getattr(data, "joint_effort_limits", None)
    if effort_limit is None:
        effort_limit = getattr(data, "default_joint_effort_limits", None)

    ratios: dict[str, float | None] = {}
    for group_name, joint_ids in group_joint_ids.items():
        max_ratio: float | None = None
        for joint_id in joint_ids.values():
            applied = _get_env0_joint_scalar(applied_torque, joint_id)
            computed = _get_env0_joint_scalar(computed_torque, joint_id)
            torque = applied
            if torque is None and computed is not None:
                torque = computed
            elif torque is not None and computed is not None and abs(torque) == 0.0 and abs(computed) > 0.0:
                torque = computed
            ratio = _safe_ratio(torque, _get_env0_joint_scalar(effort_limit, joint_id))
            if ratio is not None:
                max_ratio = ratio if max_ratio is None else max(max_ratio, ratio)
        ratios[group_name] = max_ratio
    return ratios


def _get_foot_force_imbalance(base_env: Any) -> float | None:
    try:
        contact_sensor = base_env.scene["contact_forces"]
        body_ids, body_names = contact_sensor.find_bodies(["l_foot_link", "r_foot_link"], preserve_order=True)
        forces = getattr(contact_sensor.data, "net_forces_w", None)
    except Exception:
        return None

    if forces is None or len(body_ids) != 2:
        return None

    force_by_name = {
        str(name): forces[0, int(body_id)].detach().cpu()
        for body_id, name in zip(body_ids, body_names)
    }
    if "l_foot_link" not in force_by_name or "r_foot_link" not in force_by_name:
        return None

    left_z = abs(float(force_by_name["l_foot_link"][2].item()))
    right_z = abs(float(force_by_name["r_foot_link"][2].item()))
    total = left_z + right_z
    if total <= 1.0e-8:
        return 0.0
    return abs(left_z - right_z) / total


def _pitch_slope_deg_per_step(pitches: list[float]) -> float:
    if len(pitches) < 2:
        return 0.0
    n = len(pitches)
    x_mean = (n - 1) / 2.0
    y_mean = sum(pitches) / n
    numerator = sum((i - x_mean) * (pitch - y_mean) for i, pitch in enumerate(pitches))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _format_float(value: float | None, width: int = 8, precision: int = 3) -> str:
    if value is None:
        return f"{'n/a':>{width}s}"
    return f"{value:{width}.{precision}f}"


def _candidate_joint_pos(candidate: Candidate) -> dict[str, float]:
    hip = candidate.hip
    knee = candidate.knee
    ankle = candidate.ankle
    return {
        "waist_yaw_joint": 0.0,
        "waist_roll_joint": 0.0,
        "l_hip_pitch_joint": -hip,
        "l_hip_roll_joint": 0.0,
        "l_hip_yaw_joint": 0.0,
        "l_knee_pitch_joint": -knee,
        "l_ankle_pitch_joint": ankle,
        "l_ankle_roll_joint": 0.0,
        "r_hip_pitch_joint": hip,
        "r_hip_roll_joint": 0.0,
        "r_hip_yaw_joint": 0.0,
        "r_knee_pitch_joint": knee,
        "r_ankle_pitch_joint": -ankle,
        "r_ankle_roll_joint": 0.0,
        "l_shoulder_pitch_joint": -0.30,
        "l_shoulder_roll_joint": -0.25,
        "l_shoulder_yaw_joint": 0.0,
        "l_elbow_pitch_joint": 0.97,
        "r_shoulder_pitch_joint": 0.30,
        "r_shoulder_roll_joint": 0.25,
        "r_shoulder_yaw_joint": 0.0,
        "r_elbow_pitch_joint": -0.97,
    }


def _make_env_cfg(candidate: Candidate):
    from g0_robot_lab.tasks.locomotion.robots.g0.velocity_env_cfg import G0RobotLabEnvCfg

    env_cfg = G0RobotLabEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, args_cli.root_z)
    env_cfg.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    env_cfg.scene.robot.init_state.lin_vel = (0.0, 0.0, 0.0)
    env_cfg.scene.robot.init_state.ang_vel = (0.0, 0.0, 0.0)
    env_cfg.scene.robot.init_state.joint_pos = _candidate_joint_pos(candidate)
    env_cfg.scene.robot.init_state.joint_vel = {".*": 0.0}

    if getattr(args_cli, "device", None) is not None:
        env_cfg.sim.device = args_cli.device

    env_cfg.events.physics_material = None
    env_cfg.events.base_external_force_torque = None
    env_cfg.events.reset_base = None
    env_cfg.events.reset_robot_joints = None
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "add_base_mass"):
        env_cfg.events.add_base_mass = None

    env_cfg.rewards.undesired_contacts = None

    env_cfg.curriculum.lin_vel_cmd_levels = None
    env_cfg.commands.base_velocity.rel_standing_envs = 1.0
    env_cfg.commands.base_velocity.rel_heading_envs = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1.0e9, 1.0e9)
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.limit_ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.limit_ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.limit_ranges.ang_vel_z = (0.0, 0.0)

    env_cfg.observations.policy.enable_corruption = False

    return env_cfg


def _run_candidate(candidate: Candidate, steps: int) -> CandidateResult:
    import gymnasium as gym

    env_cfg = _make_env_cfg(candidate)
    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        _reset_env(env)
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        action_dim = base_env.action_manager.total_action_dim
        zero_action = torch.zeros((1, action_dim), device=base_env.device, dtype=torch.float32)

        joint_names = _get_robot_joint_names(robot)
        group_joint_ids = {
            group_name: _resolve_joint_ids(joint_names, names)
            for group_name, names in PITCH_JOINT_GROUPS.items()
        }

        _, initial_pitch_deg, initial_yaw_deg = _quat_wxyz_to_euler_deg(robot.data.root_quat_w[0])
        root_z = float(robot.data.root_pos_w[0, 2].item())
        pitches = [initial_pitch_deg]
        min_root_z = root_z
        max_abs_pitch = abs(initial_pitch_deg)
        max_ratios = {"hip": None, "knee": None, "ankle": None}
        max_foot_imbalance: float | None = None
        done_step: int | None = None
        done_reason = "alive"
        survive_steps = 0

        for step in range(steps):
            _, _, done, terminated, truncated, _ = _step_env(env, zero_action)

            root_z = float(robot.data.root_pos_w[0, 2].item())
            _, pitch_deg, _ = _quat_wxyz_to_euler_deg(robot.data.root_quat_w[0])
            done0 = bool(done[0].item())
            terminated0 = bool(terminated[0].item())
            truncated0 = bool(truncated[0].item())

            pitches.append(pitch_deg)
            min_root_z = min(min_root_z, root_z)
            max_abs_pitch = max(max_abs_pitch, abs(pitch_deg))

            ratios = _get_torque_ratios(robot, group_joint_ids)
            for group_name, ratio in ratios.items():
                if ratio is not None:
                    current = max_ratios[group_name]
                    max_ratios[group_name] = ratio if current is None else max(current, ratio)

            imbalance = _get_foot_force_imbalance(base_env)
            if imbalance is not None:
                max_foot_imbalance = imbalance if max_foot_imbalance is None else max(max_foot_imbalance, imbalance)

            survive_steps = step + 1

            if done0:
                done_step = step
                done_reason = _done_reason(base_env, terminated0, truncated0)
                break

        return CandidateResult(
            candidate=candidate,
            survive_steps=survive_steps,
            done_step=done_step,
            done_reason=done_reason,
            max_abs_root_pitch_deg=max_abs_pitch,
            final_root_pitch_deg=pitches[-1],
            pitch_slope_deg_per_step=_pitch_slope_deg_per_step(pitches),
            min_root_z=min_root_z,
            max_ankle_torque_ratio=max_ratios["ankle"],
            max_knee_torque_ratio=max_ratios["knee"],
            max_hip_torque_ratio=max_ratios["hip"],
            max_foot_force_imbalance=max_foot_imbalance,
            initial_root_yaw_deg=initial_yaw_deg,
        )
    finally:
        env.close()


def _result_sort_key(result: CandidateResult) -> tuple[float, float, float, float, float]:
    ankle_ratio = result.max_ankle_torque_ratio
    ankle_sort = ankle_ratio if ankle_ratio is not None else float("inf")
    return (
        -result.survive_steps,
        abs(result.final_root_pitch_deg),
        abs(result.pitch_slope_deg_per_step),
        -result.min_root_z,
        ankle_sort,
    )


def _print_results(title: str, results: list[CandidateResult], limit: int = 10) -> None:
    log("")
    log("=" * 118)
    log(title)
    log("=" * 118)
    log(
        "rank  hip    knee   ankle  survive  done  final_pitch  slope     min_z   "
        "max_ankle_ratio  reason"
    )
    for rank, result in enumerate(sorted(results, key=_result_sort_key)[:limit], start=1):
        done_text = "-" if result.done_step is None else str(result.done_step)
        log(
            f"{rank:<5d} "
            f"{result.candidate.hip:0.3f}  "
            f"{result.candidate.knee:0.3f}  "
            f"{result.candidate.ankle:0.3f}  "
            f"{result.survive_steps:<7d} "
            f"{done_text:<5s} "
            f"{result.final_root_pitch_deg:11.3f} "
            f"{result.pitch_slope_deg_per_step:8.4f} "
            f"{result.min_root_z:7.3f} "
            f"{_format_float(result.max_ankle_torque_ratio, width=15, precision=3)}  "
            f"{result.done_reason}"
        )


def _print_detail(title: str, result: CandidateResult) -> None:
    log("")
    log(title)
    log(
        f"candidate hip={result.candidate.hip:.3f} knee={result.candidate.knee:.3f} ankle={result.candidate.ankle:.3f} "
        f"survive_steps={result.survive_steps} done_step={result.done_step} reason={result.done_reason}"
    )
    log(
        f"initial_root_yaw_deg={result.initial_root_yaw_deg:.3f} "
        f"final_root_pitch_deg={result.final_root_pitch_deg:.3f} "
        f"max_abs_root_pitch_deg={result.max_abs_root_pitch_deg:.3f} "
        f"pitch_slope_deg_per_step={result.pitch_slope_deg_per_step:.5f} "
        f"min_root_z={result.min_root_z:.4f}"
    )
    log(
        f"max_hip_torque_ratio={_format_float(result.max_hip_torque_ratio)} "
        f"max_knee_torque_ratio={_format_float(result.max_knee_torque_ratio)} "
        f"max_ankle_torque_ratio={_format_float(result.max_ankle_torque_ratio)} "
        f"max_foot_force_imbalance={_format_float(result.max_foot_force_imbalance)}"
    )


def main() -> None:
    import g0_robot_lab.tasks  # noqa: F401

    candidates = [Candidate(*values) for values in itertools.product(HIP_VALUES, KNEE_VALUES, ANKLE_VALUES)]

    log("")
    log("=" * 118)
    log("G0 ZERO-ACTION STANDING POSE SWEEP")
    log("=" * 118)
    log("reset yaw fixed: True")
    log("root initial position: (0.0, 0.0, %.3f)" % args_cli.root_z)
    log("root initial rotation: (1.0, 0.0, 0.0, 0.0)")
    log("reset randomization disabled: True")
    log("undesired_contacts disabled: True")
    log("command randomization disabled: True")
    log("num_envs: 1")
    log(f"steps per candidate: {args_cli.short_steps}")
    log(f"long steps for top_k: {args_cli.long_steps}")
    log(f"top_k: {args_cli.top_k}")
    log(f"candidate count: {len(candidates)}")
    log(f"log path: {LOG_PATH}")

    short_results: list[CandidateResult] = []
    for index, candidate in enumerate(candidates, start=1):
        result = _run_candidate(candidate, args_cli.short_steps)
        short_results.append(result)
        log(
            f"[short {index:02d}/{len(candidates):02d}] "
            f"hip={candidate.hip:.3f} knee={candidate.knee:.3f} ankle={candidate.ankle:.3f} "
            f"survive={result.survive_steps:3d} final_pitch={result.final_root_pitch_deg:8.3f} "
            f"slope={result.pitch_slope_deg_per_step:8.4f} min_z={result.min_root_z:7.3f} "
            f"ankle_ratio={_format_float(result.max_ankle_torque_ratio, width=6, precision=3)} "
            f"reason={result.done_reason}"
        )

    _print_results("SHORT SWEEP TOP 10", short_results, limit=10)

    top_candidates = [
        result.candidate
        for result in sorted(short_results, key=_result_sort_key)
        if result.survive_steps >= args_cli.short_steps
    ][: args_cli.top_k]

    if not top_candidates:
        log("")
        log("[WARN] No candidate survived the short sweep. Running long test on the best short candidates anyway.")
        top_candidates = [result.candidate for result in sorted(short_results, key=_result_sort_key)[: args_cli.top_k]]

    long_results: list[CandidateResult] = []
    for index, candidate in enumerate(top_candidates, start=1):
        result = _run_candidate(candidate, args_cli.long_steps)
        long_results.append(result)
        log(
            f"[long {index:02d}/{len(top_candidates):02d}] "
            f"hip={candidate.hip:.3f} knee={candidate.knee:.3f} ankle={candidate.ankle:.3f} "
            f"survive={result.survive_steps:3d} final_pitch={result.final_root_pitch_deg:8.3f} "
            f"slope={result.pitch_slope_deg_per_step:8.4f} min_z={result.min_root_z:7.3f} "
            f"ankle_ratio={_format_float(result.max_ankle_torque_ratio, width=6, precision=3)} "
            f"reason={result.done_reason}"
        )

    _print_results("LONG SWEEP RESULTS", long_results, limit=max(10, args_cli.top_k))

    best = sorted(long_results, key=_result_sort_key)[0] if long_results else sorted(short_results, key=_result_sort_key)[0]
    _print_detail("BEST CANDIDATE", best)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("")
        log("=" * 118)
        log("[SWEEP-DBG] ERROR")
        log("=" * 118)
        log(traceback.format_exc())
    finally:
        simulation_app.close()
