#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import math
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from isaaclab.app import AppLauncher

LOG_PATH = Path("logs/sweep_zero_action_standing_pose.txt")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
LOG_PATH.write_text("", encoding="utf-8")
TRACE_PATH = Path("logs") / f"zero_action_torque_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


def log(msg: str) -> None:
    print(msg, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")
        f.flush()


parser = argparse.ArgumentParser(description="Sweep G0 zero-action standing poses under fixed debug conditions.")
parser.add_argument("--task", type=str, default="G0-Velocity-v0")
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--top_k", type=int, default=10)
parser.add_argument("--root_z", type=float, default=0.23)
parser.add_argument("--mode", choices=["fixed", "sweep"], default="fixed")
parser.add_argument("--hip", type=float, default=0.20)
parser.add_argument("--knee", type=float, default=0.35)
parser.add_argument("--ankle", type=float, default=0.14)
parser.add_argument("--print-torque-every", type=int, default=10)
parser.add_argument("--effort-scale", type=float, default=1.0)
parser.add_argument(
    "--disable-terminations",
    action="store_true",
    help="Debug only: keep simulating after base_height/bad_orientation would trigger, while still logging them.",
)
parser.add_argument("--hip_pitch_stiffness", type=float, default=None)
parser.add_argument("--hip_pitch_damping", type=float, default=None)
parser.add_argument("--knee_pitch_stiffness", type=float, default=None)
parser.add_argument("--knee_pitch_damping", type=float, default=None)
parser.add_argument("--ankle_pitch_stiffness", type=float, default=None)
parser.add_argument("--ankle_pitch_damping", type=float, default=None)
parser.add_argument("--root_z_values", type=str, default="0.22,0.23,0.24")
parser.add_argument("--hip_values", type=str, default="0.18,0.19,0.20,0.21")
parser.add_argument("--knee_values", type=str, default="0.33,0.34,0.35,0.36")
parser.add_argument(
    "--ankle_values",
    type=str,
    default="0.13,0.14,0.15,0.16",
)
parser.add_argument(
    "--candidate-set",
    choices=["grid", "quick"],
    default="grid",
    help="Use full root/hip/knee/ankle grid or the six quick two-decimal candidates from the debug plan.",
)
parser.add_argument("--max_candidates", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()
if args_cli.effort_scale <= 0.0:
    raise ValueError("--effort-scale must be positive.")
if args_cli.effort_scale > 2.0:
    raise ValueError("--effort-scale above 2.0 is intentionally blocked for this debug script.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def _parse_float_list(raw_values: str) -> list[float]:
    values = []
    for raw_value in raw_values.split(","):
        raw_value = raw_value.strip()
        if raw_value:
            values.append(float(raw_value))
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


ROOT_Z_VALUES = _parse_float_list(args_cli.root_z_values)
HIP_VALUES = _parse_float_list(args_cli.hip_values)
KNEE_VALUES = _parse_float_list(args_cli.knee_values)
ANKLE_VALUES = _parse_float_list(args_cli.ankle_values)

PITCH_JOINT_GROUPS = {
    "hip": ["l_hip_pitch_joint", "r_hip_pitch_joint"],
    "knee": ["l_knee_pitch_joint", "r_knee_pitch_joint"],
    "ankle": ["l_ankle_pitch_joint", "r_ankle_pitch_joint"],
}


@dataclass(frozen=True)
class Candidate:
    root_z: float
    hip: float
    knee: float
    ankle: float


@dataclass
class JointTorqueSample:
    joint_name: str
    joint_pos: float | None
    joint_vel: float | None
    applied_torque: float | None
    computed_torque: float | None
    torque: float | None
    effort_limit: float | None
    torque_ratio: float | None
    saturated: bool


@dataclass
class CandidateResult:
    candidate: Candidate
    survive_steps: int
    done_step: int | None
    done_reason: str
    final_root_z: float
    max_abs_root_pitch_deg: float
    final_root_pitch_deg: float
    pitch_slope_deg_per_step: float
    min_root_z: float
    max_ankle_torque_ratio: float | None
    max_knee_torque_ratio: float | None
    max_hip_torque_ratio: float | None
    max_foot_force_imbalance: float | None
    final_left_foot_force_z: float | None
    final_right_foot_force_z: float | None
    initial_left_foot_height: float | None
    initial_right_foot_height: float | None
    final_left_foot_height: float | None
    final_right_foot_height: float | None
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


def _debug_done_reason(robot: Any) -> str:
    reasons = []
    try:
        if float(robot.data.root_pos_w[0, 2].item()) < 0.12:
            reasons.append("base_height")
    except Exception:
        pass
    try:
        angle = torch.acos(-robot.data.projected_gravity_b[0, 2]).abs()
        if float(angle.item()) > 1.2:
            reasons.append("bad_orientation")
    except Exception:
        pass
    return "+".join(reasons) if reasons else "alive"


def _get_whole_body_com(robot: Any) -> tuple[float, float, float] | None:
    try:
        masses = robot.root_physx_view.get_masses().to(robot.data.body_com_pos_w.device)
        if masses.ndim == 1:
            masses = masses.unsqueeze(0)
        total_mass = torch.sum(masses[0])
        if float(total_mass.item()) <= 1.0e-8:
            return None
        com = torch.sum(robot.data.body_com_pos_w[0] * masses[0].unsqueeze(-1), dim=0) / total_mass
        return tuple(float(value) for value in com.detach().cpu().tolist())
    except Exception:
        return None


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


def _sample_joint_torques(robot: Any, joint_names: list[str]) -> list[JointTorqueSample]:
    data = robot.data
    joint_pos = getattr(data, "joint_pos", None)
    joint_vel = getattr(data, "joint_vel", None)
    applied_torque = getattr(data, "applied_torque", None)
    computed_torque = getattr(data, "computed_torque", None)
    effort_limit = getattr(data, "joint_effort_limits", None)
    if effort_limit is None:
        effort_limit = getattr(data, "default_joint_effort_limits", None)

    samples: list[JointTorqueSample] = []
    for joint_id, joint_name in enumerate(joint_names):
        applied = _get_env0_joint_scalar(applied_torque, joint_id)
        computed = _get_env0_joint_scalar(computed_torque, joint_id)
        torque = applied
        if torque is None and computed is not None:
            torque = computed
        elif torque is not None and computed is not None and abs(torque) == 0.0 and abs(computed) > 0.0:
            torque = computed
        limit = _get_env0_joint_scalar(effort_limit, joint_id)
        ratio = _safe_ratio(torque, limit)
        samples.append(
            JointTorqueSample(
                joint_name=joint_name,
                joint_pos=_get_env0_joint_scalar(joint_pos, joint_id),
                joint_vel=_get_env0_joint_scalar(joint_vel, joint_id),
                applied_torque=applied,
                computed_torque=computed,
                torque=torque,
                effort_limit=limit,
                torque_ratio=ratio,
                saturated=bool(ratio is not None and ratio > 0.90),
            )
        )
    return samples


def _torque_sort_value(sample: JointTorqueSample) -> float:
    if sample.torque_ratio is not None:
        return sample.torque_ratio
    if sample.torque is not None:
        return abs(sample.torque)
    return -1.0


def _format_optional(value: float | None, precision: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


class TorqueTraceLogger:
    def __init__(self, path: Path, joint_names: list[str]):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            [
                "candidate_index",
                "candidate_root_z",
                "candidate_hip_pitch",
                "candidate_knee_pitch",
                "candidate_ankle_pitch",
                "effort_scale",
                "step",
                "joint_name",
                "joint_pos",
                "joint_vel",
                "applied_torque",
                "computed_torque",
                "torque",
                "effort_limit",
                "torque_ratio",
                "saturated_gt_0p90",
                "root_z",
                "root_pitch_deg",
                "termination_reason",
            ]
        )
        self.file.flush()
        self.joint_names = joint_names

    def close(self) -> None:
        self.file.close()

    def write_step(
        self,
        candidate_index: int,
        candidate: Candidate,
        step: int,
        root_z: float,
        root_pitch_deg: float,
        termination_reason: str,
        samples: list[JointTorqueSample],
    ) -> None:
        for sample in samples:
            self.writer.writerow(
                [
                    candidate_index,
                    candidate.root_z,
                    candidate.hip,
                    candidate.knee,
                    candidate.ankle,
                    args_cli.effort_scale,
                    step,
                    sample.joint_name,
                    sample.joint_pos,
                    sample.joint_vel,
                    sample.applied_torque,
                    sample.computed_torque,
                    sample.torque,
                    sample.effort_limit,
                    sample.torque_ratio,
                    int(sample.saturated),
                    root_z,
                    root_pitch_deg,
                    termination_reason,
                ]
            )
        self.file.flush()


def _print_torque_summary(
    step: int,
    root_z: float,
    pitch_deg: float,
    samples: list[JointTorqueSample],
    roll_deg: float | None = None,
    com_w: tuple[float, float, float] | None = None,
    foot_force_z: tuple[float, float] | None = None,
    foot_heights: tuple[float, float] | None = None,
    termination_reason: str = "alive",
) -> None:
    line = f"[step {step}] root_z={root_z:.3f} pitch={pitch_deg:.2f} deg"
    if roll_deg is not None:
        line += f" roll={roll_deg:.2f} deg"
    if termination_reason != "alive":
        line += f" termination={termination_reason}"
    log(line)
    if com_w is not None:
        log(f"COM projection: x={com_w[0]:.4f} y={com_w[1]:.4f} z={com_w[2]:.4f}")
    if foot_force_z is not None:
        log(f"foot force z: left={foot_force_z[0]:.3f} right={foot_force_z[1]:.3f}")
    if foot_heights is not None:
        log(f"foot height: left={foot_heights[0]:.4f} right={foot_heights[1]:.4f}")
    log("top saturated joints:")
    for sample in sorted(samples, key=_torque_sort_value, reverse=True)[:8]:
        log(
            "  "
            f"{sample.joint_name:24s} "
            f"torque={_format_optional(sample.torque)} "
            f"limit={_format_optional(sample.effort_limit)} "
            f"ratio={_format_optional(sample.torque_ratio)} "
            f"q={_format_optional(sample.joint_pos)} "
            f"dq={_format_optional(sample.joint_vel)}"
        )


class TorqueSaturationStats:
    def __init__(self, joint_names: list[str]):
        self.total_steps = 0
        self.max_abs_torque = {name: 0.0 for name in joint_names}
        self.max_ratio = {name: 0.0 for name in joint_names}
        self.saturated_steps = {name: 0 for name in joint_names}
        self.first_saturation_step: dict[str, int | None] = {name: None for name in joint_names}
        self.last_twenty: list[tuple[int, list[JointTorqueSample]]] = []

    def update(self, step: int, samples: list[JointTorqueSample]) -> None:
        self.total_steps += 1
        self.last_twenty.append((step, samples))
        if len(self.last_twenty) > 20:
            self.last_twenty.pop(0)

        for sample in samples:
            if sample.torque is not None:
                self.max_abs_torque[sample.joint_name] = max(self.max_abs_torque[sample.joint_name], abs(sample.torque))
            if sample.torque_ratio is not None:
                self.max_ratio[sample.joint_name] = max(self.max_ratio[sample.joint_name], sample.torque_ratio)
                if sample.torque_ratio > 0.90:
                    self.saturated_steps[sample.joint_name] += 1
                    if self.first_saturation_step[sample.joint_name] is None:
                        self.first_saturation_step[sample.joint_name] = step

    def print_summary(self, candidate: Candidate) -> None:
        log("")
        log(
            "TORQUE SATURATION SUMMARY "
            f"root_z={candidate.root_z:.2f} hip={candidate.hip:.2f} knee={candidate.knee:.2f} "
            f"ankle={candidate.ankle:.2f} effort_scale={args_cli.effort_scale:.2f}"
        )
        log("joint_name                 max_abs_torque  max_ratio  sat_steps  sat_fraction  first_sat_step")
        for joint_name in sorted(self.max_ratio, key=lambda name: self.max_ratio[name], reverse=True):
            sat_steps = self.saturated_steps[joint_name]
            sat_fraction = sat_steps / max(self.total_steps, 1)
            first_step = self.first_saturation_step[joint_name]
            log(
                f"{joint_name:26s} "
                f"{self.max_abs_torque[joint_name]:14.4f} "
                f"{self.max_ratio[joint_name]:10.4f} "
                f"{sat_steps:9d} "
                f"{sat_fraction:12.3f} "
                f"{str(first_step):>14s}"
            )

        first_saturations = [
            (step, name)
            for name, step in self.first_saturation_step.items()
            if step is not None
        ]
        if first_saturations:
            first_step, first_joint = sorted(first_saturations)[0]
            log(f"earliest ratio>0.90 joint: {first_joint} at step {first_step}")
        else:
            log("earliest ratio>0.90 joint: none")

        last_window_max: dict[str, float] = {}
        for _, samples in self.last_twenty:
            for sample in samples:
                if sample.torque_ratio is not None:
                    last_window_max[sample.joint_name] = max(last_window_max.get(sample.joint_name, 0.0), sample.torque_ratio)
        log("termination-window top torque ratios:")
        for joint_name, ratio in sorted(last_window_max.items(), key=lambda item: item[1], reverse=True)[:8]:
            log(f"  {joint_name:24s} ratio={ratio:.4f}")


def _get_foot_force_z(base_env: Any) -> tuple[float, float] | None:
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
    return left_z, right_z


def _get_foot_force_imbalance(base_env: Any) -> float | None:
    foot_force_z = _get_foot_force_z(base_env)
    if foot_force_z is None:
        return None

    left_z, right_z = foot_force_z
    total = left_z + right_z
    if total <= 1.0e-8:
        return 0.0
    return abs(left_z - right_z) / total


def _get_foot_heights(robot: Any) -> tuple[float, float] | None:
    try:
        if hasattr(robot, "find_bodies"):
            body_ids, body_names = robot.find_bodies(["l_foot_link", "r_foot_link"], preserve_order=True)
        else:
            body_names = list(getattr(robot, "body_names", getattr(robot.data, "body_names", [])))
            body_ids = [body_names.index(name) for name in ("l_foot_link", "r_foot_link")]
            body_names = ["l_foot_link", "r_foot_link"]
        body_pos_w = robot.data.body_pos_w
    except Exception:
        return None

    if body_pos_w is None or len(body_ids) != 2:
        return None

    height_by_name = {
        str(name): float(body_pos_w[0, int(body_id), 2].detach().cpu().item())
        for body_id, name in zip(body_ids, body_names)
    }
    if "l_foot_link" not in height_by_name or "r_foot_link" not in height_by_name:
        return None
    return height_by_name["l_foot_link"], height_by_name["r_foot_link"]


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


def _scale_effort_limit(value: Any, scale: float) -> Any:
    if isinstance(value, dict):
        return {key: _scale_effort_limit(item, scale) for key, item in value.items()}
    if isinstance(value, (int, float)):
        return value * scale
    return value


def _make_env_cfg():
    from g0_robot_lab.tasks.locomotion.robots.g0.velocity_env_cfg import G0RobotLabEnvCfg

    env_cfg = G0RobotLabEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, args_cli.root_z)
    env_cfg.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    env_cfg.scene.robot.init_state.lin_vel = (0.0, 0.0, 0.0)
    env_cfg.scene.robot.init_state.ang_vel = (0.0, 0.0, 0.0)
    env_cfg.scene.robot.init_state.joint_pos = _candidate_joint_pos(
        Candidate(args_cli.root_z, args_cli.hip, args_cli.knee, args_cli.ankle)
    )
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

    if args_cli.disable_terminations:
        env_cfg.terminations.base_height = None
        env_cfg.terminations.bad_orientation = None

    standard_servos = env_cfg.scene.robot.actuators["standard_servos"]
    right_angle_servos = env_cfg.scene.robot.actuators["right_angle_servos"]
    if args_cli.effort_scale != 1.0:
        for actuator_cfg in env_cfg.scene.robot.actuators.values():
            actuator_cfg.effort_limit_sim = _scale_effort_limit(actuator_cfg.effort_limit_sim, args_cli.effort_scale)
    if args_cli.hip_pitch_stiffness is not None:
        standard_servos.stiffness[".*_hip_pitch_joint"] = args_cli.hip_pitch_stiffness
    if args_cli.hip_pitch_damping is not None:
        standard_servos.damping[".*_hip_pitch_joint"] = args_cli.hip_pitch_damping
    if args_cli.knee_pitch_stiffness is not None:
        right_angle_servos.stiffness[".*_knee_pitch_joint"] = args_cli.knee_pitch_stiffness
    if args_cli.knee_pitch_damping is not None:
        right_angle_servos.damping[".*_knee_pitch_joint"] = args_cli.knee_pitch_damping
    if args_cli.ankle_pitch_stiffness is not None:
        right_angle_servos.stiffness[".*_ankle_pitch_joint"] = args_cli.ankle_pitch_stiffness
    if args_cli.ankle_pitch_damping is not None:
        right_angle_servos.damping[".*_ankle_pitch_joint"] = args_cli.ankle_pitch_damping

    return env_cfg


def _candidate_joint_tensor(robot: Any, joint_name_to_id: dict[str, int], candidate: Candidate) -> torch.Tensor:
    joint_pos = robot.data.default_joint_pos.clone()
    for joint_name, joint_value in _candidate_joint_pos(candidate).items():
        joint_pos[:, joint_name_to_id[joint_name]] = joint_value
    return joint_pos


def _call_with_optional_env_ids(func: Any, *args: Any, env_ids: torch.Tensor) -> None:
    try:
        func(*args, env_ids=env_ids)
    except TypeError:
        func(*args)


def _iter_action_terms(action_manager: Any):
    terms = getattr(action_manager, "_terms", None)
    if terms is None:
        terms = getattr(action_manager, "terms", None)
    if callable(terms):
        terms = terms()
    if isinstance(terms, dict):
        return terms.items()
    return []


def _set_candidate_action_offsets(
    base_env: Any,
    robot: Any,
    joint_name_to_id: dict[str, int],
    candidate: Candidate,
) -> None:
    candidate_joint_pos = _candidate_joint_pos(candidate)
    robot_joint_names = _get_robot_joint_names(robot)

    for _, term in _iter_action_terms(base_env.action_manager):
        joint_names = getattr(term, "_joint_names", None)
        if joint_names is None:
            joint_names = getattr(term, "joint_names", None)

        joint_ids = getattr(term, "_joint_ids", None)
        if joint_ids is None:
            joint_ids = getattr(term, "joint_ids", None)

        if joint_names is None:
            if isinstance(joint_ids, slice):
                joint_names = robot_joint_names[joint_ids]
            elif joint_ids is not None:
                joint_names = [robot_joint_names[int(joint_id)] for joint_id in joint_ids]
            else:
                continue
        else:
            joint_names = [str(joint_name) for joint_name in joint_names]

        for offset_attr in ("_offset", "offset", "_default_offset"):
            offset = getattr(term, offset_attr, None)
            if offset is None or not hasattr(offset, "shape"):
                continue
            if len(offset.shape) < 2:
                continue
            for action_id, joint_name in enumerate(joint_names):
                if joint_name in candidate_joint_pos and action_id < offset.shape[1]:
                    offset[:, action_id] = candidate_joint_pos[joint_name]

        for action_attr in ("_raw_actions", "_processed_actions", "raw_actions", "processed_actions"):
            action = getattr(term, action_attr, None)
            if action is not None and hasattr(action, "zero_"):
                action.zero_()

    # Keep articulation defaults aligned too. Some action/reward paths read these
    # buffers directly after reset; this script is a fixed-condition debug tool.
    default_joint_pos = robot.data.default_joint_pos
    for joint_name, joint_value in candidate_joint_pos.items():
        default_joint_pos[:, joint_name_to_id[joint_name]] = joint_value


def _reset_candidate_state(
    base_env: Any,
    robot: Any,
    joint_name_to_id: dict[str, int],
    candidate: Candidate,
) -> None:
    env_ids = torch.tensor([0], dtype=torch.long, device=base_env.device)

    root_pose = robot.data.default_root_state[:, :7].clone()
    root_pose[:, 0:3] = torch.tensor((0.0, 0.0, candidate.root_z), device=base_env.device, dtype=root_pose.dtype)
    root_pose[:, 3:7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=base_env.device, dtype=root_pose.dtype)

    root_velocity = torch.zeros_like(robot.data.default_root_state[:, 7:])
    joint_pos = _candidate_joint_tensor(robot, joint_name_to_id, candidate)
    joint_vel = torch.zeros_like(robot.data.default_joint_vel)

    if hasattr(base_env.scene, "reset"):
        base_env.scene.reset(env_ids)

    _call_with_optional_env_ids(robot.write_root_pose_to_sim, root_pose, env_ids=env_ids)
    _call_with_optional_env_ids(robot.write_root_velocity_to_sim, root_velocity, env_ids=env_ids)
    _call_with_optional_env_ids(robot.write_joint_state_to_sim, joint_pos, joint_vel, env_ids=env_ids)
    _call_with_optional_env_ids(robot.set_joint_position_target, joint_pos, env_ids=env_ids)

    base_env.scene.write_data_to_sim()

    if hasattr(base_env, "episode_length_buf"):
        base_env.episode_length_buf[env_ids] = 0
    if hasattr(base_env, "reset_buf"):
        base_env.reset_buf[env_ids] = False
    if hasattr(base_env, "reset_terminated"):
        base_env.reset_terminated[env_ids] = False
    if hasattr(base_env, "reset_time_outs"):
        base_env.reset_time_outs[env_ids] = False
    if hasattr(base_env, "action_manager") and hasattr(base_env.action_manager, "reset"):
        base_env.action_manager.reset(env_ids)
    if hasattr(base_env, "command_manager") and hasattr(base_env.command_manager, "reset"):
        base_env.command_manager.reset(env_ids)
    if hasattr(base_env, "termination_manager") and hasattr(base_env.termination_manager, "reset"):
        base_env.termination_manager.reset(env_ids)
    _set_candidate_action_offsets(base_env, robot, joint_name_to_id, candidate)
    _call_with_optional_env_ids(robot.set_joint_position_target, joint_pos, env_ids=env_ids)

    # Refresh asset/sensor buffers after writing the new state without advancing one RL step.
    update_dt = getattr(base_env, "physics_dt", None)
    if update_dt is None:
        update_dt = getattr(base_env, "step_dt", 0.0)
    base_env.scene.update(update_dt)


def _invalid_candidate_result(candidate: Candidate, initial_yaw_deg: float, reason: str) -> CandidateResult:
    return CandidateResult(
        candidate=candidate,
        survive_steps=0,
        done_step=0,
        done_reason=reason,
        final_root_z=0.0,
        max_abs_root_pitch_deg=float("inf"),
        final_root_pitch_deg=float("inf"),
        pitch_slope_deg_per_step=float("inf"),
        min_root_z=0.0,
        max_ankle_torque_ratio=None,
        max_knee_torque_ratio=None,
        max_hip_torque_ratio=None,
        max_foot_force_imbalance=None,
        final_left_foot_force_z=None,
        final_right_foot_force_z=None,
        initial_left_foot_height=None,
        initial_right_foot_height=None,
        final_left_foot_height=None,
        final_right_foot_height=None,
        initial_root_yaw_deg=initial_yaw_deg,
    )


def _run_candidate(
    env: Any,
    base_env: Any,
    robot: Any,
    joint_name_to_id: dict[str, int],
    group_joint_ids: dict[str, dict[str, int]],
    zero_action: torch.Tensor,
    candidate: Candidate,
    candidate_index: int,
    torque_trace: TorqueTraceLogger,
    steps: int,
) -> CandidateResult:
    _reset_candidate_state(base_env, robot, joint_name_to_id, candidate)

    _, initial_pitch_deg, initial_yaw_deg = _quat_wxyz_to_euler_deg(robot.data.root_quat_w[0])
    if abs(initial_yaw_deg) > 1.0:
        return _invalid_candidate_result(candidate, initial_yaw_deg, "invalid_initial_yaw")

    root_z = float(robot.data.root_pos_w[0, 2].item())
    initial_foot_heights = _get_foot_heights(robot)
    initial_left_foot_height: float | None = None
    initial_right_foot_height: float | None = None
    if initial_foot_heights is not None:
        initial_left_foot_height, initial_right_foot_height = initial_foot_heights

    pitches = [initial_pitch_deg]
    min_root_z = root_z
    max_abs_pitch = abs(initial_pitch_deg)
    max_ratios = {"hip": None, "knee": None, "ankle": None}
    max_foot_imbalance: float | None = None
    final_left_foot_force_z: float | None = None
    final_right_foot_force_z: float | None = None
    final_left_foot_height: float | None = initial_left_foot_height
    final_right_foot_height: float | None = initial_right_foot_height
    done_step: int | None = None
    done_reason = "alive"
    survive_steps = 0
    torque_stats = TorqueSaturationStats(_get_robot_joint_names(robot))

    for step in range(steps):
        _, _, done, terminated, truncated, _ = _step_env(env, zero_action)

        root_z = float(robot.data.root_pos_w[0, 2].item())
        roll_deg, pitch_deg, _ = _quat_wxyz_to_euler_deg(robot.data.root_quat_w[0])
        done0 = bool(done[0].item())
        terminated0 = bool(terminated[0].item())
        truncated0 = bool(truncated[0].item())
        if args_cli.disable_terminations:
            step_done_reason = _debug_done_reason(robot)
            if step_done_reason != "alive" and done_step is None:
                done_step = step
                done_reason = step_done_reason
            done0 = False
        else:
            step_done_reason = _done_reason(base_env, terminated0, truncated0) if done0 else "alive"

        pitches.append(pitch_deg)
        min_root_z = min(min_root_z, root_z)
        max_abs_pitch = max(max_abs_pitch, abs(pitch_deg))
        torque_samples = _sample_joint_torques(robot, _get_robot_joint_names(robot))
        torque_stats.update(step, torque_samples)
        torque_trace.write_step(
            candidate_index=candidate_index,
            candidate=candidate,
            step=step,
            root_z=root_z,
            root_pitch_deg=pitch_deg,
            termination_reason=step_done_reason,
            samples=torque_samples,
        )
        foot_force_z = _get_foot_force_z(base_env)
        foot_heights = _get_foot_heights(robot)
        if args_cli.print_torque_every > 0 and (step % args_cli.print_torque_every == 0 or done0):
            _print_torque_summary(
                step,
                root_z,
                pitch_deg,
                torque_samples,
                roll_deg=roll_deg,
                com_w=_get_whole_body_com(robot) if args_cli.disable_terminations else None,
                foot_force_z=foot_force_z,
                foot_heights=foot_heights,
                termination_reason=step_done_reason,
            )

        ratios = _get_torque_ratios(robot, group_joint_ids)
        for group_name, ratio in ratios.items():
            if ratio is not None:
                current = max_ratios[group_name]
                max_ratios[group_name] = ratio if current is None else max(current, ratio)

        imbalance = _get_foot_force_imbalance(base_env)
        if imbalance is not None:
            max_foot_imbalance = imbalance if max_foot_imbalance is None else max(max_foot_imbalance, imbalance)
        if foot_force_z is not None:
            final_left_foot_force_z, final_right_foot_force_z = foot_force_z
        if foot_heights is not None:
            final_left_foot_height, final_right_foot_height = foot_heights

        survive_steps = step + 1

        if done0:
            done_step = step
            done_reason = step_done_reason
            break

    torque_stats.print_summary(candidate)

    return CandidateResult(
        candidate=candidate,
        survive_steps=survive_steps,
        done_step=done_step,
        done_reason=done_reason,
        final_root_z=root_z,
        max_abs_root_pitch_deg=max_abs_pitch,
        final_root_pitch_deg=pitches[-1],
        pitch_slope_deg_per_step=_pitch_slope_deg_per_step(pitches),
        min_root_z=min_root_z,
        max_ankle_torque_ratio=max_ratios["ankle"],
        max_knee_torque_ratio=max_ratios["knee"],
        max_hip_torque_ratio=max_ratios["hip"],
        max_foot_force_imbalance=max_foot_imbalance,
        final_left_foot_force_z=final_left_foot_force_z,
        final_right_foot_force_z=final_right_foot_force_z,
        initial_left_foot_height=initial_left_foot_height,
        initial_right_foot_height=initial_right_foot_height,
        final_left_foot_height=final_left_foot_height,
        final_right_foot_height=final_right_foot_height,
        initial_root_yaw_deg=initial_yaw_deg,
    )


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
    log("=" * 178)
    log(title)
    log("=" * 178)
    log(
        "rank  root_z  hip    knee   ankle  survive  reason         final_z  min_z   final_pitch  max_abs_pitch  "
        "slope     hip_ratio  knee_ratio  ankle_ratio  foot_imb  l_force  r_force  l_h0   r_h0   l_hf   r_hf"
    )
    for rank, result in enumerate(sorted(results, key=_result_sort_key)[:limit], start=1):
        log(
            f"{rank:<5d} "
            f"{result.candidate.root_z:0.2f}    "
            f"{result.candidate.hip:0.2f}   "
            f"{result.candidate.knee:0.2f}   "
            f"{result.candidate.ankle:0.2f}   "
            f"{result.survive_steps:<7d} "
            f"{result.done_reason[:13]:13s} "
            f"{result.final_root_z:7.3f} "
            f"{result.min_root_z:7.3f} "
            f"{result.final_root_pitch_deg:11.3f} "
            f"{result.max_abs_root_pitch_deg:13.3f} "
            f"{result.pitch_slope_deg_per_step:8.4f} "
            f"{_format_float(result.max_hip_torque_ratio, width=9, precision=3)} "
            f"{_format_float(result.max_knee_torque_ratio, width=10, precision=3)} "
            f"{_format_float(result.max_ankle_torque_ratio, width=11, precision=3)} "
            f"{_format_float(result.max_foot_force_imbalance, width=8, precision=3)} "
            f"{_format_float(result.final_left_foot_force_z, width=8, precision=2)} "
            f"{_format_float(result.final_right_foot_force_z, width=8, precision=2)} "
            f"{_format_float(result.initial_left_foot_height, width=6, precision=3)} "
            f"{_format_float(result.initial_right_foot_height, width=6, precision=3)} "
            f"{_format_float(result.final_left_foot_height, width=6, precision=3)} "
            f"{_format_float(result.final_right_foot_height, width=6, precision=3)}"
        )


def _print_detail(title: str, result: CandidateResult) -> None:
    log("")
    log(title)
    log(
        f"candidate root_z={result.candidate.root_z:.2f} "
        f"hip={result.candidate.hip:.2f} knee={result.candidate.knee:.2f} ankle={result.candidate.ankle:.2f} "
        f"survive_steps={result.survive_steps} done_step={result.done_step} reason={result.done_reason}"
    )
    log(
        f"initial_root_yaw_deg={result.initial_root_yaw_deg:.3f} "
        f"final_root_z={result.final_root_z:.4f} "
        f"final_root_pitch_deg={result.final_root_pitch_deg:.3f} "
        f"max_abs_root_pitch_deg={result.max_abs_root_pitch_deg:.3f} "
        f"pitch_slope_deg_per_step={result.pitch_slope_deg_per_step:.5f} "
        f"min_root_z={result.min_root_z:.4f}"
    )
    log(
        f"max_hip_torque_ratio={_format_float(result.max_hip_torque_ratio)} "
        f"max_knee_torque_ratio={_format_float(result.max_knee_torque_ratio)} "
        f"max_ankle_torque_ratio={_format_float(result.max_ankle_torque_ratio)} "
        f"max_foot_force_imbalance={_format_float(result.max_foot_force_imbalance)} "
        f"final_l_foot_z={_format_float(result.final_left_foot_force_z)} "
        f"final_r_foot_z={_format_float(result.final_right_foot_force_z)}"
    )
    log(
        f"initial_l_foot_height={_format_float(result.initial_left_foot_height)} "
        f"initial_r_foot_height={_format_float(result.initial_right_foot_height)} "
        f"final_l_foot_height={_format_float(result.final_left_foot_height)} "
        f"final_r_foot_height={_format_float(result.final_right_foot_height)}"
    )


def main() -> None:
    import gymnasium as gym
    import g0_robot_lab.tasks  # noqa: F401

    if args_cli.mode == "fixed":
        candidates = [Candidate(args_cli.root_z, args_cli.hip, args_cli.knee, args_cli.ankle)]
    elif args_cli.candidate_set == "quick":
        candidates = [
            Candidate(0.23, 0.20, 0.35, 0.14),
            Candidate(0.23, 0.20, 0.35, 0.15),
            Candidate(0.23, 0.19, 0.35, 0.14),
            Candidate(0.23, 0.20, 0.34, 0.14),
            Candidate(0.24, 0.20, 0.35, 0.14),
            Candidate(0.22, 0.20, 0.35, 0.14),
        ]
    else:
        candidates = [
            Candidate(*values)
            for values in itertools.product(ROOT_Z_VALUES, HIP_VALUES, KNEE_VALUES, ANKLE_VALUES)
        ]

    if args_cli.max_candidates is not None:
        candidates = candidates[: max(args_cli.max_candidates, 0)]
    steps_per_candidate = max(args_cli.steps, 500)

    log("")
    log("=" * 178)
    log("G0 ZERO-ACTION STANDING POSE LONG SWEEP")
    log("=" * 178)
    log("reset yaw fixed: True")
    log("root initial position: (0.0, 0.0, %.3f)" % args_cli.root_z)
    log("root initial rotation: (1.0, 0.0, 0.0, 0.0)")
    log("reset randomization disabled: True")
    log("undesired_contacts disabled: True")
    log("command randomization disabled: True")
    log("num_envs: 1")
    log(f"steps per candidate: {steps_per_candidate}")
    log(f"mode: {args_cli.mode}")
    log(f"candidate_set: {args_cli.candidate_set}")
    log(f"fixed candidate: root_z={args_cli.root_z} hip={args_cli.hip} knee={args_cli.knee} ankle={args_cli.ankle}")
    log(f"effort_scale: {args_cli.effort_scale}")
    log(f"disable_terminations: {args_cli.disable_terminations}")
    log(f"print_torque_every: {args_cli.print_torque_every}")
    log(
        "pd overrides: "
        f"hip=({args_cli.hip_pitch_stiffness},{args_cli.hip_pitch_damping}) "
        f"knee=({args_cli.knee_pitch_stiffness},{args_cli.knee_pitch_damping}) "
        f"ankle=({args_cli.ankle_pitch_stiffness},{args_cli.ankle_pitch_damping})"
    )
    log(f"top_k: {args_cli.top_k}")
    log(f"max_candidates: {args_cli.max_candidates}")
    log(f"candidate count: {len(candidates)}")
    log(f"root_z values: {ROOT_Z_VALUES}")
    log(f"hip values: {HIP_VALUES}")
    log(f"knee values: {KNEE_VALUES}")
    log(f"ankle values: {ANKLE_VALUES}")
    log(f"log path: {LOG_PATH}")
    log(f"torque trace csv path: {TRACE_PATH}")

    if not candidates:
        log("[WARN] No candidates selected.")
        return

    env_cfg = _make_env_cfg()
    env = gym.make(args_cli.task, cfg=env_cfg)

    try:
        _reset_env(env)
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        action_dim = base_env.action_manager.total_action_dim
        zero_action = torch.zeros((1, action_dim), device=base_env.device, dtype=torch.float32)

        joint_names = _get_robot_joint_names(robot)
        torque_trace = TorqueTraceLogger(TRACE_PATH, joint_names)
        joint_name_to_id = {joint_name: joint_id for joint_id, joint_name in enumerate(joint_names)}
        group_joint_ids = {
            group_name: _resolve_joint_ids(joint_names, names)
            for group_name, names in PITCH_JOINT_GROUPS.items()
        }

        results: list[CandidateResult] = []
        try:
            for index, candidate in enumerate(candidates, start=1):
                result = _run_candidate(
                    env=env,
                    base_env=base_env,
                    robot=robot,
                    joint_name_to_id=joint_name_to_id,
                    group_joint_ids=group_joint_ids,
                    zero_action=zero_action,
                    candidate=candidate,
                    candidate_index=index,
                    torque_trace=torque_trace,
                    steps=steps_per_candidate,
                )
                results.append(result)
                log(
                    f"[sweep {index:03d}/{len(candidates):03d}] "
                    f"root_z={candidate.root_z:.2f} hip={candidate.hip:.2f} knee={candidate.knee:.2f} ankle={candidate.ankle:.2f} "
                    f"yaw0={result.initial_root_yaw_deg:7.3f} "
                    f"survive={result.survive_steps:3d} final_z={result.final_root_z:7.3f} "
                    f"min_z={result.min_root_z:7.3f} final_pitch={result.final_root_pitch_deg:8.3f} "
                    f"max_abs_pitch={result.max_abs_root_pitch_deg:8.3f} "
                    f"slope={result.pitch_slope_deg_per_step:8.4f} "
                    f"hip_ratio={_format_float(result.max_hip_torque_ratio, width=6, precision=3)} "
                    f"knee_ratio={_format_float(result.max_knee_torque_ratio, width=6, precision=3)} "
                    f"ankle_ratio={_format_float(result.max_ankle_torque_ratio, width=6, precision=3)} "
                    f"foot_imb={_format_float(result.max_foot_force_imbalance, width=6, precision=3)} "
                    f"l_foot_z={_format_float(result.final_left_foot_force_z, width=7, precision=2)} "
                    f"r_foot_z={_format_float(result.final_right_foot_force_z, width=7, precision=2)} "
                    f"l_h0={_format_float(result.initial_left_foot_height, width=6, precision=3)} "
                    f"r_h0={_format_float(result.initial_right_foot_height, width=6, precision=3)} "
                    f"l_hf={_format_float(result.final_left_foot_height, width=6, precision=3)} "
                    f"r_hf={_format_float(result.final_right_foot_height, width=6, precision=3)} "
                    f"reason={result.done_reason}"
                )
        finally:
            torque_trace.close()

        _print_results("SWEEP TOP CANDIDATES", results, limit=args_cli.top_k)

        best = sorted(results, key=_result_sort_key)[0]
        _print_detail("BEST CANDIDATE", best)
    finally:
        env.close()


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
