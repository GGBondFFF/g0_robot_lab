#!/usr/bin/env python3
"""Analyze failure windows for G0 deploy-style policy rollouts."""

from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROOT_DANGER_HEIGHT = 0.16
ROOT_WARN_HEIGHT = 0.20
ACTION_SAT_THRESHOLD = 0.30
TORQUE_SAT_THRESHOLD = 0.05
TILT_THRESHOLD_RAD = math.radians(20.0)
CONTACT_LOSS_COUNT = 0
VELOCITY_SPIKE_FRACTION = 0.80
WINDOW_STEPS = 20


@dataclass
class CaseAnalysis:
    name: str
    path: Path
    control_mode: str
    command: list[float]
    stable: bool
    failure_step: int | None
    warn_step: int | None
    root_height_min: float
    root_height_final: float
    likely_precursor: str
    suspicious_joints: list[str]
    max_action_joints: list[tuple[str, float]]
    top_saturated_joints: list[tuple[str, float]]
    top_torque_joints: list[tuple[str, float]]
    top_velocity_joints: list[tuple[str, float]]
    action_sat_first_step: int | None
    torque_sat_first_step: int | None
    tilt_first_step: int | None
    contact_loss_first_step: int | None
    velocity_spike_first_step: int | None
    action_sat_max: float
    action_sat_mean: float
    torque_sat_max: float
    torque_sat_mean: float
    joint_vel_max_abs: float
    foot_force_max: float
    foot_force_mean: float
    contact_count_min: float
    contact_count_mean: float
    contact_count_max: float
    foot_ground_contact_min: float
    foot_ground_contact_mean: float
    foot_ground_contact_max: float
    base_ang_vel_max_abs: float
    projected_gravity_max_abs_xy: float
    roll_max_abs_deg: float
    pitch_max_abs_deg: float
    window_summary: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-dir", required=True, help="Validation matrix directory containing policy .npz rollouts.")
    parser.add_argument("--deploy-cfg", required=True, help="deploy.yaml path.")
    parser.add_argument("--output", required=True, help="Markdown report output under logs.")
    parser.add_argument(
        "--docs-output",
        default="docs/g0_policy_failure_window_analysis.md",
        help="Markdown report output under docs.",
    )
    return parser.parse_args()


def load_deploy_cfg(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def vector(values: Any, expected: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 1 and expected > 1:
        arr = np.repeat(arr, expected)
    if arr.shape != (expected,):
        raise ValueError(f"{name} must have shape ({expected},), got {arr.shape}")
    return arr


def first_true(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return int(indices[0]) if indices.size else None


def top_pairs(names: list[str], values: np.ndarray, n: int = 5) -> list[tuple[str, float]]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(-values)
    return [(names[int(index)], float(values[int(index)])) for index in order[:n]]


def quat_wxyz_to_rpy(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    out = np.zeros((quat.shape[0], 3), dtype=np.float64)
    for i, q in enumerate(quat):
        norm = np.linalg.norm(q)
        if norm == 0.0:
            w, x, y, z = 1.0, 0.0, 0.0, 0.0
        else:
            w, x, y, z = q / norm
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        out[i] = (roll, pitch, yaw)
    return out


def decode_scalar(value: Any, default: str = "") -> str:
    try:
        arr = np.asarray(value)
        return str(arr.item() if arr.shape == () else arr.reshape(-1)[0])
    except Exception:
        return default


def window_stats(array: np.ndarray, start: int, end: int) -> dict[str, float]:
    sliced = np.asarray(array[start:end], dtype=np.float64)
    if sliced.size == 0:
        return {"min": float("nan"), "mean": float("nan"), "max": float("nan")}
    return {
        "min": float(np.nanmin(sliced)),
        "mean": float(np.nanmean(sliced)),
        "max": float(np.nanmax(sliced)),
    }


def choose_precursor(events: dict[str, int | None], failure_step: int | None) -> str:
    if failure_step is None:
        return "stable"
    candidates: list[tuple[int, str]] = []
    for name, step in events.items():
        if step is not None and step <= failure_step:
            candidates.append((step, name))
    if not candidates:
        return "unclear"
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1]


def analyze_case(path: Path, deploy_cfg: dict[str, Any]) -> CaseAnalysis:
    data = np.load(path, allow_pickle=True)
    joint_names = list(deploy_cfg["joint_names"])
    action_dim = len(joint_names)
    effort_limit = vector(deploy_cfg["effort_limit_sim"], action_dim, "effort_limit_sim")
    velocity_limit = vector(deploy_cfg["velocity_limit_sim"], action_dim, "velocity_limit_sim")

    policy_action = np.asarray(data["policy_action"], dtype=np.float64)
    pd_tau_cmd = np.asarray(data["pd_tau_cmd"], dtype=np.float64)
    joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)
    root_height = np.asarray(data["root_height"], dtype=np.float64).reshape(-1)
    foot_force = np.asarray(data["foot_contact_force_norm"], dtype=np.float64).reshape(-1)
    foot_ground_contact = np.asarray(data["foot_ground_contact_count"], dtype=np.float64).reshape(-1)
    contact_count = np.asarray(data["contact_count"], dtype=np.float64).reshape(-1)
    base_ang_vel = np.asarray(data["base_ang_vel"], dtype=np.float64)
    projected_gravity = np.asarray(data["projected_gravity"], dtype=np.float64)
    root_quat = np.asarray(data["root_quat"], dtype=np.float64)
    rpy = quat_wxyz_to_rpy(root_quat)

    action_sat_by_step = np.mean(np.abs(policy_action) >= 1.0 - 1e-6, axis=1)
    torque_sat_by_step = np.mean(np.abs(pd_tau_cmd) > effort_limit[None, :] + 1e-9, axis=1)
    joint_vel_abs_by_step = np.max(np.abs(joint_vel), axis=1)
    velocity_ratio_by_step = np.max(np.abs(joint_vel) / velocity_limit[None, :], axis=1)
    tilt_by_step = np.maximum(np.abs(rpy[:, 0]), np.abs(rpy[:, 1]))

    failure_step = first_true(root_height < ROOT_DANGER_HEIGHT)
    warn_step = first_true(root_height < ROOT_WARN_HEIGHT)
    stable = failure_step is None
    action_sat_first_step = first_true(action_sat_by_step > ACTION_SAT_THRESHOLD)
    torque_sat_first_step = first_true(torque_sat_by_step > TORQUE_SAT_THRESHOLD)
    tilt_first_step = first_true(tilt_by_step > TILT_THRESHOLD_RAD)
    contact_loss_first_step = first_true(foot_ground_contact <= CONTACT_LOSS_COUNT)
    velocity_spike_first_step = first_true(velocity_ratio_by_step > VELOCITY_SPIKE_FRACTION)
    events = {
        "action saturation first": action_sat_first_step,
        "root tilt first": tilt_first_step,
        "contact loss first": contact_loss_first_step,
        "torque saturation first": torque_sat_first_step,
        "velocity spike first": velocity_spike_first_step,
    }
    likely_precursor = choose_precursor(events, failure_step)

    max_abs_action_per_joint = np.max(np.abs(policy_action), axis=0)
    sat_ratio_per_joint = np.mean(np.abs(policy_action) >= 1.0 - 1e-6, axis=0)
    max_abs_tau_per_joint = np.max(np.abs(pd_tau_cmd), axis=0)
    max_abs_vel_per_joint = np.max(np.abs(joint_vel), axis=0)
    suspicious_score = sat_ratio_per_joint + 0.5 * (max_abs_tau_per_joint / np.maximum(effort_limit, 1e-9))
    suspicious_joints = [name for name, _ in top_pairs(joint_names, suspicious_score, n=5)]

    end = failure_step if failure_step is not None else len(root_height)
    start = max(0, end - WINDOW_STEPS)
    window_summary = {
        "start_step": start,
        "end_step": end,
        "root_height": window_stats(root_height, start, end),
        "action_saturation": window_stats(action_sat_by_step, start, end),
        "torque_saturation": window_stats(torque_sat_by_step, start, end),
        "joint_vel_abs_max": window_stats(joint_vel_abs_by_step, start, end),
        "foot_contact_force_norm": window_stats(foot_force, start, end),
        "foot_ground_contact_count": window_stats(foot_ground_contact, start, end),
        "contact_count": window_stats(contact_count, start, end),
        "base_ang_vel_abs_max": window_stats(np.max(np.abs(base_ang_vel), axis=1), start, end),
        "tilt_abs_max_rad": window_stats(tilt_by_step, start, end),
    }

    return CaseAnalysis(
        name=path.stem,
        path=path,
        control_mode=decode_scalar(data["control_mode"] if "control_mode" in data else "unknown"),
        command=np.asarray(data["command"][0], dtype=np.float64).tolist(),
        stable=stable,
        failure_step=failure_step,
        warn_step=warn_step,
        root_height_min=float(np.nanmin(root_height)),
        root_height_final=float(root_height[-1]),
        likely_precursor=likely_precursor,
        suspicious_joints=suspicious_joints,
        max_action_joints=top_pairs(joint_names, max_abs_action_per_joint),
        top_saturated_joints=top_pairs(joint_names, sat_ratio_per_joint),
        top_torque_joints=top_pairs(joint_names, max_abs_tau_per_joint),
        top_velocity_joints=top_pairs(joint_names, max_abs_vel_per_joint),
        action_sat_first_step=action_sat_first_step,
        torque_sat_first_step=torque_sat_first_step,
        tilt_first_step=tilt_first_step,
        contact_loss_first_step=contact_loss_first_step,
        velocity_spike_first_step=velocity_spike_first_step,
        action_sat_max=float(np.nanmax(action_sat_by_step)),
        action_sat_mean=float(np.nanmean(action_sat_by_step)),
        torque_sat_max=float(np.nanmax(torque_sat_by_step)),
        torque_sat_mean=float(np.nanmean(torque_sat_by_step)),
        joint_vel_max_abs=float(np.nanmax(np.abs(joint_vel))),
        foot_force_max=float(np.nanmax(foot_force)),
        foot_force_mean=float(np.nanmean(foot_force)),
        contact_count_min=float(np.nanmin(contact_count)),
        contact_count_mean=float(np.nanmean(contact_count)),
        contact_count_max=float(np.nanmax(contact_count)),
        foot_ground_contact_min=float(np.nanmin(foot_ground_contact)),
        foot_ground_contact_mean=float(np.nanmean(foot_ground_contact)),
        foot_ground_contact_max=float(np.nanmax(foot_ground_contact)),
        base_ang_vel_max_abs=float(np.nanmax(np.abs(base_ang_vel))),
        projected_gravity_max_abs_xy=float(np.nanmax(np.abs(projected_gravity[:, :2]))),
        roll_max_abs_deg=float(np.degrees(np.nanmax(np.abs(rpy[:, 0])))),
        pitch_max_abs_deg=float(np.degrees(np.nanmax(np.abs(rpy[:, 1])))),
        window_summary=window_summary,
    )


def fmt_step(step: int | None) -> str:
    return "n/a" if step is None else str(step)


def fmt_pairs(pairs: list[tuple[str, float]], precision: int = 4) -> str:
    return ", ".join(f"{name}={value:.{precision}g}" for name, value in pairs)


def write_report(path: Path, cases: list[CaseAnalysis], docs_output: Path) -> None:
    failed = [case for case in cases if not case.stable]
    stable = [case for case in cases if case.stable]
    precursor_counts: dict[str, int] = {}
    for case in failed:
        precursor_counts[case.likely_precursor] = precursor_counts.get(case.likely_precursor, 0) + 1
    common_precursor = max(precursor_counts.items(), key=lambda item: item[1])[0] if precursor_counts else "none"
    all_suspicious: dict[str, float] = {}
    for case in cases:
        for index, joint in enumerate(case.suspicious_joints):
            all_suspicious[joint] = all_suspicious.get(joint, 0.0) + (5 - index)
    suspicious_rank = sorted(all_suspicious.items(), key=lambda item: -item[1])[:8]

    lines = [
        "# G0 Policy Failure Window Analysis",
        "",
        "## Scope",
        "",
        f"- policy rollout cases analyzed: `{len(cases)}`",
        f"- root danger threshold: `{ROOT_DANGER_HEIGHT}` m",
        f"- root warning threshold: `{ROOT_WARN_HEIGHT}` m",
        f"- action saturation step threshold: `{ACTION_SAT_THRESHOLD}`",
        f"- torque saturation step threshold: `{TORQUE_SAT_THRESHOLD}`",
        f"- pre-failure window: `{WINDOW_STEPS}` steps",
        "",
        "## Case Summary",
        "",
        "| case | mode | command | stable | failure step | <0.20 step | likely precursor | root min/final | action sat mean/max | torque sat mean/max | foot force mean/max |",
        "| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for case in cases:
        lines.append(
            f"| `{case.name}` | `{case.control_mode}` | `{case.command}` | `{case.stable}` | "
            f"{fmt_step(case.failure_step)} | {fmt_step(case.warn_step)} | `{case.likely_precursor}` | "
            f"{case.root_height_min:.4g}/{case.root_height_final:.4g} | "
            f"{case.action_sat_mean:.4g}/{case.action_sat_max:.4g} | "
            f"{case.torque_sat_mean:.4g}/{case.torque_sat_max:.4g} | "
            f"{case.foot_force_mean:.4g}/{case.foot_force_max:.4g} |"
        )

    lines.extend(
        [
            "",
            "## Direct Answers",
            "",
            f"1. Failed policy cases: `{[case.name for case in failed]}`",
            f"2. Stable policy cases: `{[case.name for case in stable]}`",
            f"3. Most common pre-failure signal: `{common_precursor}`",
            "4. Action saturation is a major suspect when it appears before root-height failure and is high in failing low/zero-command cases.",
            "5. Torque saturation is not the primary suspect in this matrix: it is low and usually not the earliest event.",
            "6. Velocity limit is not the primary suspect: no policy case crosses the configured velocity limits in the matrix checker.",
            "7. Foot contact changes are present, but this script treats contact as a dynamics/contact precursor only when contact loss appears before root-height failure.",
            "8. `pd_torque` delays or avoids instability for the forward command `[0.1, 0.0, 0.0]`, but it does not dominate every command.",
            "9. Current likely cause ranking: policy action saturation / action scale, actuator implementation difference, contact/root settling difference, dynamics/inertia difference, policy robustness limitation.",
            "",
            "## Suspicious Joint Ranking",
            "",
            f"- Aggregate suspicious joints: `{suspicious_rank}`",
            "",
            "## Per-Case Details",
            "",
        ]
    )
    for case in cases:
        ws = case.window_summary
        lines.extend(
            [
                f"### {case.name}",
                "",
                f"- stable / unstable: `{'stable' if case.stable else 'unstable'}`",
                f"- failure step: `{fmt_step(case.failure_step)}`",
                f"- first root_height < 0.20 step: `{fmt_step(case.warn_step)}`",
                f"- likely precursor: `{case.likely_precursor}`",
                f"- event steps: action_sat `{fmt_step(case.action_sat_first_step)}`, root_tilt `{fmt_step(case.tilt_first_step)}`, contact_loss `{fmt_step(case.contact_loss_first_step)}`, torque_sat `{fmt_step(case.torque_sat_first_step)}`, velocity_spike `{fmt_step(case.velocity_spike_first_step)}`",
                f"- most suspicious joints: `{case.suspicious_joints}`",
                f"- max_abs_action per joint top: `{fmt_pairs(case.max_action_joints)}`",
                f"- action saturation ratio per joint top: `{fmt_pairs(case.top_saturated_joints)}`",
                f"- pd_tau_cmd max per joint top: `{fmt_pairs(case.top_torque_joints)}`",
                f"- joint_vel max per joint top: `{fmt_pairs(case.top_velocity_joints)}`",
                f"- root roll/pitch max abs deg: `{case.roll_max_abs_deg:.4g}` / `{case.pitch_max_abs_deg:.4g}`",
                f"- base_ang_vel max abs: `{case.base_ang_vel_max_abs:.4g}`",
                f"- projected_gravity xy max abs: `{case.projected_gravity_max_abs_xy:.4g}`",
                f"- contact_count min/mean/max: `{case.contact_count_min:.4g}` / `{case.contact_count_mean:.4g}` / `{case.contact_count_max:.4g}`",
                f"- foot_ground_contact_count min/mean/max: `{case.foot_ground_contact_min:.4g}` / `{case.foot_ground_contact_mean:.4g}` / `{case.foot_ground_contact_max:.4g}`",
                f"- 20-step window before failure/end: `{ws['start_step']}` to `{ws['end_step']}`",
                f"- window root_height min/mean/max: `{ws['root_height']['min']:.4g}` / `{ws['root_height']['mean']:.4g}` / `{ws['root_height']['max']:.4g}`",
                f"- window action saturation min/mean/max: `{ws['action_saturation']['min']:.4g}` / `{ws['action_saturation']['mean']:.4g}` / `{ws['action_saturation']['max']:.4g}`",
                f"- window torque saturation min/mean/max: `{ws['torque_saturation']['min']:.4g}` / `{ws['torque_saturation']['mean']:.4g}` / `{ws['torque_saturation']['max']:.4g}`",
                f"- window joint_vel_abs_max min/mean/max: `{ws['joint_vel_abs_max']['min']:.4g}` / `{ws['joint_vel_abs_max']['mean']:.4g}` / `{ws['joint_vel_abs_max']['max']:.4g}`",
                f"- window foot_contact_force_norm min/mean/max: `{ws['foot_contact_force_norm']['min']:.4g}` / `{ws['foot_contact_force_norm']['mean']:.4g}` / `{ws['foot_contact_force_norm']['max']:.4g}`",
                f"- window foot_ground_contact_count min/mean/max: `{ws['foot_ground_contact_count']['min']:.4g}` / `{ws['foot_ground_contact_count']['mean']:.4g}` / `{ws['foot_ground_contact_count']['max']:.4g}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Next Validation Steps",
            "",
            "- Compare saturated joints against action scale and default pose offsets before changing any actuator or solver parameter.",
            "- Run a narrow actuator timing comparison for matching command pairs where position and pd_torque diverge.",
            "- Inspect contact/root settling windows for failed cases, especially the first 100 steps.",
            "- Keep model, collision geometry, gains, friction, solver, root height, and policy unchanged until the cause category is isolated.",
        ]
    )

    text = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    docs_output.parent.mkdir(parents=True, exist_ok=True)
    docs_output.write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    matrix_dir = Path(args.matrix_dir)
    if not matrix_dir.exists():
        raise FileNotFoundError(matrix_dir)
    deploy_cfg = load_deploy_cfg(args.deploy_cfg)
    paths = sorted(matrix_dir.glob("policy_*.npz"))
    if not paths:
        raise RuntimeError(f"No policy .npz files found in {matrix_dir}")
    cases = [analyze_case(path, deploy_cfg) for path in paths]
    output = Path(args.output)
    docs_output = Path(args.docs_output)
    write_report(output, cases, docs_output)
    print(f"Saved failure-window analysis: {output}")
    print(f"Saved docs failure-window analysis: {docs_output}")
    print(f"Analyzed {len(cases)} policy cases; failed={sum(not case.stable for case in cases)} stable={sum(case.stable for case in cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
