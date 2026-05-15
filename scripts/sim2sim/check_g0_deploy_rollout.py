#!/usr/bin/env python3
"""Check a G0 deploy-style MuJoCo rollout against deploy.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", required=True, help="Deploy rollout .npz path.")
    parser.add_argument("--deploy-cfg", required=True, help="deploy.yaml path.")
    parser.add_argument("--output", required=True, help="Markdown report output path.")
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def action_cfg(deploy_cfg: dict[str, Any]) -> dict[str, Any]:
    actions = deploy_cfg.get("actions") or {}
    if "JointPositionAction" in actions:
        return actions["JointPositionAction"]
    if len(actions) == 1:
        return next(iter(actions.values()))
    raise KeyError(f"No JointPositionAction in deploy.yaml actions: {list(actions)}")


def vector(value: Any, expected: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 1 and expected > 1:
        array = np.repeat(array, expected)
    if array.shape != (expected,):
        raise ValueError(f"{name} shape mismatch: expected ({expected},), got {array.shape}")
    return array


def finite_check(name: str, array: np.ndarray, failures: list[str], lines: list[str]) -> None:
    finite = bool(np.all(np.isfinite(array)))
    lines.append(f"- `{name}` finite: `{finite}`")
    if not finite:
        failures.append(f"{name} contains NaN or Inf")


def shape_check(name: str, array: np.ndarray, expected_tail: tuple[int, ...], failures: list[str], lines: list[str]) -> None:
    ok = array.ndim >= len(expected_tail) and tuple(array.shape[-len(expected_tail) :]) == expected_tail
    lines.append(f"- `{name}` shape: `{array.shape}`, expected tail `{expected_tail}` -> `{ok}`")
    if not ok:
        failures.append(f"{name} shape mismatch")


def scalar_max(array: np.ndarray) -> float:
    if array.size == 0:
        return float("nan")
    return float(np.nanmax(np.abs(array)))


def main() -> int:
    args = parse_args()
    deploy_cfg = load_yaml(args.deploy_cfg)
    rollout_path = Path(args.rollout)
    if not rollout_path.exists():
        raise FileNotFoundError(rollout_path)
    data = np.load(rollout_path, allow_pickle=True)

    joint_names = list(deploy_cfg["joint_names"])
    action_dim = len(joint_names)
    obs_dim = int(deploy_cfg["observations"]["policy_obs_dim"])
    act_cfg = action_cfg(deploy_cfg)
    scale = vector(act_cfg["scale"], action_dim, "action scale")
    offset = vector(act_cfg["offset"], action_dim, "action offset")
    effort_limit = vector(deploy_cfg["effort_limit_sim"], action_dim, "effort_limit_sim")
    velocity_limit = vector(deploy_cfg["velocity_limit_sim"], action_dim, "velocity_limit_sim")

    failures: list[str] = []
    lines: list[str] = [
        "# G0 Deploy Rollout Check",
        "",
        f"- rollout: `{rollout_path}`",
        f"- deploy_cfg: `{args.deploy_cfg}`",
        f"- steps: `{data['obs'].shape[0] if 'obs' in data else 0}`",
        f"- action_dim: `{action_dim}`",
        f"- policy_obs_dim: `{obs_dim}`",
        "",
        "## Shape Checks",
    ]

    required = [
        "obs",
        "policy_action",
        "processed_action",
        "target_joint_pos",
        "pd_q_des",
        "pd_dq_des",
        "pd_kp",
        "pd_kd",
        "pd_tau_ff",
        "pd_tau_cmd",
        "pd_tau_cmd_clipped",
        "joint_pos",
        "joint_vel",
        "root_height",
        "command",
        "contact_count",
        "foot_ground_contact_count",
        "foot_contact_force_norm",
    ]
    for key in required:
        if key not in data:
            failures.append(f"missing rollout key: {key}")

    if failures:
        lines.append("")
        lines.append("## Failures")
        lines.extend(f"- {failure}" for failure in failures)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Check FAILED: {args.output}")
        return 1

    shape_check("obs", data["obs"], (obs_dim,), failures, lines)
    for key in [
        "policy_action",
        "processed_action",
        "target_joint_pos",
        "pd_q_des",
        "pd_dq_des",
        "pd_kp",
        "pd_kd",
        "pd_tau_ff",
        "pd_tau_cmd",
        "pd_tau_cmd_clipped",
        "joint_pos",
        "joint_vel",
    ]:
        shape_check(key, data[key], (action_dim,), failures, lines)
    shape_check("command", data["command"], (3,), failures, lines)

    lines.append("")
    lines.append("## Consistency Checks")
    expected_target = offset[None, :] + scale[None, :] * np.clip(data["policy_action"], -1.0, 1.0)
    target_err = scalar_max(data["target_joint_pos"] - expected_target)
    lines.append(f"- target formula max abs error: `{target_err:.6g}`")
    if target_err > 1e-8:
        failures.append("target_joint_pos does not match offset + scale * clipped action")

    q_des_err = scalar_max(data["pd_q_des"] - data["target_joint_pos"])
    lines.append(f"- pd_q_des target max abs error: `{q_des_err:.6g}`")
    if q_des_err > 1e-10:
        failures.append("pd_q_des does not match target_joint_pos")

    clipped_expected = np.clip(data["pd_tau_cmd"], -effort_limit[None, :], effort_limit[None, :])
    clip_err = scalar_max(data["pd_tau_cmd_clipped"] - clipped_expected)
    lines.append(f"- effort clip max abs error: `{clip_err:.6g}`")
    if clip_err > 1e-8:
        failures.append("pd_tau_cmd_clipped does not match effort_limit_sim clipping")

    tau_after_clip_over = scalar_max(np.maximum(np.abs(data["pd_tau_cmd_clipped"]) - effort_limit[None, :], 0.0))
    tau_raw_over = scalar_max(np.maximum(np.abs(data["pd_tau_cmd"]) - effort_limit[None, :], 0.0))
    velocity_over = scalar_max(np.maximum(np.abs(data["joint_vel"]) - velocity_limit[None, :], 0.0))
    lines.append(f"- raw pd_tau_cmd exceed effort max: `{tau_raw_over:.6g}`")
    lines.append(f"- clipped pd_tau_cmd exceed effort max: `{tau_after_clip_over:.6g}`")
    lines.append(f"- joint velocity exceed velocity_limit_sim max: `{velocity_over:.6g}`")
    if tau_after_clip_over > 1e-8:
        failures.append("pd_tau_cmd_clipped exceeds effort_limit_sim")

    rollout_joint_names = [str(name) for name in np.asarray(data["joint_names"]).tolist()]
    joint_order_ok = rollout_joint_names == joint_names
    lines.append(f"- joint order matches deploy.yaml: `{joint_order_ok}`")
    if not joint_order_ok:
        failures.append("joint order does not match deploy.yaml")

    command = data["command"]
    command_constant = scalar_max(command - command[0:1])
    lines.append(f"- command constant max abs diff: `{command_constant:.6g}`")

    for key in required:
        finite_check(key, np.asarray(data[key]), failures, lines)

    root_height = np.asarray(data["root_height"], dtype=np.float64)
    contact_count = np.asarray(data["contact_count"], dtype=np.float64)
    foot_force = np.asarray(data["foot_contact_force_norm"], dtype=np.float64)
    lines.append("")
    lines.append("## Ranges")
    lines.append(f"- root_height min/max: `{np.nanmin(root_height):.6g}` / `{np.nanmax(root_height):.6g}`")
    lines.append(f"- contact_count min/max: `{np.nanmin(contact_count):.6g}` / `{np.nanmax(contact_count):.6g}`")
    lines.append(f"- foot_contact_force_norm min/max: `{np.nanmin(foot_force):.6g}` / `{np.nanmax(foot_force):.6g}`")

    is_zero_action = scalar_max(data["policy_action"]) < 1e-8
    lines.append("")
    lines.append("## Zero-Action Metrics")
    lines.append(f"- detected zero-action: `{is_zero_action}`")
    if is_zero_action:
        max_target_default_abs_err = scalar_max(data["target_joint_pos"] - offset[None, :])
        lines.append(f"- max_target_default_abs_err: `{max_target_default_abs_err:.6g}`")
        lines.append(f"- max_abs_policy_action: `{scalar_max(data['policy_action']):.6g}`")
        lines.append(f"- max_abs_tau_cmd: `{scalar_max(data['pd_tau_cmd']):.6g}`")
        lines.append(f"- max_abs_joint_vel: `{scalar_max(data['joint_vel']):.6g}`")
        lines.append(f"- max_abs_root_ang_vel: `{scalar_max(data['base_ang_vel']):.6g}`")
        if max_target_default_abs_err > 1e-8:
            failures.append("zero-action target_joint_pos is not default_joint_pos")

    lines.append("")
    lines.append("## Result")
    if failures:
        lines.append("FAILED")
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("OK")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Check {'FAILED' if failures else 'OK'}: {output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
