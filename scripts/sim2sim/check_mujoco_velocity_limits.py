#!/usr/bin/env python3
"""Check MuJoCo rollout joint velocities against Isaac G0 velocity limits.

This is a diagnostic report only. It does not enforce limits or modify the
MuJoCo model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from scripts.sim2sim import g0_sim2sim_config as cfg
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.sim2sim import g0_sim2sim_config as cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", default="logs/sim2sim/mujoco_zero_action_rollout.npz", help="MuJoCo rollout .npz path.")
    parser.add_argument("--output", default="logs/sim2sim/mujoco_velocity_limit_report.md", help="Markdown report path.")
    return parser.parse_args()


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def main() -> int:
    args = parse_args()
    rollout_path = Path(args.rollout)
    if not rollout_path.exists():
        raise FileNotFoundError(f"MuJoCo rollout file does not exist: {rollout_path}")
    data = np.load(rollout_path, allow_pickle=True)
    if "joint_vel" not in data.files:
        raise KeyError(f"{rollout_path} does not contain `joint_vel`")

    joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)
    if joint_vel.ndim != 2 or joint_vel.shape[1] != cfg.get_action_dim():
        raise ValueError(f"Expected joint_vel shape (steps, {cfg.get_action_dim()}), got {joint_vel.shape}")

    specs = cfg.get_isaac_actuator_specs()
    lines = [
        "# MuJoCo Velocity Limit Diagnostic Report",
        "",
        f"- Rollout: `{rollout_path}`",
        f"- Steps: `{joint_vel.shape[0]}`",
        "",
        "| joint_name | velocity_limit_sim | max_abs_joint_vel | ratio | exceeded | first_exceed_step |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]

    exceeded_count = 0
    max_ratio = 0.0
    worst_joint = None
    for column, joint_name in enumerate(cfg.get_joint_names()):
        limit = float(specs[joint_name].velocity_limit_sim)
        abs_vel = np.abs(joint_vel[:, column])
        max_abs = float(np.nanmax(abs_vel))
        ratio = max_abs / limit if limit > 0.0 else float("nan")
        exceeded_steps = np.flatnonzero(abs_vel > limit)
        exceeded = exceeded_steps.size > 0
        first_exceed = int(exceeded_steps[0]) if exceeded else None
        exceeded_count += int(exceeded)
        if np.isfinite(ratio) and ratio > max_ratio:
            max_ratio = ratio
            worst_joint = joint_name
        lines.append(
            f"| `{joint_name}` | {_fmt(limit)} | {_fmt(max_abs)} | {_fmt(ratio)} | {exceeded} | {_fmt(first_exceed)} |"
        )

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- exceeded joints: `{exceeded_count}/{cfg.get_action_dim()}`",
            f"- worst ratio: `{_fmt(max_ratio)}` on `{worst_joint}`",
            "- This report only checks observed rollout velocities against Isaac `velocity_limit_sim` values.",
            "- The current MuJoCo model does not yet implement a verified equivalent velocity-limit enforcement mechanism.",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved velocity limit report: {output}")
    print(f"exceeded joints: {exceeded_count}/{cfg.get_action_dim()}")
    print(f"worst ratio: {_fmt(max_ratio)} on {worst_joint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
