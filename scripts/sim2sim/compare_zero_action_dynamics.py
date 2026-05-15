#!/usr/bin/env python3
"""Compare Isaac and MuJoCo zero-action dynamics diagnostics.

This report is intentionally tolerant of missing keys: Isaac and MuJoCo do not
yet export every diagnostic term. Missing fields are listed instead of causing
the comparison to fail.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ARRAY_KEYS = [
    "joint_pos",
    "joint_vel",
    "root_quat",
    "base_ang_vel",
    "projected_gravity",
    "command",
    "target_joint_pos",
]

OPTIONAL_KEYS = [
    "qacc",
    "joint_acc",
    "contact_count",
    "max_contact_force_norm",
    "foot_ground_contact_count",
    "contact_force",
    "foot_contact_force",
    "foot_contact_force_norm",
    "left_foot_contact_force",
    "right_foot_contact_force",
    "left_foot_contact_force_norm",
    "right_foot_contact_force_norm",
    "total_foot_contact_force_norm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac", default="logs/sim2sim/isaac_zero_action_golden_io.npz", help="Isaac golden .npz path.")
    parser.add_argument("--mujoco", default="logs/sim2sim/mujoco_zero_action_rollout.npz", help="MuJoCo rollout .npz path.")
    parser.add_argument("--output", default="logs/sim2sim/zero_action_dynamics_compare_report.md", help="Markdown report path.")
    return parser.parse_args()


def _root_height(data: np.lib.npyio.NpzFile) -> np.ndarray | None:
    if "root_height" in data.files:
        return np.asarray(data["root_height"], dtype=np.float64)
    if "root_pos" not in data.files:
        return None
    root_pos = np.asarray(data["root_pos"], dtype=np.float64)
    if root_pos.ndim < 2 or root_pos.shape[-1] < 3:
        return None
    return root_pos[..., 2]


def _stats(left: np.ndarray, right: np.ndarray) -> tuple[str, float | None, float | None, tuple[int, ...], tuple[int, ...]]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    original_shapes = (left.shape, right.shape)
    if left.shape != right.shape:
        if left.ndim == 0 or right.ndim == 0:
            return "shape mismatch", None, None, *original_shapes
        min_len = min(left.shape[0], right.shape[0])
        if min_len <= 0:
            return "shape mismatch", None, None, *original_shapes
        left = left[:min_len]
        right = right[:min_len]
        status = f"shape mismatch, compared first {min_len}"
    else:
        status = "ok"
    err = np.abs(left - right)
    return status, float(np.nanmean(err)), float(np.nanmax(err)), *original_shapes


def _line(key: str, isaac_value: np.ndarray | None, mujoco_value: np.ndarray | None) -> str:
    if isaac_value is None or mujoco_value is None:
        missing = []
        if isaac_value is None:
            missing.append("Isaac")
        if mujoco_value is None:
            missing.append("MuJoCo")
        return f"| `{key}` | missing in {', '.join(missing)} | n/a | n/a | n/a | n/a |"
    status, mean_abs, max_abs, isaac_shape, mujoco_shape = _stats(isaac_value, mujoco_value)
    mean_text = "n/a" if mean_abs is None else f"{mean_abs:.6g}"
    max_text = "n/a" if max_abs is None else f"{max_abs:.6g}"
    return f"| `{key}` | {status} | `{isaac_shape}` | `{mujoco_shape}` | {mean_text} | {max_text} |"


def _summary_series(data: np.lib.npyio.NpzFile, key: str) -> list[str]:
    if key not in data.files:
        return [f"- `{key}`: missing"]
    value = np.asarray(data[key], dtype=np.float64)
    return [
        f"- `{key}` shape: `{value.shape}`",
        f"- `{key}` min/mean/max: `{np.nanmin(value):.6g}` / `{np.nanmean(value):.6g}` / `{np.nanmax(value):.6g}`",
    ]


def main() -> int:
    args = parse_args()
    isaac_path = Path(args.isaac)
    mujoco_path = Path(args.mujoco)
    if not isaac_path.exists():
        raise FileNotFoundError(f"Isaac golden file does not exist: {isaac_path}")
    if not mujoco_path.exists():
        raise FileNotFoundError(f"MuJoCo rollout file does not exist: {mujoco_path}")

    isaac = np.load(isaac_path, allow_pickle=True)
    mujoco = np.load(mujoco_path, allow_pickle=True)

    lines = [
        "# Zero-Action Dynamics Compare Report",
        "",
        f"- Isaac file: `{isaac_path}`",
        f"- MuJoCo file: `{mujoco_path}`",
        "",
        "| key | status | Isaac shape | MuJoCo shape | mean abs error | max abs error |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for key in ARRAY_KEYS:
        lines.append(_line(key, isaac[key] if key in isaac.files else None, mujoco[key] if key in mujoco.files else None))
    lines.append(_line("root_height", _root_height(isaac), _root_height(mujoco)))

    lines.extend(["", "## Optional Dynamics Diagnostics", ""])
    for key in OPTIONAL_KEYS:
        lines.append(_line(key, isaac[key] if key in isaac.files else None, mujoco[key] if key in mujoco.files else None))

    lines.extend(["", "## MuJoCo Diagnostic Ranges", ""])
    for key in [
        "qacc",
        "joint_acc",
        "contact_count",
        "max_contact_force_norm",
        "foot_ground_contact_count",
        "foot_contact_force_norm",
        "left_foot_contact_force_norm",
        "right_foot_contact_force_norm",
        "total_foot_contact_force_norm",
    ]:
        lines.extend(_summary_series(mujoco, key))

    lines.extend(
        [
            "",
            "## Missing Keys",
            "",
            f"- Isaac-only keys: `{sorted(set(isaac.files) - set(mujoco.files))}`",
            f"- MuJoCo-only keys: `{sorted(set(mujoco.files) - set(isaac.files))}`",
            "",
            "## Notes",
            "",
            "- `action` and `target_joint_pos` alignment should be checked in the rollout compare report; this file focuses on dynamics terms.",
            "- Contact and acceleration diagnostics are currently MuJoCo-side unless Isaac export is extended with equivalent fields.",
            "- Differences here are not interpreted as policy failure. They identify remaining simulator/model alignment work.",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved zero-action dynamics compare report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
