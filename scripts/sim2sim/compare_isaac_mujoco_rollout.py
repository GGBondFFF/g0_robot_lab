#!/usr/bin/env python3
"""Compare Isaac Lab golden I/O with a MuJoCo rollout and write a Markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


COMPARE_KEYS = ["joint_pos", "joint_vel", "action", "target_joint_pos", "command"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac", default="logs/sim2sim/isaac_golden_io.npz", help="Isaac golden .npz path.")
    parser.add_argument("--mujoco", default="logs/sim2sim/mujoco_rollout.npz", help="MuJoCo rollout .npz path.")
    parser.add_argument("--output", default="logs/sim2sim/compare_report.md", help="Markdown report path.")
    return parser.parse_args()


def _root_height(data: np.lib.npyio.NpzFile) -> np.ndarray | None:
    if "root_pos" not in data.files:
        return None
    root_pos = np.asarray(data["root_pos"])
    if root_pos.ndim < 2 or root_pos.shape[-1] < 3:
        return None
    return root_pos[..., 2]


def compare_arrays(a: np.ndarray, b: np.ndarray) -> tuple[str, float | None, float | None]:
    if a.shape != b.shape:
        min_len = min(a.shape[0], b.shape[0]) if a.ndim > 0 and b.ndim > 0 else 0
        if min_len == 0:
            return f"shape mismatch: {a.shape} vs {b.shape}", None, None
        a = a[:min_len]
        b = b[:min_len]
        note = f"shape mismatch, compared first {min_len}: original {a.shape} vs {b.shape}"
    else:
        note = "ok"
    err = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    return note, float(np.nanmean(err)), float(np.nanmax(err))


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
        "# Isaac Lab / MuJoCo Rollout Compare Report",
        "",
        f"- Isaac file: `{isaac_path}`",
        f"- MuJoCo file: `{mujoco_path}`",
        "",
        "| key | status | mean abs error | max abs error |",
        "| --- | --- | ---: | ---: |",
    ]

    for key in COMPARE_KEYS:
        if key not in isaac.files or key not in mujoco.files:
            missing = []
            if key not in isaac.files:
                missing.append("Isaac")
            if key not in mujoco.files:
                missing.append("MuJoCo")
            lines.append(f"| `{key}` | missing in {', '.join(missing)} | n/a | n/a |")
            continue
        status, mean_abs, max_abs = compare_arrays(isaac[key], mujoco[key])
        mean_text = "n/a" if mean_abs is None else f"{mean_abs:.6g}"
        max_text = "n/a" if max_abs is None else f"{max_abs:.6g}"
        lines.append(f"| `{key}` | {status} | {mean_text} | {max_text} |")

    isaac_height = _root_height(isaac)
    mujoco_height = _root_height(mujoco)
    if isaac_height is None or mujoco_height is None:
        lines.append("| `root_height` | missing root_pos in one file | n/a | n/a |")
    else:
        status, mean_abs, max_abs = compare_arrays(isaac_height, mujoco_height)
        lines.append(f"| `root_height` | {status} | {mean_abs:.6g} | {max_abs:.6g} |")

    lines.extend(
        [
            "",
            "## Missing Keys",
            "",
            f"- Isaac-only keys: `{sorted(set(isaac.files) - set(mujoco.files))}`",
            f"- MuJoCo-only keys: `{sorted(set(mujoco.files) - set(isaac.files))}`",
            "",
        ]
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved compare report: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

