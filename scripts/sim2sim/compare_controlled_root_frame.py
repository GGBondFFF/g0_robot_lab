#!/usr/bin/env python3
"""Compare Isaac and MuJoCo controlled root-frame diagnostics."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac", required=True, help="Isaac controlled root state .npz.")
    parser.add_argument("--mujoco", required=True, help="MuJoCo controlled root state .npz.")
    parser.add_argument("--output", required=True, help="Markdown report output.")
    parser.add_argument("--docs-output", default="docs/root_frame_convention_diagnostics.md", help="Documentation report output.")
    return parser.parse_args()


def names(data: np.lib.npyio.NpzFile) -> list[str]:
    return [str(name) for name in np.asarray(data["sample_name"]).tolist()]


def max_abs(array: np.ndarray) -> float:
    return float(np.nanmax(np.abs(array))) if array.size else float("nan")


def write_report(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    isaac = np.load(args.isaac, allow_pickle=True)
    mujoco = np.load(args.mujoco, allow_pickle=True)
    isaac_names = names(isaac)
    mujoco_names = names(mujoco)
    if isaac_names != mujoco_names:
        raise RuntimeError(f"sample_name mismatch: {isaac_names} != {mujoco_names}")

    isaac_quat = np.asarray(isaac["root_quat_w"], dtype=np.float64)
    mujoco_quat = np.asarray(mujoco["root_quat"], dtype=np.float64)
    isaac_pg = np.asarray(isaac["projected_gravity"], dtype=np.float64)
    mujoco_pg = np.asarray(mujoco["projected_gravity"], dtype=np.float64)
    isaac_w = np.asarray(isaac["base_ang_vel"], dtype=np.float64)
    mujoco_w = np.asarray(mujoco["base_ang_vel"], dtype=np.float64)
    isaac_h = np.asarray(isaac["root_height"], dtype=np.float64)
    mujoco_h = np.asarray(mujoco["root_height"], dtype=np.float64)

    quat_err = np.minimum(max_abs(isaac_quat - mujoco_quat), max_abs(isaac_quat + mujoco_quat))
    quat_xyzw_err = max_abs(isaac_quat[:, [1, 2, 3, 0]] - mujoco_quat)
    pg_err = max_abs(isaac_pg - mujoco_pg)
    w_err = max_abs(isaac_w - mujoco_w)
    height_err = max_abs(isaac_h - mujoco_h)

    quat_order = "wxyz" if quat_err < quat_xyzw_err else "possible xyzw mismatch"
    projected_gravity_ok = pg_err < 1e-4
    base_ang_vel_ok = w_err < 1e-4
    height_note = (
        "root height matches controlled value closely"
        if height_err < 1e-4
        else "root height differs; this is likely root definition/default-state difference, not a frame convention by itself"
    )

    lines = [
        "# Controlled Root Frame Convention Diagnostics",
        "",
        f"- isaac: `{args.isaac}`",
        f"- mujoco: `{args.mujoco}`",
        f"- samples: `{len(isaac_names)}`",
        "",
        "## Summary",
        "",
        f"- quaternion order assessment: `{quat_order}`",
        f"- quaternion max abs error allowing sign flip: `{quat_err:.6g}`",
        f"- xyzw reorder candidate max abs error: `{quat_xyzw_err:.6g}`",
        f"- projected_gravity max abs error: `{pg_err:.6g}`",
        f"- projected_gravity aligned: `{projected_gravity_ok}`",
        f"- base_ang_vel max abs error: `{w_err:.6g}`",
        f"- base_ang_vel aligned: `{base_ang_vel_ok}`",
        f"- root_height max abs error: `{height_err:.6g}`",
        f"- root_height note: {height_note}",
        "",
        "## Interpretation",
        "",
    ]
    if quat_order == "wxyz":
        lines.append("- Isaac and MuJoCo root quaternions use the same `w, x, y, z` order for these controlled samples.")
    else:
        lines.append("- Quaternion ordering may differ. Inspect root quaternion writes before trusting projected frame terms.")
    if projected_gravity_ok:
        lines.append("- `G0MuJoCoInterface.get_projected_gravity()` matches Isaac `projected_gravity_b` for controlled orientations.")
    else:
        lines.append("- `projected_gravity` differs. If the quaternion order is correct, inspect `get_projected_gravity()` and the world/body rotation direction.")
    if base_ang_vel_ok:
        lines.append("- `G0MuJoCoInterface.get_base_ang_vel()` matches Isaac `root_ang_vel_b` for the controlled angular velocity samples.")
    else:
        lines.append("- `base_ang_vel` differs. This indicates a likely world/body angular velocity convention mismatch.")
    lines.append("- No actuator, collision, root-height, friction, or solver parameters were changed by this diagnostic.")

    lines.extend(["", "## Per-Sample Errors", "", "| sample | quat err | projected gravity err | base ang vel err | root height err |", "| --- | ---: | ---: | ---: | ---: |"])
    for index, name in enumerate(isaac_names):
        q_err_i = min(max_abs(isaac_quat[index] - mujoco_quat[index]), max_abs(isaac_quat[index] + mujoco_quat[index]))
        lines.append(
            f"| `{name}` | {q_err_i:.6g} | {max_abs(isaac_pg[index] - mujoco_pg[index]):.6g} | "
            f"{max_abs(isaac_w[index] - mujoco_w[index]):.6g} | {abs(float(isaac_h[index] - mujoco_h[index])):.6g} |"
        )

    output = Path(args.output)
    docs_output = Path(args.docs_output)
    write_report(output, lines)
    write_report(docs_output, lines)
    print(f"Saved controlled root frame report: {output}")
    print(f"Saved docs report: {docs_output}")
    if not projected_gravity_ok or not base_ang_vel_ok:
        print("Frame diagnostics found mismatches.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
