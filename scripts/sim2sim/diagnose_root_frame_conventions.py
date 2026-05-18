#!/usr/bin/env python3
"""Print root-frame convention candidates for controlled RPY and angular velocity."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sim2sim.root_frame_samples import (  # noqa: E402
    body_ang_vel_from_world,
    matrix_from_quat_wxyz,
    projected_gravity_body_from_wxyz,
    quat_wxyz_from_rpy,
    quat_xyzw_from_wxyz,
    world_ang_vel_from_body,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roll", type=float, default=10.0, help="Roll in degrees.")
    parser.add_argument("--pitch", type=float, default=10.0, help="Pitch in degrees.")
    parser.add_argument("--yaw", type=float, default=30.0, help="Yaw in degrees.")
    parser.add_argument("--ang-vel", nargs=3, type=float, default=[1.0, 0.0, 0.0], help="Angular velocity candidate vector.")
    return parser.parse_args()


def fmt(array: np.ndarray) -> str:
    return np.array2string(np.asarray(array, dtype=np.float64), precision=6, suppress_small=False)


def main() -> int:
    args = parse_args()
    rpy = np.radians([args.roll, args.pitch, args.yaw])
    quat_wxyz = quat_wxyz_from_rpy(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    quat_xyzw = quat_xyzw_from_wxyz(quat_wxyz)
    rot = matrix_from_quat_wxyz(quat_wxyz)
    gravity_w = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    ang_vel = np.asarray(args.ang_vel, dtype=np.float64)

    print("Root frame convention candidates")
    print(f"rpy_deg: {[args.roll, args.pitch, args.yaw]}")
    print(f"rpy_rad: {fmt(rpy)}")
    print(f"quat_wxyz: {fmt(quat_wxyz)}")
    print(f"quat_xyzw: {fmt(quat_xyzw)}")
    print("")
    print("Projected gravity candidates")
    print(f"R_world_body.T @ [0,0,-1] body-frame gravity: {fmt(projected_gravity_body_from_wxyz(quat_wxyz))}")
    print(f"R_world_body @ [0,0,-1] world/body swapped candidate: {fmt(rot @ gravity_w)}")
    print(f"identity/no projection candidate: {fmt(gravity_w)}")
    print("")
    print("Base angular velocity candidates")
    print(f"input interpreted as world angular velocity: {fmt(ang_vel)}")
    print(f"world -> body via R.T @ w_world: {fmt(body_ang_vel_from_world(quat_wxyz, ang_vel))}")
    print(f"input interpreted as body angular velocity: {fmt(ang_vel)}")
    print(f"body -> world via R @ w_body: {fmt(world_ang_vel_from_body(quat_wxyz, ang_vel))}")
    print("")
    print("Assumption labels")
    print("- Isaac root_quat_w is expected to be wxyz.")
    print("- Isaac projected_gravity_b is expected to be body-frame gravity, R_world_body.T @ [0,0,-1].")
    print("- Isaac root_ang_vel_b is body-frame angular velocity.")
    print("- If MuJoCo qvel freejoint angular part behaves as body-frame angular velocity, get_base_ang_vel can return qvel[3:6].")
    print("- If MuJoCo qvel freejoint angular part behaves as world-frame angular velocity, get_base_ang_vel must rotate it into body frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
