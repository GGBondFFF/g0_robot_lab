"""Shared controlled root-frame diagnostic samples and math helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RootFrameSample:
    name: str
    rpy: tuple[float, float, float]
    ang_vel: tuple[float, float, float]


ORIENTATION_CASES = [
    ("upright", (0.0, 0.0, 0.0)),
    ("roll10", (math.radians(10.0), 0.0, 0.0)),
    ("pitch10", (0.0, math.radians(10.0), 0.0)),
    ("yaw30", (0.0, 0.0, math.radians(30.0))),
    ("roll10_pitch10_yaw30", (math.radians(10.0), math.radians(10.0), math.radians(30.0))),
]

ANGULAR_VELOCITY_CASES = [
    ("w000", (0.0, 0.0, 0.0)),
    ("w100", (1.0, 0.0, 0.0)),
    ("w010", (0.0, 1.0, 0.0)),
    ("w001", (0.0, 0.0, 1.0)),
]


def controlled_samples() -> list[RootFrameSample]:
    samples: list[RootFrameSample] = []
    for orientation_name, rpy in ORIENTATION_CASES:
        for velocity_name, ang_vel in ANGULAR_VELOCITY_CASES:
            samples.append(RootFrameSample(f"{orientation_name}_{velocity_name}", rpy, ang_vel))
    return samples


def quat_wxyz_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return quaternion in w, x, y, z order for XYZ roll-pitch-yaw angles."""

    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def quat_xyzw_from_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return np.asarray([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)


def matrix_from_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        return np.eye(3)
    w, x, y, z = quat / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def projected_gravity_body_from_wxyz(quat: np.ndarray) -> np.ndarray:
    """Project normalized world gravity into body frame as R_world_body.T @ g_w."""

    return matrix_from_quat_wxyz(quat).T @ np.asarray([0.0, 0.0, -1.0], dtype=np.float64)


def body_ang_vel_from_world(quat: np.ndarray, world_ang_vel: np.ndarray) -> np.ndarray:
    return matrix_from_quat_wxyz(quat).T @ np.asarray(world_ang_vel, dtype=np.float64)


def world_ang_vel_from_body(quat: np.ndarray, body_ang_vel: np.ndarray) -> np.ndarray:
    return matrix_from_quat_wxyz(quat) @ np.asarray(body_ang_vel, dtype=np.float64)
