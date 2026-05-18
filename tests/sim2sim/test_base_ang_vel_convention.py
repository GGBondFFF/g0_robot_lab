from __future__ import annotations

import numpy as np

from scripts.sim2sim.g0_mujoco_interface import G0MuJoCoInterface


def test_world_angular_velocity_is_rotated_into_body_frame() -> None:
    angle = np.deg2rad(90.0)
    quat_wxyz = np.asarray([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])
    rot_world_body = G0MuJoCoInterface.quat_wxyz_to_matrix(quat_wxyz)
    world_ang_vel = np.asarray([1.0, 0.0, 0.0])
    body_ang_vel = rot_world_body.T @ world_ang_vel
    np.testing.assert_allclose(body_ang_vel, np.asarray([0.0, -1.0, 0.0]), atol=1e-12)
