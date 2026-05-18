from __future__ import annotations

import numpy as np

from scripts.sim2sim.g0_mujoco_interface import G0MuJoCoInterface


def test_projected_gravity_uses_wxyz_body_frame_projection() -> None:
    angle = np.deg2rad(90.0)
    quat_wxyz = np.asarray([np.cos(angle / 2.0), np.sin(angle / 2.0), 0.0, 0.0])
    rot_world_body = G0MuJoCoInterface.quat_wxyz_to_matrix(quat_wxyz)
    projected = rot_world_body.T @ np.asarray([0.0, 0.0, -1.0])
    np.testing.assert_allclose(projected, np.asarray([0.0, -1.0, 0.0]), atol=1e-12)
