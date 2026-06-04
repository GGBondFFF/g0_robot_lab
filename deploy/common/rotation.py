"""Rotation utilities (wxyz quaternion convention, matches IsaacLab)."""

import math
import numpy as np


def quat_apply_inverse_wxyz(q_wxyz, v):
    """Rotate vector v from world into body frame using inverse of q."""
    w, x, y, z = q_wxyz
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    return R.T @ np.asarray(v, dtype=np.float64)


def quat_wxyz_from_gravity(acc_body, sign=-1.0):
    """Build a yaw-free wxyz quaternion from a body-frame accelerometer reading.

    Used by the real-robot backend when only raw IMU acc/gyro are available
    (no fused quaternion). A stationary accelerometer measures specific force
    = -gravity in the body frame, so the (signed, normalised) acc vector IS the
    body-frame gravity direction the policy needs:

        proj_grav_b  ==  quat_apply_inverse_wxyz(q, [0,0,-1])  ==  g_meas

    We return the shortest-arc rotation taking g_meas -> world-down [0,0,-1],
    which carries no yaw (rotation axis is horizontal), exactly what
    projected_gravity wants (it is yaw-invariant anyway).

    ``sign``: +1 or -1 depending on the IMU's accel convention. With the robot
    standing upright, proj_grav_b must be ~[0,0,-1]; flip ``sign`` if a static
    test shows acc pointing the other way. THIS MUST BE VERIFIED ON HARDWARE.

    NOTE: only valid quasi-statically. Under dynamic motion acc includes linear
    acceleration; switch to a fused quaternion (Imu::ImuOutput.q) for walking.
    """
    a = np.asarray(acc_body, dtype=np.float64)
    n = np.linalg.norm(a)
    if n < 1e-6:
        return np.array([1.0, 0.0, 0.0, 0.0])  # no info -> upright
    g_meas = sign * a / n
    down = np.array([0.0, 0.0, -1.0])
    d = float(np.clip(np.dot(g_meas, down), -1.0, 1.0))
    if d > 0.999999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if d < -0.999999:
        # antiparallel: 180deg about any horizontal axis
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross(g_meas, down)
    s = math.sqrt((1.0 + d) * 2.0)
    q = np.array([s * 0.5, axis[0] / s, axis[1] / s, axis[2] / s])
    return q / np.linalg.norm(q)


def quat_to_rpy_wxyz(q):
    w, x, y, z = q
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw
