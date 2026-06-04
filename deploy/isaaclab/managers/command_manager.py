"""Velocity command holder. Mirrors UniformVelocityCommand at deploy time:
the policy just reads (vx, vy, wz); randomization happens in training only."""

import numpy as np


class VelocityCommandManager:
    def __init__(self, default=(0.0, 0.0, 0.0), ranges=None):
        self._cmd = np.array(default, dtype=np.float64)
        self.ranges = ranges or {
            "vx": (-1.0, 1.0), "vy": (-1.0, 1.0), "wz": (-2.0, 2.0)
        }

    def set(self, vx, vy, wz):
        lo, hi = self.ranges["vx"]; vx = float(np.clip(vx, lo, hi))
        lo, hi = self.ranges["vy"]; vy = float(np.clip(vy, lo, hi))
        lo, hi = self.ranges["wz"]; wz = float(np.clip(wz, lo, hi))
        self._cmd[:] = (vx, vy, wz)

    @property
    def command(self):
        return self._cmd.copy()
