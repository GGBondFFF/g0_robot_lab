"""Virtual elastic band on base_link.

Ports unitree_mujoco/simulate/src/main.cc:54-86 (ElasticBand class).

Two force models (``mode``):

``rope`` (Unitree default)
    Spring along the world-frame vector (anchor - base):
        f = k * (||anchor - base|| - L_rest) - c * v_along
    When the robot tips sideways, base_link shifts horizontally and the
    pull direction tilts — it is NOT body-local, but it is also NOT world
    vertical. This model suspends the robot during stand-up; it does NOT
    actively upright a fallen robot.

``world_up``
    Spring on vertical span only (anchor_z - base_z):
        f_w = [0, 0, k * stretch - c * v_z]
    Always pulls straight up in world +Z regardless of roll/pitch. Useful
    when you want height support during sim2sim even if the torso tilts.
    Still does not apply uprighting torque — use PD / policy for that.
"""

import numpy as np


class ElasticBand:
    """Virtual elastic band.

    one_sided=True (default): band only pulls when stretch > 0 (catch-on-fall).
    Use `calibrate(backend)` once the robot is standing to set L_rest.

    one_sided=False: two-sided spring (can push down as well as pull up).
    """

    MODES = ("rope", "world_up")

    def __init__(self, anchor=(0.0, 0.0, 2.0), stiffness=50.0, damping=10.0,
                 rest_length=0.0, enabled=True, one_sided=True,
                 mode: str = "rope"):
        self.anchor = np.array(anchor, dtype=np.float64)
        self.k = float(stiffness)
        self.c = float(damping)
        self.L = float(rest_length)
        self.enabled = bool(enabled)
        self.one_sided = bool(one_sided)
        if mode not in self.MODES:
            raise ValueError(f"elastic_band.mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode

    def _span(self, base_pos_w: np.ndarray) -> float:
        """Signed 'length' used for stretch = span - L_rest."""
        if self.mode == "world_up":
            return float(self.anchor[2] - base_pos_w[2])
        return float(np.linalg.norm(self.anchor - base_pos_w))

    def calibrate(self, backend):
        """Snap L_rest to the current span (slack at this pose)."""
        if not self.enabled:
            return
        s = backend.read_state()
        self.L = self._span(s["base_pos_w"])
        print(f"[ElasticBand] calibrated L_rest = {self.L:.3f} m "
              f"(mode={self.mode}, one_sided={self.one_sided})")

    def update(self, backend):
        if not self.enabled:
            backend.clear_external_force()
            return
        s = backend.read_state()
        base_pos = s["base_pos_w"]

        if self.mode == "world_up":
            stretch = self._span(base_pos) - self.L
            if self.one_sided and stretch <= 0.0:
                backend.clear_external_force()
                return
            vz = float(s["base_lin_vel_w"][2])
            f_scalar = self.k * stretch - self.c * vz
            if self.one_sided:
                f_scalar = max(f_scalar, 0.0)
            backend.apply_external_force(np.array([0.0, 0.0, f_scalar], dtype=np.float64))
            return

        # rope — matches unitree_mujoco
        delta = self.anchor - base_pos
        dist = float(np.linalg.norm(delta))
        if dist < 1e-9:
            backend.clear_external_force()
            return
        stretch = dist - self.L
        if self.one_sided and stretch <= 0.0:
            backend.clear_external_force()
            return
        dir_unit = delta / dist
        v_along = float(np.dot(s["base_lin_vel_w"], dir_unit))
        f_scalar = self.k * stretch - self.c * v_along
        if self.one_sided:
            f_scalar = max(f_scalar, 0.0)  # never push the robot down
        backend.apply_external_force(f_scalar * dir_unit)
