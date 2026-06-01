"""FixStand: cosine-ramp from current q to default_joint_pos over ramp_time_s,
then hold. PD torque only — no policy."""

import numpy as np
from .fsm_state import FSMState


class StateFixStand(FSMState):
    name = "fix_stand"

    def __init__(self, env, ort_runner=None, cfg=None, rc=None,
                 default_q_mj=None, ramp_time_s: float = 1.5, step_dt: float = 0.02,
                 band=None):
        super().__init__(env, ort_runner, cfg, rc)
        self.default_q_mj = default_q_mj
        self.ramp_time_s = float(ramp_time_s)
        self.step_dt = float(step_dt)
        self.band = band
        self._q_start = None
        self._t = 0.0
        self._calibrated = False

    def enter(self):
        s = self.env.backend.read_state()
        self._q_start = s["q_mj"].copy()
        self._t = 0.0
        self._calibrated = False
        self.env.act_mgr.reset()
        # Slacken the band immediately so the ramp PD does the lifting, not
        # the band. We'll recalibrate again at end-of-ramp for the true
        # standing distance.
        if self.band is not None:
            self.band.calibrate(self.env.backend)
        print(f"[FSM] -> FixStand (ramp {self.ramp_time_s:.2f}s)")

    def run(self):
        alpha = 0.5 * (1.0 - np.cos(np.pi * min(self._t / self.ramp_time_s, 1.0)))
        target_q_mj = (1.0 - alpha) * self._q_start + alpha * self.default_q_mj
        self.env.step_to_target_q(target_q_mj)
        self._t += self.step_dt
        # Once ramp is done and robot has settled for a moment, slacken the band
        # so it doesn't unload the feet during RLBase.
        if (not self._calibrated
                and self._t >= self.ramp_time_s + 0.3
                and self.band is not None):
            self.band.calibrate(self.env.backend)
            self._calibrated = True

    def check_transition(self, requested):
        if requested == "passive":
            return "passive"
        # Allow RLBase only after ramp completes (Unitree gates this similarly).
        if requested == "rl_base" and self._t >= self.ramp_time_s:
            return "rl_base"
        return None
