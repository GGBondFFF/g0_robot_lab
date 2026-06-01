"""Passive: tau = 0. Robot hangs limp. Safe initial / fallback state."""

import numpy as np
from .fsm_state import FSMState


class StatePassive(FSMState):
    name = "passive"

    def enter(self):
        self.env.act_mgr.reset()
        print("[FSM] -> Passive (tau=0)")

    def run(self):
        zero_tau = np.zeros(self.env.backend.model.nu, dtype=np.float64)
        self.env.step_torque(zero_tau)

    def check_transition(self, requested):
        # Passive can transition to FixStand only (Unitree convention).
        if requested == "fix_stand":
            return "fix_stand"
        return None
