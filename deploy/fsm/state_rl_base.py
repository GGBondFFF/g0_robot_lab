"""RLBase: run ONNX policy in closed loop."""

import numpy as np
from .fsm_state import FSMState


class StateRLBase(FSMState):
    name = "rl_base"

    def enter(self):
        # rebuild history from current state so the policy doesn't see a stale buffer
        self.env.obs_mgr.reset()
        # prime history with current single-step obs replicated L times
        s = self.env.backend.read_state()
        obs77 = self.env.obs_mgr.build_single_step(
            q_mj=s["q_mj"], dq_mj=s["dq_mj"],
            base_quat_wxyz=s["base_quat_wxyz"],
            base_ang_vel_b=s["base_ang_vel_b"],
            velocity_cmd=self.env.cmd_mgr.command,
            last_action_sdk=self.env.act_mgr.last_action_sdk,
            sim_time=s["sim_time"],
        )
        for _ in range(self.env.obs_mgr.history_length):
            self.env.obs_mgr.push_and_flatten(obs77)
        print("[FSM] -> RLBase (policy.onnx closed loop)")

    def run(self):
        obs_flat = self.env.obs_mgr.flatten()
        action_sdk = self.ort(obs_flat)
        self.env.step_policy(action_sdk)

    def check_transition(self, requested):
        # RLBase can only fall back to Passive (emergency stop).
        if requested == "passive":
            return "passive"
        return None
