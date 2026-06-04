"""ManagerBasedRLEnv — Python deploy-side stand-in.

Mirrors unitree_rl_lab/deploy/include/isaaclab/envs/manager_based_rl_env.h
but stripped to what matters for a single, sim2sim closed loop:

    env.step(action_sdk) -> obs_flat (385,)

The env owns:
    - backend (MujocoBackend)
    - command_manager (vx, vy, wz)
    - action_manager (JointPositionAction → torque)
    - observation_manager (per-term scale + history)

It does NOT own the FSM. The FSM picks which action to send each step
(zero torque, ramp-to-default, or policy output) and the env just executes.
"""

import numpy as np


class ManagerBasedRLEnv:
    def __init__(self, backend, command_manager, action_manager, observation_manager,
                 decimation: int):
        self.backend = backend
        self.cmd_mgr = command_manager
        self.act_mgr = action_manager
        self.obs_mgr = observation_manager
        self.decimation = int(decimation)

    def reset(self):
        self.act_mgr.reset()
        self.obs_mgr.reset()
        return self._build_obs()

    # ---- the three control "primitives" the FSM picks among.
    # All of them advance time by exactly one policy step (decimation substeps).

    def step_policy(self, action_sdk: np.ndarray):
        """Standard RL step: action -> target q -> PD torque -> mj_step xN."""
        target_q_mj = self.act_mgr.compute_target_q_mj(action_sdk)
        state = self.backend.read_state()
        for _ in range(self.decimation):
            state = self.backend.read_state()
            tau = self.act_mgr.compute_torque(target_q_mj, state["q_mj"], state["dq_mj"])
            self.backend.apply_torque(tau, n_substeps=1)
        return self._build_obs()

    def step_torque(self, tau_mj: np.ndarray):
        """Apply a fixed torque vector for one policy step (Passive sends 0)."""
        for _ in range(self.decimation):
            self.backend.apply_torque(tau_mj, n_substeps=1)
        return self._build_obs()

    def step_to_target_q(self, target_q_mj: np.ndarray):
        """PD-track a fixed joint target for one policy step (FixStand)."""
        for _ in range(self.decimation):
            state = self.backend.read_state()
            tau = self.act_mgr.compute_torque(target_q_mj, state["q_mj"], state["dq_mj"])
            self.backend.apply_torque(tau, n_substeps=1)
        return self._build_obs()

    # ---- obs build (always available; FSM uses it for inspection too)

    def _build_obs(self):
        s = self.backend.read_state()
        obs77 = self.obs_mgr.build_single_step(
            q_mj=s["q_mj"], dq_mj=s["dq_mj"],
            base_quat_wxyz=s["base_quat_wxyz"],
            base_ang_vel_b=s["base_ang_vel_b"],
            velocity_cmd=self.cmd_mgr.command,
            last_action_sdk=self.act_mgr.last_action_sdk,
            sim_time=s["sim_time"],
        )
        return self.obs_mgr.push_and_flatten(obs77)
