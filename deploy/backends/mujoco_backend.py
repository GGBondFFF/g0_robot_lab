"""MuJoCo backend.

Single point of contact with mujoco-python. Upper layers (FSM/Env/Action)
talk to this class via:

    state = backend.read_state()        # base + joint q/dq in MJ order
    backend.apply_torque(tau_mj)        # data.ctrl[:] = tau, then mj_step xN
    backend.step_n(n)                   # advance n substeps without changing ctrl

The real-robot backend will expose the same surface, so FSM doesn't change.
"""

import numpy as np
import mujoco


class MujocoBackend:
    def __init__(self, mjcf_path: str, sim_dt: float = None, keyframe: str = None,
                 joint_names_mj: list = None,
                 joint_damping: float = None, joint_frictionloss: float = None):
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        if sim_dt is not None:
            self.model.opt.timestep = float(sim_dt)

        if self.model.nu != 22:
            raise RuntimeError(f"Expected nu=22, got {self.model.nu}")

        # actuator -> joint qpos/qvel addresses, and per-actuator joint name
        qadr = np.empty(self.model.nu, dtype=np.int64)
        dadr = np.empty(self.model.nu, dtype=np.int64)
        joint_per_actuator = []
        for i in range(self.model.nu):
            jid = int(self.model.actuator_trnid[i, 0])
            qadr[i] = int(self.model.jnt_qposadr[jid])
            dadr[i] = int(self.model.jnt_dofadr[jid])
            joint_per_actuator.append(
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            )
        if joint_names_mj is not None:
            for i, jn in enumerate(joint_per_actuator):
                if jn != joint_names_mj[i]:
                    raise RuntimeError(
                        f"Actuator order drift at i={i}: {jn} vs {joint_names_mj[i]}"
                    )
        self.qadr = qadr
        self.dadr = dadr
        self.joint_per_actuator = joint_per_actuator

        if joint_damping is not None:
            for i in range(self.model.nu):
                self.model.dof_damping[int(dadr[i])] = float(joint_damping)
        if joint_frictionloss is not None:
            for i in range(self.model.nu):
                self.model.dof_frictionloss[int(dadr[i])] = float(joint_frictionloss)

        self.ctrl_lo = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = self.model.actuator_ctrlrange[:, 1].copy()

        self.base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )
        if self.base_body_id < 0:
            raise RuntimeError("Missing body 'base_link' in MJCF")

        if keyframe is not None:
            kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, keyframe)
            if kid < 0:
                raise RuntimeError(f"Missing keyframe {keyframe!r}")
            mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
            mujoco.mj_forward(self.model, self.data)

    # ---- state read

    def read_state(self):
        """Return dict with base + actuated-joint state in MJ order."""
        d = self.data
        m = self.model
        q_mj = np.empty(m.nu, dtype=np.float64)
        dq_mj = np.empty(m.nu, dtype=np.float64)
        for i in range(m.nu):
            q_mj[i] = d.qpos[self.qadr[i]]
            dq_mj[i] = d.qvel[self.dadr[i]]
        vel6 = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.base_body_id, vel6, 1)
        return {
            "q_mj": q_mj,
            "dq_mj": dq_mj,
            "base_quat_wxyz": d.qpos[3:7].copy(),
            "base_pos_w": d.qpos[0:3].copy(),
            "base_lin_vel_w": d.qvel[0:3].copy(),
            "base_ang_vel_b": vel6[0:3].copy(),
            "sim_time": float(d.time),
        }

    # ---- control + stepping

    def apply_torque(self, tau_mj: np.ndarray, n_substeps: int = 1):
        tau = np.clip(tau_mj, self.ctrl_lo, self.ctrl_hi)
        self.data.ctrl[:] = tau
        for _ in range(int(n_substeps)):
            mujoco.mj_step(self.model, self.data)

    def apply_external_force(self, force_w: np.ndarray):
        """Apply a world-frame linear force to base_link (for elastic band)."""
        self.data.xfrc_applied[self.base_body_id, 0:3] = force_w
        self.data.xfrc_applied[self.base_body_id, 3:6] = 0.0

    def clear_external_force(self):
        self.data.xfrc_applied[self.base_body_id, :] = 0.0
