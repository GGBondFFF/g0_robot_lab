"""ObservationManager — per-term scale + grouped, oldest-first history.

Mirrors IsaacLab's manager with these terms (must match training cfg):
    base_ang_vel   (3)    scale 0.2
    projected_gravity (3) scale 1.0
    velocity_commands (3) scale 1.0
    joint_pos_rel (22)    scale 1.0
    joint_vel_rel (22)    scale 0.05
    last_action   (22)    scale 1.0
    gait_phase    (2)     scale 1.0   period_s 0.8

History layout (length L): per-term group, oldest first, concatenated.
    out = [term0 across L steps, term1 across L steps, ...]
"""

import math
import numpy as np

from ...common.rotation import quat_apply_inverse_wxyz


class ObservationManager:
    def __init__(self, cfg: dict, joint_names_sdk, joint_names_isaac, joint_names_mj,
                 default_q_isaac: np.ndarray):
        self.terms = cfg["terms"]
        self.history_length = int(cfg["history_length"])
        self.term_dims = [int(t["dim"]) for t in self.terms]
        self.term_scales = [float(t["scale"]) for t in self.terms]
        self.obs_dim_per_step = int(sum(self.term_dims))   # 77
        self.obs_dim_flat = self.obs_dim_per_step * self.history_length  # 385

        # gait term lookup
        self.gait_period_s = None
        for t in self.terms:
            if t["name"] == "gait_phase":
                self.gait_period_s = float(t.get("period_s", 0.8))
                break

        # joint index maps
        mj_index_by_name = {n: i for i, n in enumerate(joint_names_mj)}
        self.isaac_to_mj = np.array(
            [mj_index_by_name[n] for n in joint_names_isaac], dtype=np.int64
        )
        self.default_q_isaac = default_q_isaac.astype(np.float64)
        self.default_dq_isaac = np.zeros(22, dtype=np.float64)

        self._history = np.zeros(
            (self.history_length, self.obs_dim_per_step), dtype=np.float64
        )

    def reset(self):
        self._history[:] = 0.0

    # ---- single-step 77-D observation

    def build_single_step(self, q_mj, dq_mj, base_quat_wxyz, base_ang_vel_b,
                          velocity_cmd, last_action_sdk, sim_time):
        proj_grav_b = quat_apply_inverse_wxyz(base_quat_wxyz,
                                              np.array([0.0, 0.0, -1.0]))
        q_isaac = q_mj[self.isaac_to_mj]
        dq_isaac = dq_mj[self.isaac_to_mj]
        jpos_rel = q_isaac - self.default_q_isaac
        jvel_rel = dq_isaac - self.default_dq_isaac

        if self.gait_period_s is not None:
            phase = (sim_time / self.gait_period_s) % 1.0
            gait = np.array([
                math.sin(phase * 2.0 * math.pi),
                math.cos(phase * 2.0 * math.pi),
            ], dtype=np.float64)
        else:
            gait = np.zeros(2, dtype=np.float64)

        obs = np.zeros(self.obs_dim_per_step, dtype=np.float64)
        off = 0
        for term, scale in zip(self.terms, self.term_scales):
            name = term["name"]; dim = int(term["dim"])
            if name == "base_ang_vel":
                v = base_ang_vel_b
            elif name == "projected_gravity":
                v = proj_grav_b
            elif name == "velocity_commands":
                v = velocity_cmd
            elif name == "joint_pos_rel":
                v = jpos_rel
            elif name == "joint_vel_rel":
                v = jvel_rel
            elif name == "last_action":
                v = last_action_sdk
            elif name == "gait_phase":
                v = gait
            else:
                raise RuntimeError(f"Unknown obs term {name!r}")
            obs[off:off + dim] = np.asarray(v, dtype=np.float64) * scale
            off += dim
        return obs

    # ---- roll history forward and flatten

    def push_and_flatten(self, obs_77: np.ndarray) -> np.ndarray:
        # roll oldest-first: shift left, append newest at the end.
        self._history[:-1] = self._history[1:]
        self._history[-1] = obs_77
        return self.flatten()

    def flatten(self) -> np.ndarray:
        out = []
        off = 0
        for d in self.term_dims:
            out.append(self._history[:, off:off + d].reshape(-1))
            off += d
        return np.concatenate(out, axis=0)
