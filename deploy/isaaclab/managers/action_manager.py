"""ActionManager — JointPositionAction in SDK order.

Mirrors IsaacLab's JointPositionAction:
    target_q_sdk = default_q_sdk + scale * raw_action_sdk
Followed by SDK -> MJ remap and explicit PD on the MuJoCo side.
"""

import numpy as np


class JointPositionAction:
    def __init__(self, default_q_sdk: np.ndarray, scale: float,
                 sdk_to_mj: np.ndarray, kp_mj: np.ndarray, kd_mj: np.ndarray):
        self.default_q_sdk = default_q_sdk.astype(np.float64)
        self.scale = float(scale)
        self.sdk_to_mj = sdk_to_mj.astype(np.int64)
        self.kp_mj = kp_mj.astype(np.float64)
        self.kd_mj = kd_mj.astype(np.float64)
        self.last_action_sdk = np.zeros_like(self.default_q_sdk)

    def reset(self):
        self.last_action_sdk[:] = 0.0

    @staticmethod
    def _remap(vec_sdk, sdk_to_mj):
        vec_mj = np.empty_like(vec_sdk)
        vec_mj[sdk_to_mj] = vec_sdk
        return vec_mj

    def compute_target_q_mj(self, action_sdk: np.ndarray) -> np.ndarray:
        self.last_action_sdk[:] = action_sdk
        target_q_sdk = self.default_q_sdk + self.scale * action_sdk
        return self._remap(target_q_sdk, self.sdk_to_mj)

    def compute_torque(self, target_q_mj: np.ndarray,
                       q_mj: np.ndarray, dq_mj: np.ndarray) -> np.ndarray:
        return self.kp_mj * (target_q_mj - q_mj) - self.kd_mj * dq_mj
