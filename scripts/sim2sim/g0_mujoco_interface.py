"""MuJoCo-side interface bridge for G0 sim2sim validation.

This module delays importing ``mujoco`` until a model is actually loaded. That
keeps ordinary Python tests useful on machines where MuJoCo is not installed.
The observation builder is a first engineering scaffold: joint terms, commands,
last action, and gait phase are wired, while base angular velocity and projected
gravity include explicit TODO notes because their frame conventions must be
verified against Isaac Lab before relying on policy rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.sim2sim import g0_sim2sim_config as cfg
except ModuleNotFoundError:
    import g0_sim2sim_config as cfg


def import_mujoco() -> Any:
    """Import MuJoCo on demand and return the module."""

    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MuJoCo is not installed. Install the `mujoco` Python package to load mujoco/g0.xml."
        ) from exc
    if not hasattr(mujoco, "MjModel"):
        raise ModuleNotFoundError(
            "Imported a module named `mujoco`, but it does not expose `MjModel`. "
            "Install the DeepMind `mujoco` Python package to load mujoco/g0.xml."
        )
    return mujoco


@dataclass(frozen=True)
class JointIndex:
    """MuJoCo indices for one G0 joint."""

    qpos: int
    qvel: int
    actuator: int


class G0MuJoCoInterface:
    """Bridge between policy actions/observations and a MuJoCo G0 model."""

    def __init__(self, model_path: str | Path, command: np.ndarray | None = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"MuJoCo model does not exist: {self.model_path}")

        self.mujoco = import_mujoco()
        self.model = self.mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = self.mujoco.MjData(self.model)
        self.joint_names = cfg.get_joint_names()
        self.joint_indices: dict[str, JointIndex] = {}
        self.last_action = np.zeros(cfg.get_action_dim(), dtype=np.float64)
        self.command = np.zeros(3, dtype=np.float64) if command is None else np.asarray(command, dtype=np.float64)
        self.sim_time = 0.0
        self._history: list[np.ndarray] = []
        self.validate_model()
        self.reset()

    @staticmethod
    def expected_observation_dim() -> int:
        """Return flattened policy observation width including history."""

        return cfg.get_policy_observation_dim()

    @staticmethod
    def build_observation_frame(
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        last_action: np.ndarray,
        command: np.ndarray,
        sim_time: float,
        base_ang_vel: np.ndarray | None = None,
        projected_gravity: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build one non-history policy observation frame.

        TODO: ``base_ang_vel`` must be verified against Isaac Lab's body-frame
        convention. The current default is zeros.

        TODO: ``projected_gravity`` must be verified against Isaac Lab's
        quaternion/frame convention. The current default assumes upright base
        with gravity projected as ``[0, 0, -1]``.
        """

        joint_pos = np.asarray(joint_pos, dtype=np.float64)
        joint_vel = np.asarray(joint_vel, dtype=np.float64)
        last_action = np.asarray(last_action, dtype=np.float64)
        command = np.asarray(command, dtype=np.float64)
        base_ang_vel = np.zeros(3, dtype=np.float64) if base_ang_vel is None else np.asarray(base_ang_vel, dtype=np.float64)
        projected_gravity = (
            np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
            if projected_gravity is None
            else np.asarray(projected_gravity, dtype=np.float64)
        )

        if joint_pos.shape != (cfg.get_action_dim(),):
            raise ValueError(f"joint_pos shape must be ({cfg.get_action_dim()},), got {joint_pos.shape}")
        if joint_vel.shape != (cfg.get_action_dim(),):
            raise ValueError(f"joint_vel shape must be ({cfg.get_action_dim()},), got {joint_vel.shape}")
        if last_action.shape != (cfg.get_action_dim(),):
            raise ValueError(f"last_action shape must be ({cfg.get_action_dim()},), got {last_action.shape}")
        if command.shape != (3,):
            raise ValueError(f"command shape must be (3,), got {command.shape}")

        joint_pos_rel = joint_pos - cfg.get_default_joint_pos_array()
        phase = (sim_time % cfg.GAIT_PERIOD) / cfg.GAIT_PERIOD
        gait_phase = np.asarray([np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)], dtype=np.float64)
        return np.concatenate(
            [base_ang_vel * 0.2, projected_gravity, command, joint_pos_rel, joint_vel * 0.05, last_action, gait_phase]
        )

    def validate_model(self) -> None:
        """Validate that all G0 joints and actuators are present in the MuJoCo model."""

        self.joint_indices.clear()
        missing_joints: list[str] = []
        missing_actuators: list[str] = []
        for name in self.joint_names:
            joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                missing_joints.append(name)
                continue

            actuator_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                actuator_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_actuator")
            if actuator_id < 0:
                missing_actuators.append(name)
                continue

            self.joint_indices[name] = JointIndex(
                qpos=int(self.model.jnt_qposadr[joint_id]),
                qvel=int(self.model.jnt_dofadr[joint_id]),
                actuator=int(actuator_id),
            )

        if missing_joints or missing_actuators:
            details = []
            if missing_joints:
                details.append(f"missing joints: {missing_joints}")
            if missing_actuators:
                details.append(f"missing actuators: {missing_actuators}")
            raise ValueError("; ".join(details))

    def reset(self) -> None:
        """Reset the MuJoCo state to the default G0 pose."""

        self.mujoco.mj_resetData(self.model, self.data)
        default_q = cfg.get_default_joint_pos_array()
        for index, name in enumerate(self.joint_names):
            self.data.qpos[self.joint_indices[name].qpos] = default_q[index]
            self.data.qvel[self.joint_indices[name].qvel] = 0.0
        self.last_action.fill(0.0)
        self.sim_time = 0.0
        self._history.clear()
        self.mujoco.mj_forward(self.model, self.data)

    def get_joint_pos(self) -> np.ndarray:
        """Read G0 joint positions in policy/action order."""

        return np.asarray([self.data.qpos[self.joint_indices[name].qpos] for name in self.joint_names], dtype=np.float64)

    def get_joint_vel(self) -> np.ndarray:
        """Read G0 joint velocities in policy/action order."""

        return np.asarray([self.data.qvel[self.joint_indices[name].qvel] for name in self.joint_names], dtype=np.float64)

    def set_position_target(self, target_joint_pos: np.ndarray) -> None:
        """Set MuJoCo position actuator targets in policy/action order."""

        target_joint_pos = np.asarray(target_joint_pos, dtype=np.float64)
        if target_joint_pos.shape != (cfg.get_action_dim(),):
            raise ValueError(f"target_joint_pos shape must be ({cfg.get_action_dim()},), got {target_joint_pos.shape}")
        for index, name in enumerate(self.joint_names):
            self.data.ctrl[self.joint_indices[name].actuator] = target_joint_pos[index]

    def apply_policy_action(self, action: np.ndarray) -> np.ndarray:
        """Apply a policy action and return the target joint position."""

        target_joint_pos = cfg.compute_target_joint_pos(action)
        self.set_position_target(target_joint_pos)
        self.last_action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        return target_joint_pos

    def build_observation(self) -> np.ndarray:
        """Build flattened policy observation with history padding."""

        frame = self.build_observation_frame(
            joint_pos=self.get_joint_pos(),
            joint_vel=self.get_joint_vel(),
            last_action=self.last_action,
            command=self.command,
            sim_time=self.sim_time,
        )
        self._history.append(frame)
        self._history = self._history[-cfg.POLICY_HISTORY_LENGTH :]
        padded = [np.zeros_like(frame) for _ in range(cfg.POLICY_HISTORY_LENGTH - len(self._history))]
        return np.concatenate([*padded, *self._history])

    def step(self, action: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Apply an optional action, step MuJoCo for one control interval, and return obs/target."""

        if action is None:
            action = self.last_action
        target_joint_pos = self.apply_policy_action(action)
        substeps = max(1, int(round(cfg.CONTROL_DT / float(self.model.opt.timestep))))
        for _ in range(substeps):
            self.mujoco.mj_step(self.model, self.data)
        self.sim_time += cfg.CONTROL_DT
        return self.build_observation(), target_joint_pos

    def get_root_pose(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return root position and quaternion when a free root joint is available."""

        if self.model.nq < 7:
            return None, None
        return np.asarray(self.data.qpos[:3], dtype=np.float64), np.asarray(self.data.qpos[3:7], dtype=np.float64)
