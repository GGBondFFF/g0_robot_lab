"""G0 Unitree LowCmd-style MuJoCo bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from scripts.sim2sim import g0_sim2sim_config as cfg


@dataclass
class G0MotorCmd:
    q: float = 0.0
    dq: float = 0.0
    tau: float = 0.0
    kp: float = 0.0
    kd: float = 0.0


@dataclass
class G0MotorState:
    q: float = 0.0
    dq: float = 0.0
    tau_est: float = 0.0


@dataclass
class G0LowCmd:
    motor_cmd: list[G0MotorCmd] = field(default_factory=list)

    @classmethod
    def zeros(cls, motor_count: int) -> "G0LowCmd":
        return cls([G0MotorCmd() for _ in range(motor_count)])


@dataclass
class G0LowState:
    motor_state: list[G0MotorState] = field(default_factory=list)
    imu_quat: np.ndarray = field(default_factory=lambda: np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    imu_gyro: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    imu_acc: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))


def import_mujoco() -> Any:
    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install the `mujoco` Python package to run the Unitree-style G0 prototype.") from exc
    return mujoco


@dataclass(frozen=True)
class JointIndex:
    qpos: int
    qvel: int
    actuator: int


class G0UnitreeMujocoBridge:
    """A Python equivalent of the unitree_mujoco LowCmd control path for G0."""

    def __init__(self, model_path: str | Path, command: np.ndarray | None = None):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"MuJoCo model does not exist: {self.model_path}")
        self.mujoco = import_mujoco()
        self.model = self.mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = self.mujoco.MjData(self.model)
        self.joint_names = cfg.get_joint_names()
        self.joint_indices: dict[str, JointIndex] = {}
        self.specs = cfg.get_isaac_actuator_specs()
        self.kp = np.asarray([self.specs[name].stiffness for name in self.joint_names], dtype=np.float64)
        self.kd = np.asarray([self.specs[name].damping for name in self.joint_names], dtype=np.float64)
        self.effort_limit = np.asarray([self.specs[name].effort_limit_sim for name in self.joint_names], dtype=np.float64)
        self.default_joint_pos = cfg.get_default_joint_pos_array()
        self.command = np.zeros(3, dtype=np.float64) if command is None else np.asarray(command, dtype=np.float64)
        self.last_action = np.zeros(cfg.get_action_dim(), dtype=np.float64)
        self.sim_time = 0.0
        self._history: list[np.ndarray] = []
        self.last_lowcmd = G0LowCmd.zeros(cfg.get_action_dim())
        self.last_tau_cmd = np.zeros(cfg.get_action_dim(), dtype=np.float64)
        self.last_tau_cmd_clipped = np.zeros(cfg.get_action_dim(), dtype=np.float64)
        self.validate_model()
        self.reset()

    def validate_model(self) -> None:
        if self.model.nu != cfg.get_action_dim():
            raise ValueError(f"Expected {cfg.get_action_dim()} MuJoCo actuators, got {self.model.nu}")
        missing: list[str] = []
        for name in self.joint_names:
            joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, name)
            actuator_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if joint_id < 0 or actuator_id < 0:
                missing.append(name)
                continue
            self.joint_indices[name] = JointIndex(
                qpos=int(self.model.jnt_qposadr[joint_id]),
                qvel=int(self.model.jnt_dofadr[joint_id]),
                actuator=int(actuator_id),
            )
        if missing:
            raise ValueError(f"Missing G0 policy joints/actuators in Unitree-style MJCF: {missing}")
        actuator_order = [
            self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_ACTUATOR, index)
            for index in range(self.model.nu)
        ]
        if actuator_order != self.joint_names:
            raise ValueError(f"Actuator order must match policy order: {actuator_order} != {self.joint_names}")

    def reset(self) -> None:
        self.mujoco.mj_resetData(self.model, self.data)
        for index, name in enumerate(self.joint_names):
            self.data.qpos[self.joint_indices[name].qpos] = self.default_joint_pos[index]
            self.data.qvel[self.joint_indices[name].qvel] = 0.0
        self.last_action.fill(0.0)
        self.last_tau_cmd.fill(0.0)
        self.last_tau_cmd_clipped.fill(0.0)
        self.sim_time = 0.0
        self._history.clear()
        self.mujoco.mj_forward(self.model, self.data)

    def get_joint_pos(self) -> np.ndarray:
        return np.asarray([self.data.qpos[self.joint_indices[name].qpos] for name in self.joint_names], dtype=np.float64)

    def get_joint_vel(self) -> np.ndarray:
        return np.asarray([self.data.qvel[self.joint_indices[name].qvel] for name in self.joint_names], dtype=np.float64)

    def get_motor_tau_est(self) -> np.ndarray:
        return np.asarray([self.data.actuator_force[self.joint_indices[name].actuator] for name in self.joint_names], dtype=np.float64)

    def get_low_state(self) -> G0LowState:
        q = self.get_joint_pos()
        dq = self.get_joint_vel()
        tau_est = self.get_motor_tau_est()
        return G0LowState(
            motor_state=[G0MotorState(float(q[i]), float(dq[i]), float(tau_est[i])) for i in range(cfg.get_action_dim())],
            imu_quat=self.get_root_pose()[1],
            imu_gyro=self.get_base_ang_vel(),
            imu_acc=np.zeros(3, dtype=np.float64),
        )

    def build_lowcmd(self, target_joint_pos: np.ndarray) -> G0LowCmd:
        target_joint_pos = np.asarray(target_joint_pos, dtype=np.float64)
        if target_joint_pos.shape != (cfg.get_action_dim(),):
            raise ValueError(f"target_joint_pos shape must be ({cfg.get_action_dim()},), got {target_joint_pos.shape}")
        return G0LowCmd(
            [
                G0MotorCmd(
                    q=float(target_joint_pos[i]),
                    dq=0.0,
                    tau=0.0,
                    kp=float(self.kp[i]),
                    kd=float(self.kd[i]),
                )
                for i in range(cfg.get_action_dim())
            ]
        )

    def apply_lowcmd(self, lowcmd: G0LowCmd) -> tuple[np.ndarray, np.ndarray]:
        if len(lowcmd.motor_cmd) != cfg.get_action_dim():
            raise ValueError(f"Expected {cfg.get_action_dim()} motor commands, got {len(lowcmd.motor_cmd)}")
        q = self.get_joint_pos()
        dq = self.get_joint_vel()
        tau_cmd = np.zeros(cfg.get_action_dim(), dtype=np.float64)
        for index, motor_cmd in enumerate(lowcmd.motor_cmd):
            tau_cmd[index] = motor_cmd.tau + motor_cmd.kp * (motor_cmd.q - q[index]) + motor_cmd.kd * (motor_cmd.dq - dq[index])
        tau_cmd_clipped = np.clip(tau_cmd, -self.effort_limit, self.effort_limit)
        for index, name in enumerate(self.joint_names):
            self.data.ctrl[self.joint_indices[name].actuator] = tau_cmd_clipped[index]
        self.last_lowcmd = lowcmd
        self.last_tau_cmd = tau_cmd
        self.last_tau_cmd_clipped = tau_cmd_clipped
        return tau_cmd, tau_cmd_clipped

    def step_control_interval(self) -> None:
        substeps = max(1, int(round(cfg.CONTROL_DT / float(self.model.opt.timestep))))
        for _ in range(substeps):
            self.mujoco.mj_step(self.model, self.data)
        self.sim_time += cfg.CONTROL_DT

    def step_lowcmd(self, lowcmd: G0LowCmd, last_action: np.ndarray | None = None) -> np.ndarray:
        self.apply_lowcmd(lowcmd)
        if last_action is not None:
            cfg.validate_action_shape(np.asarray(last_action, dtype=np.float64))
            self.last_action = np.clip(np.asarray(last_action, dtype=np.float64), -1.0, 1.0)
        self.step_control_interval()
        return self.build_observation()

    def policy_action_to_target(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        clipped = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        return clipped, self.default_joint_pos + cfg.ACTION_SCALE * clipped

    def get_root_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if self.model.nq < 7:
            return np.full(3, np.nan), np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return np.asarray(self.data.qpos[:3], dtype=np.float64).copy(), np.asarray(self.data.qpos[3:7], dtype=np.float64).copy()

    @staticmethod
    def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
        quat = np.asarray(quat, dtype=np.float64)
        norm = np.linalg.norm(quat)
        if norm == 0.0:
            return np.eye(3)
        w, x, y, z = quat / norm
        return np.asarray(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    def get_base_ang_vel(self) -> np.ndarray:
        if self.model.nv < 6:
            return np.zeros(3, dtype=np.float64)
        _, quat = self.get_root_pose()
        return self.quat_wxyz_to_matrix(quat).T @ np.asarray(self.data.qvel[3:6], dtype=np.float64)

    def get_projected_gravity(self) -> np.ndarray:
        _, quat = self.get_root_pose()
        return self.quat_wxyz_to_matrix(quat).T @ np.asarray([0.0, 0.0, -1.0], dtype=np.float64)

    def build_observation_frame(self) -> np.ndarray:
        joint_pos_rel = self.get_joint_pos() - self.default_joint_pos
        phase = (self.sim_time % cfg.GAIT_PERIOD) / cfg.GAIT_PERIOD
        gait_phase = np.asarray([np.sin(2.0 * np.pi * phase), np.cos(2.0 * np.pi * phase)], dtype=np.float64)
        return np.concatenate(
            [
                self.get_base_ang_vel() * 0.2,
                self.get_projected_gravity(),
                self.command,
                joint_pos_rel,
                self.get_joint_vel() * 0.05,
                self.last_action,
                gait_phase,
            ]
        )

    def build_observation(self) -> np.ndarray:
        frame = self.build_observation_frame()
        if not self._history:
            self._history = [frame.copy() for _ in range(cfg.POLICY_HISTORY_LENGTH)]
        else:
            self._history.append(frame)
        self._history = self._history[-cfg.POLICY_HISTORY_LENGTH :]
        return cfg.flatten_history_term_major(self._history)
