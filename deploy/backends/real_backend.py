"""Real-robot backend — kingkong G0 over CycloneDDS (mbus).

Drop-in sibling of ``MujocoBackend``: upper layers (FSM / managers) keep talking
through ``read_state()``; only the *control* path differs because the real
motors run **MIT-mode onboard PD**, so we send (q_des, kp, kd) ONCE per policy
step instead of computing torque in Python and sub-stepping. That one-shot
semantics lives in ``RealEnv`` below, which replaces ``ManagerBasedRLEnv``'s
per-substep PD loop.

FIRST-BRINGUP SCOPE (this file): only the two fully-known motor topics
    mc/motor_control  (PUB, MotorControl::Control)   <- we send commands
    mc/motor_state    (SUB, MotorControl::State)     <- we read q/dq + raw IMU
Orientation is derived from the embedded raw accelerometer (quasi-static only).
For walking, switch projected_gravity to the fused Imu::ImuOutput.q topic.

Wire/protocol facts (authoritative: kkos/.claude/skills/motor-control-dds.md):
  * angle = deg, angular velocity = deg/s, torque = Nm  ->  rad<->deg at the edge
  * CycloneDDS domain 0, QoS from /etc/mbus/config/mbus_qos.xml
  * MotorCmd{pos,kp,dq,kd,tau}; MotorState{isvalid,pos,dq,tau,status,fpc_temper,pcb_temper}
  * MotorControl::State.imu = ImuData{imu_data_valid,mag_data_valid,acc,gyro,mag}

THREE THINGS THAT MUST BE VERIFIED ON HARDWARE before this runs for real:
  (1) BUS_MOTOR_ORDER — the 22 motor index -> joint-name map for THIS humanoid
      (the value below is a PLACEHOLDER copied from MJ order and is almost
      certainly wrong; get the real map from the motor-controller config).
  (2) IMU_GYRO_IN_DEG — is imu.gyro deg/s or rad/s? (obs base_ang_vel is rad/s)
  (3) ACC_SIGN — sign so that upright proj_grav_b ~ [0,0,-1] (static test).
"""

import time
import numpy as np

from ..common.rotation import quat_wxyz_from_gravity

try:
    from mbus.wrapper import MbusNode
    from mbus.idl.MotorControl import Control as McControl, MotorCmd
except Exception as _e:  # mbus only present on the deploy/robot machine
    MbusNode = None
    _MBUS_IMPORT_ERROR = _e


_DEG2RAD = np.pi / 180.0
_RAD2DEG = 180.0 / np.pi

# TODO(hardware): replace with the real motor-index -> joint-name order from the
# kingkong motor controller. motion-planner's order is HEXAPOD; do NOT reuse it.
# Placeholder = same as G0_JOINT_NAMES_MJ so the file is runnable for wiring tests.
BUS_MOTOR_ORDER = None  # e.g. ["waist_yaw_joint", "waist_roll_joint", ...] len 22


class RealBackend:
    def __init__(self, joint_names_mj, *, domain_id: int = 0,
                 bus_motor_order=None,
                 imu_gyro_in_deg: bool = False,   # TODO(hardware) verify
                 acc_sign: float = -1.0,          # TODO(hardware) verify
                 motor_count: int = 22):
        if MbusNode is None:
            raise RuntimeError(
                f"mbus python bindings unavailable: {_MBUS_IMPORT_ERROR}. "
                "Install cyclonedds>=11, cyclonedds-python>=11, libmbus>=2.0."
            )
        self.joint_names_mj = list(joint_names_mj)
        self.motor_count = int(motor_count)
        self.imu_gyro_in_deg = bool(imu_gyro_in_deg)
        self.acc_sign = float(acc_sign)

        # ---- bus<->MJ index maps
        bus_order = bus_motor_order or BUS_MOTOR_ORDER or list(self.joint_names_mj)
        if bus_order == list(self.joint_names_mj) and bus_motor_order is None \
                and BUS_MOTOR_ORDER is None:
            print("[RealBackend] WARNING: BUS_MOTOR_ORDER not set — using MJ "
                  "order as a PLACEHOLDER. Joint mapping is almost certainly "
                  "wrong; set the real motor-index map before driving hardware.")
        bus_index = {n: i for i, n in enumerate(bus_order)}
        # mj_to_bus[k] = bus slot holding joint joint_names_mj[k]
        self.mj_to_bus = np.array(
            [bus_index[n] for n in self.joint_names_mj], dtype=np.int64
        )

        # ---- DDS
        self.node = MbusNode(domain_id=domain_id)
        self.ctrl_topic = self.node.register_topic(
            "mc/motor_control", McControl, "mbus::MotorControl_Control")
        self.state_topic = self.node.register_topic(
            "mc/motor_state", __import__(
                "mbus.idl.MotorControl", fromlist=["State"]).State,
            "mbus::MotorControl_State")

        self._seq = 1
        self._t0 = time.monotonic()
        self._last_state = None  # cache last good MotorControl::State

    # ---- state read (returns the exact dict contract MujocoBackend uses) ----

    def read_state(self):
        samples = self.state_topic.read(1)
        if samples:
            self._last_state = samples[0]
        s = self._last_state
        if s is None:
            raise RuntimeError("No mc/motor_state received yet (motors offline?)")

        n = self.motor_count
        # bus arrays (deg / deg/s) -> rad
        bus_pos = np.array([s.motors[i].pos for i in range(n)], dtype=np.float64)
        bus_dq = np.array([s.motors[i].dq for i in range(n)], dtype=np.float64)
        q_bus = bus_pos * _DEG2RAD
        dq_bus = bus_dq * _DEG2RAD
        # bus order -> MJ order
        q_mj = q_bus[self.mj_to_bus]
        dq_mj = dq_bus[self.mj_to_bus]

        imu = s.imu
        gyro = np.array([imu.gyro.x, imu.gyro.y, imu.gyro.z], dtype=np.float64)
        if self.imu_gyro_in_deg:
            gyro = gyro * _DEG2RAD
        acc = np.array([imu.acc.x, imu.acc.y, imu.acc.z], dtype=np.float64)
        base_quat_wxyz = quat_wxyz_from_gravity(acc, sign=self.acc_sign)

        return {
            "q_mj": q_mj,
            "dq_mj": dq_mj,
            "base_quat_wxyz": base_quat_wxyz,
            "base_ang_vel_b": gyro,
            # base pos / lin vel are NOT in the obs; the elastic band (sim-only)
            # is the only consumer and is disabled on the real robot.
            "base_pos_w": np.zeros(3),
            "base_lin_vel_w": np.zeros(3),
            "sim_time": time.monotonic() - self._t0,
        }

    # ---- control: MIT-mode one-shot send (MJ order in) --------------------

    def apply_joint_command(self, q_des_mj, kp_mj, kd_mj, tau_ff_mj=None):
        """Send one MotorControl::Control frame. q_des/kp/kd/tau in MJ order."""
        q_des_mj = np.asarray(q_des_mj, dtype=np.float64)
        kp_mj = np.asarray(kp_mj, dtype=np.float64)
        kd_mj = np.asarray(kd_mj, dtype=np.float64)
        tau_ff_mj = np.zeros(self.motor_count) if tau_ff_mj is None \
            else np.asarray(tau_ff_mj, dtype=np.float64)

        motors = [MotorCmd() for _ in range(self.motor_count)]
        for k in range(len(q_des_mj)):
            b = int(self.mj_to_bus[k])
            motors[b].pos = float(q_des_mj[k] * _RAD2DEG)   # rad -> deg
            motors[b].dq = 0.0                              # target vel 0
            motors[b].kp = float(kp_mj[k])                  # TODO(hardware): kp unit (Nm/rad vs Nm/deg)
            motors[b].kd = float(kd_mj[k])
            motors[b].tau = float(tau_ff_mj[k])             # Nm, unchanged
        msg = McControl(
            timestamp_ns=time.time_ns(),
            sequence_id=self._seq & 0xFFFF,
            motor_count=self.motor_count,
            motors=motors,
        )
        self._seq += 1
        self.ctrl_topic.write(msg)

    def apply_damping(self, kd_mj):
        """Safe limp/shutdown: zero stiffness, damping only (NOT zero torque).

        Real Passive must NOT be tau=0 (the robot would collapse). Hold current
        pose stiffness-free with kd damping so it settles softly.
        """
        kd_mj = np.asarray(kd_mj, dtype=np.float64)
        s = self.read_state()
        self.apply_joint_command(s["q_mj"], np.zeros_like(kd_mj), kd_mj)

    # ---- elastic-band surface: sim-only, no-ops on hardware so the shared
    # tick()/ElasticBand path runs unchanged (band is forced off for real).

    def apply_external_force(self, force_w):
        pass

    def clear_external_force(self):
        pass


# ---------------------------------------------------------------------------
# RealEnv — replaces ManagerBasedRLEnv's per-substep Python PD with one MIT
# command per policy step, paced to step_dt. Backend does the PD onboard.
# ---------------------------------------------------------------------------

from ..isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv


class RealEnv(ManagerBasedRLEnv):
    def __init__(self, *args, step_dt: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_dt = float(step_dt)
        self._next_t = time.monotonic()

    def _pace(self):
        self._next_t += self.step_dt
        sleep = self._next_t - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        else:
            self._next_t = time.monotonic()  # fell behind; resync

    def step_policy(self, action_sdk):
        target_q_mj = self.act_mgr.compute_target_q_mj(action_sdk)
        self.backend.apply_joint_command(
            target_q_mj, self.act_mgr.kp_mj, self.act_mgr.kd_mj)
        self._pace()
        return self._build_obs()

    def step_to_target_q(self, target_q_mj):
        self.backend.apply_joint_command(
            target_q_mj, self.act_mgr.kp_mj, self.act_mgr.kd_mj)
        self._pace()
        return self._build_obs()

    def step_torque(self, tau_mj):
        # Passive on real hardware = damping (kd only), NOT zero torque — a
        # tau=0 limp robot would collapse. This is the software E-stop ('p').
        self.backend.apply_damping(self.act_mgr.kd_mj)
        self._pace()
        return self._build_obs()
