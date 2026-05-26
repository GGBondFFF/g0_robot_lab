"""Python mirror of cpp/include/sim_frame_proto.hpp.

Three fixed-size little-endian binary frames travel over the UDS bridges.
This module exposes struct.Struct encoders/decoders + namedtuple-style
dataclasses so the policy can decode each channel without dragging in
ctypes / pybind / cyclonedds.

Keep byte layouts identical to the C++ header — the regression test for
that is byte-counting an end-to-end round-trip (Phase 1 verified with
`nc -U` + wc).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional

# ─── magic / version (must match sim_frame_proto.hpp) ─────────────────────
MAGIC_MOTOR_CONTROL = 0x47305343  # 'G0SC'
MAGIC_MOTOR_STATE   = 0x47305353  # 'G0SS'
MAGIC_ROBOT_CONTROL = 0x47305243  # 'G0RC'
WIRE_VERSION = 1
NUM_MOTORS = 22

# ─── shared 24-byte FrameHeader ───────────────────────────────────────────
# struct FrameHeader { u32 magic; u16 version; u16 payload_count;
#                      u64 timestamp_ns; u64 sequence_id; }
_HEADER = struct.Struct("<IHHQQ")
HEADER_BYTES = _HEADER.size
assert HEADER_BYTES == 24

# ─── MotorControlFrame (Python policy -> DDS sim) 464 B ───────────────────
# Each MotorCmdWire: <5 floats> = 20B, 22 of them.
_MC_MOTORS = struct.Struct("<" + "fffff" * NUM_MOTORS)
MC_FRAME_BYTES = HEADER_BYTES + _MC_MOTORS.size
assert MC_FRAME_BYTES == 464


@dataclass
class MotorCmd:
    pos: float = 0.0  # degrees
    dq: float = 0.0   # deg/s
    kp: float = 0.0   # N*m / deg
    kd: float = 0.0   # N*m / (deg/s)
    tau: float = 0.0  # N*m


@dataclass
class MotorControlFrame:
    timestamp_ns: int = 0
    sequence_id: int = 0
    motors: List[MotorCmd] = field(
        default_factory=lambda: [MotorCmd() for _ in range(NUM_MOTORS)]
    )

    def encode(self) -> bytes:
        flat = []
        for m in self.motors:
            flat.extend((m.pos, m.dq, m.kp, m.kd, m.tau))
        return (
            _HEADER.pack(MAGIC_MOTOR_CONTROL, WIRE_VERSION, NUM_MOTORS,
                         int(self.timestamp_ns), int(self.sequence_id))
            + _MC_MOTORS.pack(*flat)
        )


def decode_motor_control(buf: bytes) -> Optional[MotorControlFrame]:
    if len(buf) != MC_FRAME_BYTES:
        return None
    magic, ver, cnt, ts, seq = _HEADER.unpack_from(buf, 0)
    if magic != MAGIC_MOTOR_CONTROL or ver != WIRE_VERSION or cnt != NUM_MOTORS:
        return None
    vals = _MC_MOTORS.unpack_from(buf, HEADER_BYTES)
    frame = MotorControlFrame(timestamp_ns=ts, sequence_id=seq, motors=[])
    for i in range(NUM_MOTORS):
        p, dq, kp, kd, tau = vals[i * 5 : i * 5 + 5]
        frame.motors.append(MotorCmd(p, dq, kp, kd, tau))
    return frame


# ─── MotorStateFrame (Isaac plant -> Python policy) 554 B ─────────────────
# MotorStateWire: <B f f f B f f> with pack(1) -> 22 bytes
# We have to honor pack(1) — no padding between fields. struct '<BfffBff' on
# little-endian gives 1+4+4+4+1+4+4 = 22 with no native alignment because the
# '<' prefix already disables alignment.
_MS_MOTOR = struct.Struct("<BfffBff")
assert _MS_MOTOR.size == 22
_MS_MOTORS = struct.Struct("<" + "BfffBff" * NUM_MOTORS)

# ImuDataWire: <B B I I 9f> = 1+1+4+4+36 = 46
_MS_IMU = struct.Struct("<BBII9f")
assert _MS_IMU.size == 46

MS_FRAME_BYTES = HEADER_BYTES + NUM_MOTORS * 22 + 46
assert MS_FRAME_BYTES == 554


@dataclass
class MotorState:
    isvalid: int = 0
    pos: float = 0.0       # degrees
    dq: float = 0.0        # deg/s
    tau: float = 0.0       # N*m
    status: int = 0
    fpc_temper: float = 0.0
    pcb_temper: float = 0.0


@dataclass
class ImuData:
    imu_valid: int = 0
    mag_valid: int = 0
    imu_timestamp: int = 0
    mag_timestamp: int = 0
    acc: tuple = (0.0, 0.0, 0.0)    # m/s^2
    gyro: tuple = (0.0, 0.0, 0.0)   # rad/s
    mag: tuple = (0.0, 0.0, 0.0)


@dataclass
class MotorStateFrame:
    timestamp_ns: int = 0
    sequence_id: int = 0
    motors: List[MotorState] = field(
        default_factory=lambda: [MotorState() for _ in range(NUM_MOTORS)]
    )
    imu: ImuData = field(default_factory=ImuData)

    def encode(self) -> bytes:
        flat_m = []
        for m in self.motors:
            flat_m.extend((m.isvalid, m.pos, m.dq, m.tau,
                           m.status, m.fpc_temper, m.pcb_temper))
        return (
            _HEADER.pack(MAGIC_MOTOR_STATE, WIRE_VERSION, NUM_MOTORS,
                         int(self.timestamp_ns), int(self.sequence_id))
            + _MS_MOTORS.pack(*flat_m)
            + _MS_IMU.pack(
                self.imu.imu_valid, self.imu.mag_valid,
                self.imu.imu_timestamp, self.imu.mag_timestamp,
                *self.imu.acc, *self.imu.gyro, *self.imu.mag,
            )
        )


def decode_motor_state(buf: bytes) -> Optional[MotorStateFrame]:
    if len(buf) != MS_FRAME_BYTES:
        return None
    magic, ver, cnt, ts, seq = _HEADER.unpack_from(buf, 0)
    if magic != MAGIC_MOTOR_STATE or ver != WIRE_VERSION or cnt != NUM_MOTORS:
        return None
    flat = _MS_MOTORS.unpack_from(buf, HEADER_BYTES)
    frame = MotorStateFrame(timestamp_ns=ts, sequence_id=seq, motors=[])
    for i in range(NUM_MOTORS):
        iv, p, dq, tau, st, fpc, pcb = flat[i * 7 : i * 7 + 7]
        frame.motors.append(MotorState(iv, p, dq, tau, st, fpc, pcb))
    imu_vals = _MS_IMU.unpack_from(buf, HEADER_BYTES + _MS_MOTORS.size)
    frame.imu = ImuData(
        imu_valid=imu_vals[0], mag_valid=imu_vals[1],
        imu_timestamp=imu_vals[2], mag_timestamp=imu_vals[3],
        acc=imu_vals[4:7], gyro=imu_vals[7:10], mag=imu_vals[10:13],
    )
    return frame


# ─── RobotControlFrame (operator -> Python policy) 126 B ──────────────────
# After 24B header:
#   mode(B) has_l(B) has_r(B) reserved(B)             -> 4 B
#   BodyState: 10 floats                              -> 40 B
#   GripperWire left:  <B 7f>                         -> 29 B
#   GripperWire right: <B 7f>                         -> 29 B
_RC_HEAD  = struct.Struct("<BBBB")
_RC_BODY  = struct.Struct("<10f")
_RC_GRIP  = struct.Struct("<B7f")
assert _RC_HEAD.size == 4 and _RC_BODY.size == 40 and _RC_GRIP.size == 29
RC_FRAME_BYTES = HEADER_BYTES + 4 + 40 + 29 + 29
assert RC_FRAME_BYTES == 126


@dataclass
class GripperWire:
    is_valid: int = 0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    lin_x: float = 0.0
    lin_y: float = 0.0
    lin_z: float = 0.0
    opening: float = 0.0


@dataclass
class BodyState:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    lin_x: float = 0.0     # m/s, normalised to [-1,+1] by fake_control_agent
    lin_y: float = 0.0
    lin_z: float = 0.0
    ang_x: float = 0.0     # rad/s on the wire (deg/s -> rad/s done in C++)
    ang_y: float = 0.0
    ang_z: float = 0.0
    height: float = 0.0


@dataclass
class RobotControlFrame:
    timestamp_ns: int = 0
    sequence_id: int = 0
    mode: int = 0
    has_left_gripper: int = 0
    has_right_gripper: int = 0
    body: BodyState = field(default_factory=BodyState)
    left_gripper: GripperWire = field(default_factory=GripperWire)
    right_gripper: GripperWire = field(default_factory=GripperWire)


def decode_robot_control(buf: bytes) -> Optional[RobotControlFrame]:
    if len(buf) != RC_FRAME_BYTES:
        return None
    magic, ver, cnt, ts, seq = _HEADER.unpack_from(buf, 0)
    if magic != MAGIC_ROBOT_CONTROL or ver != WIRE_VERSION:
        return None
    off = HEADER_BYTES
    mode, has_l, has_r, _ = _RC_HEAD.unpack_from(buf, off); off += 4
    body_vals = _RC_BODY.unpack_from(buf, off); off += 40
    body = BodyState(*body_vals)
    l_vals = _RC_GRIP.unpack_from(buf, off); off += 29
    r_vals = _RC_GRIP.unpack_from(buf, off); off += 29
    return RobotControlFrame(
        timestamp_ns=ts, sequence_id=seq,
        mode=mode, has_left_gripper=has_l, has_right_gripper=has_r,
        body=body,
        left_gripper=GripperWire(*l_vals),
        right_gripper=GripperWire(*r_vals),
    )
