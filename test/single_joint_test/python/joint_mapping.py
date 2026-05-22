"""G0 motor_id <-> joint_name mapping for the virtual single-joint DDS test.

This module is *only* for the Isaac Lab virtual DDS test under
``test/single_joint_test``. It must NOT be used to drive real hardware.

The ``sim_sign_observed`` column was hand-verified by visual inspection in
Isaac Lab GUI; it describes whether the joint's positive rotation in the
*sim* already follows the right-hand rule under the robot body frame
(forward = +X). Public API uses right-hand-rule values; ``apply_sign``
converts them to the sim's native sign.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class JointEntry:
    motor_id: int
    joint_name: str
    sim_sign_observed: int  # +1 or -1


_TABLE: Tuple[JointEntry, ...] = (
    JointEntry(1,  "waist_yaw_joint",         -1),
    JointEntry(2,  "waist_roll_joint",        -1),
    JointEntry(3,  "l_shoulder_pitch_joint",  -1),
    JointEntry(4,  "l_shoulder_roll_joint",   -1),
    JointEntry(5,  "l_shoulder_yaw_joint",    -1),
    JointEntry(6,  "l_elbow_pitch_joint",     -1),
    JointEntry(7,  "r_shoulder_pitch_joint",  +1),
    JointEntry(8,  "r_shoulder_roll_joint",   -1),
    JointEntry(9,  "r_shoulder_yaw_joint",    -1),
    JointEntry(10, "r_elbow_pitch_joint",     +1),
    JointEntry(11, "l_hip_pitch_joint",       +1),
    JointEntry(12, "l_hip_roll_joint",        -1),
    JointEntry(13, "l_hip_yaw_joint",         -1),
    JointEntry(14, "l_knee_pitch_joint",      -1),
    JointEntry(15, "l_ankle_pitch_joint",     -1),
    JointEntry(16, "l_ankle_roll_joint",      -1),
    JointEntry(17, "r_hip_pitch_joint",       -1),
    JointEntry(18, "r_hip_roll_joint",        -1),
    JointEntry(19, "r_hip_yaw_joint",         -1),
    JointEntry(20, "r_knee_pitch_joint",      +1),
    JointEntry(21, "r_ankle_pitch_joint",     +1),
    JointEntry(22, "r_ankle_roll_joint",      -1),
)

NUM_MOTORS: int = 22

JOINT_TABLE: Tuple[JointEntry, ...] = _TABLE
_BY_ID: Dict[int, JointEntry] = {e.motor_id: e for e in _TABLE}
_BY_NAME: Dict[str, JointEntry] = {e.joint_name: e for e in _TABLE}


def get_entry(motor_id: int) -> JointEntry:
    if motor_id not in _BY_ID:
        raise KeyError("unknown motor_id: %d (valid 1..22)" % motor_id)
    return _BY_ID[motor_id]


def motor_id_to_joint_name(motor_id: int) -> str:
    return get_entry(motor_id).joint_name


def sim_sign_for_motor(motor_id: int) -> int:
    return get_entry(motor_id).sim_sign_observed


def apply_sign(motor_id: int, desired_rhr_value: float) -> float:
    """Convert a right-hand-rule command into the value that should be sent
    to Isaac Lab so the observed motion matches RHR."""
    return sim_sign_for_motor(motor_id) * desired_rhr_value


def build_single_joint_frame(motor_id: int, desired_rhr_value: float) -> List[float]:
    """Return a 22-length list where only the target motor slot is set to the
    sign-converted command; all other slots are 0.0.

    Slot index = motor_id - 1 (1-based motor id, 0-based list).
    """
    if motor_id not in _BY_ID:
        raise KeyError("unknown motor_id: %d" % motor_id)
    frame = [0.0] * NUM_MOTORS
    frame[motor_id - 1] = apply_sign(motor_id, desired_rhr_value)
    return frame


def validate_against_articulation(joint_names: Sequence[str]) -> Dict[int, int]:
    """Resolve every motor_id to an articulation joint index.

    Returns a dict ``{motor_id: articulation_joint_index}``. Raises
    ``ValueError`` if any of the 22 mapped joint names is missing from the
    articulation's joint_names list.
    """
    name_to_idx = {n: i for i, n in enumerate(joint_names)}
    missing = [e.joint_name for e in _TABLE if e.joint_name not in name_to_idx]
    if missing:
        raise ValueError(
            "articulation is missing %d mapped joint(s): %s"
            % (len(missing), missing)
        )
    return {e.motor_id: name_to_idx[e.joint_name] for e in _TABLE}
