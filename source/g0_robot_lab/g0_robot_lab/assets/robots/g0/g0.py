from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass
from . import g0_actuators


_G0_DIR = Path(__file__).resolve().parent
_G0_USD_PATH = _G0_DIR / "usd" / "g0.usd"


@configclass
class G0ArticulationCfg(ArticulationCfg):
    """Configuration for the G0 humanoid articulation."""

    joint_sdk_names: list[str] = None

    # Keep the joint position limits slightly conservative during early locomotion training.
    soft_joint_pos_limit_factor = 0.9


# -------------------------------------------------------------------------------
# Joint names
# -------------------------------------------------------------------------------

# This order comes directly from the current G0 URDF movable joint order.
# Do not change this list casually. It is useful for debugging USD/URDF joint order.
G0_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "l_shoulder_pitch_joint",
    "l_shoulder_roll_joint",
    "l_shoulder_yaw_joint",
    "l_elbow_pitch_joint",
    "r_shoulder_pitch_joint",
    "r_shoulder_roll_joint",
    "r_shoulder_yaw_joint",
    "r_elbow_pitch_joint",
    "l_hip_pitch_joint",
    "l_hip_roll_joint",
    "l_hip_yaw_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_hip_pitch_joint",
    "r_hip_roll_joint",
    "r_hip_yaw_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
]

G0_LEFT_LEG_JOINT_NAMES = [
    "l_hip_pitch_joint",
    "l_hip_roll_joint",
    "l_hip_yaw_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
]

G0_RIGHT_LEG_JOINT_NAMES = [
    "r_hip_pitch_joint",
    "r_hip_roll_joint",
    "r_hip_yaw_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
]

G0_LEG_JOINT_NAMES = G0_LEFT_LEG_JOINT_NAMES + G0_RIGHT_LEG_JOINT_NAMES

G0_WAIST_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
]

G0_LEFT_ARM_JOINT_NAMES = [
    "l_shoulder_pitch_joint",
    "l_shoulder_roll_joint",
    "l_shoulder_yaw_joint",
    "l_elbow_pitch_joint",
]

G0_RIGHT_ARM_JOINT_NAMES = [
    "r_shoulder_pitch_joint",
    "r_shoulder_roll_joint",
    "r_shoulder_yaw_joint",
    "r_elbow_pitch_joint",
]

G0_ARM_JOINT_NAMES = G0_LEFT_ARM_JOINT_NAMES + G0_RIGHT_ARM_JOINT_NAMES

G0_JOINT_SDK_NAMES = [
    "l_hip_pitch_joint",
    "l_hip_roll_joint",
    "l_hip_yaw_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_hip_pitch_joint",
    "r_hip_roll_joint",
    "r_hip_yaw_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "l_shoulder_pitch_joint",
    "l_shoulder_roll_joint",
    "l_shoulder_yaw_joint",
    "l_elbow_pitch_joint",
    "r_shoulder_pitch_joint",
    "r_shoulder_roll_joint",
    "r_shoulder_yaw_joint",
    "r_elbow_pitch_joint",
]

# -------------------------------------------------------------------------------
# Actuator hardware groups
# -------------------------------------------------------------------------------

# Six right-angle servos on the current G0 hardware.
# Assumption:
#   "ankle" refers to ankle_pitch joints.
# If the real right-angle ankle servos are ankle_roll instead, replace these two ankle_pitch names.
G0_RIGHT_ANGLE_SERVO_JOINT_NAMES = [
    "l_elbow_pitch_joint",
    "r_elbow_pitch_joint",
    "l_knee_pitch_joint",
    "r_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "r_ankle_pitch_joint",
]

# Remaining 16 joints use standard servos.
G0_STANDARD_SERVO_JOINT_NAMES = [
    name for name in G0_JOINT_SDK_NAMES if name not in G0_RIGHT_ANGLE_SERVO_JOINT_NAMES
]


# SDK / deployment-friendly order.
# This is not necessarily the same as the URDF traversal order.
# We put legs first because locomotion policy/action debugging usually focuses on legs first.



# -------------------------------------------------------------------------------
# Default standing pose
# -------------------------------------------------------------------------------

# First locomotion standing pose.
# The sagittal leg signs are mirrored because the left/right pitch axes in the
# G0 URDF are mirrored. These values describe the same physical pose on both
# sides. Keep these defaults at two-decimal precision so the standing baseline
# does not depend on unrealistically fine deployment accuracy.
G0_DEFAULT_JOINT_POS = {
    # waist
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,

    # left leg
    "l_hip_pitch_joint": -0.20,
    "l_hip_roll_joint": 0.0,
    "l_hip_yaw_joint": 0.0,
    "l_knee_pitch_joint": -0.34,
    "l_ankle_pitch_joint": 0.14,
    "l_ankle_roll_joint": 0.0,

    # right leg
    "r_hip_pitch_joint": 0.20,
    "r_hip_roll_joint": 0.0,
    "r_hip_yaw_joint": 0.0,
    "r_knee_pitch_joint": 0.34,
    "r_ankle_pitch_joint": -0.14,
    "r_ankle_roll_joint": 0.0,

    # left arm
    "l_shoulder_pitch_joint": -0.30,
    "l_shoulder_roll_joint": -0.25,
    "l_shoulder_yaw_joint": 0.0,
    "l_elbow_pitch_joint": 0.97,

    # right arm
    "r_shoulder_pitch_joint": 0.30,
    "r_shoulder_roll_joint": 0.25,
    "r_shoulder_yaw_joint": 0.0,
    "r_elbow_pitch_joint": -0.97,
}


# -------------------------------------------------------------------------------
# Robot cfg
# -------------------------------------------------------------------------------

G0_CFG = G0ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_G0_USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Formal locomotion must use real gravity.
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Keep self-collision disabled for the first G0 locomotion stage.
            # URDF-converted humanoid assets often have overlapping collision meshes.
            # After the default standing pose is validated, this can be revisited.
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Matches the current standing pose foot mesh height closely enough to
        # avoid a visible initial drop while keeping both feet in contact.
        pos=(0.0, 0.0, 0.23),
        joint_pos=G0_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 16 standard servos.
        # These include:
        #   hips except knee,
        #   ankle_roll,
        #   waist,
        #   shoulder joints.
        "standard_servos": ImplicitActuatorCfg(
            joint_names_expr=G0_STANDARD_SERVO_JOINT_NAMES,
            effort_limit_sim=g0_actuators.STANDARD_SERVO_RATED_TORQUE,
            velocity_limit_sim=g0_actuators.STANDARD_SERVO_MAX_VELOCITY,
            stiffness={
                ".*_hip_pitch_joint": 6.0,
                ".*_hip_roll_joint": 5.0,
                ".*_hip_yaw_joint": 3.0,
                ".*_ankle_roll_joint": 4.5,
                "waist_yaw_joint": 2.0,
                "waist_roll_joint": 2.0,
                ".*_shoulder_pitch_joint": 1.5,
                ".*_shoulder_roll_joint": 1.5,
                ".*_shoulder_yaw_joint": 1.5,
            },
            damping={
                ".*_hip_pitch_joint": 0.18,
                ".*_hip_roll_joint": 0.16,
                ".*_hip_yaw_joint": 0.10,
                ".*_ankle_roll_joint": 0.15,
                "waist_yaw_joint": 0.08,
                "waist_roll_joint": 0.08,
                ".*_shoulder_pitch_joint": 0.06,
                ".*_shoulder_roll_joint": 0.06,
                ".*_shoulder_yaw_joint": 0.06,
            },
            armature=g0_actuators.STANDARD_SERVO_ARMATURE,
        ),

        # 6 right-angle servos.
        # Current assumption:
        #   elbows + knees + ankle_pitch joints.
        "right_angle_servos": ImplicitActuatorCfg(
            joint_names_expr=G0_RIGHT_ANGLE_SERVO_JOINT_NAMES,
            effort_limit_sim=g0_actuators.RIGHT_ANGLE_SERVO_RATED_TORQUE,
            velocity_limit_sim=g0_actuators.RIGHT_ANGLE_SERVO_MAX_VELOCITY,
            stiffness={
                ".*_knee_pitch_joint": 8.0,
                ".*_ankle_pitch_joint": 6.5,
                ".*_elbow_pitch_joint": 2.0,
            },
            damping={
                ".*_knee_pitch_joint": 0.26,
                ".*_ankle_pitch_joint": 0.22,
                ".*_elbow_pitch_joint": 0.08,
            },
            armature=g0_actuators.RIGHT_ANGLE_SERVO_ARMATURE,
        ),
    },
    joint_sdk_names=G0_JOINT_SDK_NAMES,
)
