from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass


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


# SDK / deployment-friendly order.
# This is not necessarily the same as the URDF traversal order.
# We put legs first because locomotion policy/action debugging usually focuses on legs first.
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
# Default standing pose
# -------------------------------------------------------------------------------

# First locomotion standing pose.
# This is adapted from the Unitree G1 style:
#   hip pitch: slight forward bend
#   knee pitch: slight bend
#   ankle pitch: compensation
#   shoulder/elbow: non-zero relaxed pose
#
# Important:
# If the knees bend in the wrong direction in GUI, flip the sign of knee/ankle pitch.
G0_DEFAULT_JOINT_POS = {
    # waist
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,

    # left leg
    "l_hip_pitch_joint": 0.0,
    "l_hip_roll_joint": 0.0,
    "l_hip_yaw_joint": 0.0,
    "l_knee_pitch_joint": -0.30,
    "l_ankle_pitch_joint": 0.30,
    "l_ankle_roll_joint": 0.0,

    # right leg
    "r_hip_pitch_joint": 0.0,
    "r_hip_roll_joint": 0.0,
    "r_hip_yaw_joint": 0.0,
    "r_knee_pitch_joint": 0.30,
    "r_ankle_pitch_joint": -0.30,
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
        # Initial estimate. Tune this after GUI checking foot-ground contact.
        pos=(0.0, 0.0, 0.235),
        joint_pos=G0_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Hip pitch, hip yaw and waist yaw are treated like the stronger G1-style group.
        "hip_pitch_yaw_waist_yaw": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_hip_yaw_joint",
                "waist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_hip_pitch_joint": 88.0,
                ".*_hip_yaw_joint": 88.0,
                "waist_yaw_joint": 88.0,
            },
            velocity_limit_sim={
                ".*_hip_pitch_joint": 32.0,
                ".*_hip_yaw_joint": 32.0,
                "waist_yaw_joint": 32.0,
            },
            stiffness={
                ".*_hip_pitch_joint": 100.0,
                ".*_hip_yaw_joint": 100.0,
                "waist_yaw_joint": 120.0,
            },
            damping={
                ".*_hip_pitch_joint": 2.0,
                ".*_hip_yaw_joint": 2.0,
                "waist_yaw_joint": 4.0,
            },
            armature=0.01,
        ),

        # Hip roll and knee pitch usually need higher torque.
        "hip_roll_knee": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_roll_joint",
                ".*_knee_pitch_joint",
            ],
            effort_limit_sim={
                ".*_hip_roll_joint": 139.0,
                ".*_knee_pitch_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_roll_joint": 20.0,
                ".*_knee_pitch_joint": 20.0,
            },
            stiffness={
                ".*_hip_roll_joint": 100.0,
                ".*_knee_pitch_joint": 150.0,
            },
            damping={
                ".*_hip_roll_joint": 2.0,
                ".*_knee_pitch_joint": 4.0,
            },
            armature=0.01,
        ),

        # Ankles and waist roll are lower-torque stabilizing joints.
        "ankles_waist_roll": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
                "waist_roll_joint",
            ],
            effort_limit_sim={
                ".*_ankle_pitch_joint": 35.0,
                ".*_ankle_roll_joint": 35.0,
                "waist_roll_joint": 35.0,
            },
            velocity_limit_sim={
                ".*_ankle_pitch_joint": 30.0,
                ".*_ankle_roll_joint": 30.0,
                "waist_roll_joint": 30.0,
            },
            stiffness={
                ".*_ankle_pitch_joint": 40.0,
                ".*_ankle_roll_joint": 40.0,
                "waist_roll_joint": 40.0,
            },
            damping={
                ".*_ankle_pitch_joint": 2.0,
                ".*_ankle_roll_joint": 2.0,
                "waist_roll_joint": 4.0,
            },
            armature=0.01,
        ),

        # Arms are kept softer than legs for early walking.
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
            ],
            effort_limit_sim=25.0,
            velocity_limit_sim=37.0,
            stiffness={
                ".*_shoulder_pitch_joint": 40.0,
                ".*_shoulder_roll_joint": 40.0,
                ".*_shoulder_yaw_joint": 40.0,
                ".*_elbow_pitch_joint": 40.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 1.0,
                ".*_shoulder_roll_joint": 1.0,
                ".*_shoulder_yaw_joint": 1.0,
                ".*_elbow_pitch_joint": 1.0,
            },
            armature=0.01,
        ),
    },
    joint_sdk_names=G0_JOINT_SDK_NAMES,
)
