"""Joint-ordering bridge for the policy_test stack.

Three orderings the policy traffic has to translate between:

  wire  : motor_id 1..22 from cpp/include/joint_mapping.hpp. This is what
          MotorControlFrame / MotorStateFrame use slot-for-slot.
  sdk   : G0_JOINT_SDK_NAMES from g0.py. The trained policy emits actions
          in this order (JointPositionActionCfg(joint_names=G0_JOINT_SDK_NAMES,
          preserve_order=True)).
  urdf  : Isaac Lab articulation joint order — what joint_pos / joint_vel
          observations come back in. The articulation traversal isn't fully
          derivable from g0.py alone; we default to alphabetical (Isaac Lab's
          usual sort) and let Phase 5 sanity-check by feeding zero command
          and verifying the resulting raw action is close to zero.

The class exposes integer index lookup tables (wire_to_sdk, wire_to_urdf,
sdk_to_wire) so per-tick reordering is just a `np.take`.

g0.py is imported via importlib without triggering g0_robot_lab/__init__.py
(which would pull in isaaclab_tasks). Same trick as
test/single_joint_test/python/codegen_pd_gains.py.
"""
from __future__ import annotations

import importlib.util
import math
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Motor id (1..22) -> joint name. Must stay in lockstep with
# cpp/include/joint_mapping.hpp (and single_joint_test's copy).
MOTOR_ID_TO_JOINT_NAME: List[str] = [
    "waist_yaw_joint", "waist_roll_joint",
    "l_shoulder_pitch_joint", "l_shoulder_roll_joint",
    "l_shoulder_yaw_joint", "l_elbow_pitch_joint",
    "r_shoulder_pitch_joint", "r_shoulder_roll_joint",
    "r_shoulder_yaw_joint", "r_elbow_pitch_joint",
    "l_hip_pitch_joint", "l_hip_roll_joint",
    "l_hip_yaw_joint", "l_knee_pitch_joint",
    "l_ankle_pitch_joint", "l_ankle_roll_joint",
    "r_hip_pitch_joint", "r_hip_roll_joint",
    "r_hip_yaw_joint", "r_knee_pitch_joint",
    "r_ankle_pitch_joint", "r_ankle_roll_joint",
]
assert len(MOTOR_ID_TO_JOINT_NAME) == 22


def _install_isaaclab_stubs() -> None:
    """Stub the isaaclab modules g0.py imports so we can read G0_CFG / pose
    without an IsaacLab install."""
    class _Bag:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

        def replace(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
            return self

    def _bag_factory(*_a, **kw):
        return _Bag(**kw)

    def _passthrough(obj):
        return obj

    sim_mod = types.ModuleType("isaaclab.sim")
    for name in ("UsdFileCfg", "RigidBodyPropertiesCfg",
                 "ArticulationRootPropertiesCfg", "SimulationCfg",
                 "GroundPlaneCfg", "DomeLightCfg"):
        setattr(sim_mod, name, _bag_factory)
    isaaclab_mod = types.ModuleType("isaaclab")
    isaaclab_mod.sim = sim_mod

    act_mod = types.ModuleType("isaaclab.actuators")
    act_mod.ImplicitActuatorCfg = _bag_factory
    isaaclab_mod.actuators = act_mod

    assets_mod = types.ModuleType("isaaclab.assets")

    class _ArticulationCfg(_Bag):
        InitialStateCfg = staticmethod(_bag_factory)

    assets_mod.ArticulationCfg = _ArticulationCfg
    isaaclab_mod.assets = assets_mod

    utils_mod = types.ModuleType("isaaclab.utils")
    utils_mod.configclass = _passthrough
    isaaclab_mod.utils = utils_mod

    for name, mod in {
        "isaaclab": isaaclab_mod,
        "isaaclab.sim": sim_mod,
        "isaaclab.actuators": act_mod,
        "isaaclab.assets": assets_mod,
        "isaaclab.utils": utils_mod,
    }.items():
        sys.modules.setdefault(name, mod)


def load_g0_module(g0_path: str):
    """Import g0.py without going through g0_robot_lab/__init__.py."""
    _install_isaaclab_stubs()
    g0_path = Path(g0_path).resolve()
    actuators_path = g0_path.parent / "g0_actuators.py"

    spec_a = importlib.util.spec_from_file_location(
        "g0_actuators_inline_pt", actuators_path)
    mod_a = importlib.util.module_from_spec(spec_a)
    sys.modules["g0_actuators_inline_pt"] = mod_a
    spec_a.loader.exec_module(mod_a)

    pkg = types.ModuleType("g0_robot_lab_inline_pt_pkg")
    pkg.__path__ = []
    pkg.g0_actuators = mod_a
    sys.modules["g0_robot_lab_inline_pt_pkg"] = pkg
    sys.modules["g0_robot_lab_inline_pt_pkg.g0_actuators"] = mod_a

    spec_g = importlib.util.spec_from_file_location(
        "g0_robot_lab_inline_pt_pkg.g0", g0_path)
    mod_g = importlib.util.module_from_spec(spec_g)
    sys.modules["g0_robot_lab_inline_pt_pkg.g0"] = mod_g
    spec_g.loader.exec_module(mod_g)
    return mod_g


@dataclass
class JointOrdering:
    """Pre-computed index tables between wire, sdk, and urdf orderings."""

    wire_names: List[str]
    sdk_names: List[str]
    urdf_names: List[str]
    default_pos_by_name: Dict[str, float]   # radians

    # Index permutations (length 22 each). Use as np.take(arr, perm).
    wire_to_sdk: np.ndarray
    sdk_to_wire: np.ndarray
    wire_to_urdf: np.ndarray
    urdf_to_wire: np.ndarray
    urdf_to_sdk: np.ndarray
    sdk_to_urdf: np.ndarray

    # Default pose, one per ordering (radians). Each is 22-D float32.
    default_pos_wire: np.ndarray
    default_pos_sdk: np.ndarray
    default_pos_urdf: np.ndarray

    # Per-wire-slot PD gains in DDS wire units (matches the firmware /
    # codegen_pd_gains.py convention):
    #   kp_wire : N*m / deg            = g0.py stiffness  (N*m/rad)    * (pi/180)
    #   kd_wire : N*m / (deg/s)        = g0.py damping    (N*m/(rad/s)) * (pi/180)
    kp_wire: np.ndarray
    kd_wire: np.ndarray


def _resolve_pd_gains_per_wire_slot(g0_cfg) -> tuple:
    """Walk G0_CFG.actuators, resolve regex-keyed stiffness/damping per joint,
    return (kp_rad_by_wire, kd_rad_by_wire) in g0.py's native rad units.
    Same algorithm as codegen_pd_gains.py.
    """
    kp_rad: List[Optional[float]] = [None] * 22
    kd_rad: List[Optional[float]] = [None] * 22
    source_group: List[Optional[str]] = [None] * 22

    def joint_in_group(joint_name: str, exprs) -> bool:
        for expr in exprs:
            if re.fullmatch(expr, joint_name) is not None:
                return True
        return False

    def resolve(joint_name: str, spec) -> float:
        if isinstance(spec, (int, float)):
            return float(spec)
        if isinstance(spec, dict):
            for pattern, value in spec.items():
                if re.fullmatch(pattern, joint_name) is not None:
                    return float(value)
            raise KeyError(
                f"no stiffness/damping pattern matched '{joint_name}'")
        raise TypeError(f"unsupported gain spec type: {type(spec)!r}")

    for group_name, act in g0_cfg.actuators.items():
        for mid, jname in enumerate(MOTOR_ID_TO_JOINT_NAME, start=1):
            if not joint_in_group(jname, act.joint_names_expr):
                continue
            if kp_rad[mid - 1] is not None:
                raise RuntimeError(
                    f"joint '{jname}' matched two actuator groups: "
                    f"{source_group[mid - 1]!r} and {group_name!r}")
            kp_rad[mid - 1] = resolve(jname, act.stiffness)
            kd_rad[mid - 1] = resolve(jname, act.damping)
            source_group[mid - 1] = group_name

    missing = [MOTOR_ID_TO_JOINT_NAME[i] for i, v in enumerate(kp_rad) if v is None]
    if missing:
        raise RuntimeError(f"no actuator group covers: {missing}")
    return (np.asarray(kp_rad, dtype=np.float32),
            np.asarray(kd_rad, dtype=np.float32))


def build_ordering(g0_module_path: str,
                   urdf_order_strategy: str = "alphabetical",
                   urdf_order_explicit: Optional[Sequence[str]] = None,
                   ) -> JointOrdering:
    g0 = load_g0_module(g0_module_path)
    sdk_names: List[str] = list(g0.G0_JOINT_SDK_NAMES)
    default_pos: Dict[str, float] = dict(g0.G0_DEFAULT_JOINT_POS)
    kp_wire_rad, kd_wire_rad = _resolve_pd_gains_per_wire_slot(g0.G0_CFG)
    # Convert rad -> deg for wire transmission (firmware PD-gain convention).
    RAD_PER_DEG = math.pi / 180.0
    kp_wire = (kp_wire_rad * RAD_PER_DEG).astype(np.float32)
    kd_wire = (kd_wire_rad * RAD_PER_DEG).astype(np.float32)

    # Sanity: wire and sdk lists must reference the same 22-name set.
    if set(sdk_names) != set(MOTOR_ID_TO_JOINT_NAME):
        missing = set(MOTOR_ID_TO_JOINT_NAME) - set(sdk_names)
        extra = set(sdk_names) - set(MOTOR_ID_TO_JOINT_NAME)
        raise RuntimeError(
            f"wire/sdk joint name sets diverge — missing={missing}, extra={extra}")
    for n in MOTOR_ID_TO_JOINT_NAME:
        if n not in default_pos:
            raise RuntimeError(f"G0_DEFAULT_JOINT_POS missing '{n}'")

    if urdf_order_strategy == "alphabetical":
        urdf_names = sorted(MOTOR_ID_TO_JOINT_NAME)
    elif urdf_order_strategy == "wire":
        urdf_names = list(MOTOR_ID_TO_JOINT_NAME)
    elif urdf_order_strategy == "sdk":
        urdf_names = list(sdk_names)
    elif urdf_order_strategy == "explicit":
        if urdf_order_explicit is None or len(urdf_order_explicit) != 22:
            raise ValueError("urdf_order_strategy=explicit requires 22 names")
        if set(urdf_order_explicit) != set(MOTOR_ID_TO_JOINT_NAME):
            raise ValueError("urdf_order_explicit names don't match wire set")
        urdf_names = list(urdf_order_explicit)
    else:
        raise ValueError(f"unknown urdf_order_strategy: {urdf_order_strategy}")

    def perm(src: List[str], dst: List[str]) -> np.ndarray:
        idx = {n: i for i, n in enumerate(src)}
        return np.asarray([idx[n] for n in dst], dtype=np.int64)

    wire = MOTOR_ID_TO_JOINT_NAME
    wire_to_sdk  = perm(wire, sdk_names)
    sdk_to_wire  = perm(sdk_names, wire)
    wire_to_urdf = perm(wire, urdf_names)
    urdf_to_wire = perm(urdf_names, wire)
    urdf_to_sdk  = perm(urdf_names, sdk_names)
    sdk_to_urdf  = perm(sdk_names, urdf_names)

    def pose(names: List[str]) -> np.ndarray:
        return np.asarray([default_pos[n] for n in names], dtype=np.float32)

    return JointOrdering(
        wire_names=list(wire),
        sdk_names=sdk_names,
        urdf_names=urdf_names,
        default_pos_by_name=default_pos,
        wire_to_sdk=wire_to_sdk,
        sdk_to_wire=sdk_to_wire,
        wire_to_urdf=wire_to_urdf,
        urdf_to_wire=urdf_to_wire,
        urdf_to_sdk=urdf_to_sdk,
        sdk_to_urdf=sdk_to_urdf,
        default_pos_wire=pose(wire),
        default_pos_sdk=pose(sdk_names),
        default_pos_urdf=pose(urdf_names),
        kp_wire=kp_wire,
        kd_wire=kd_wire,
    )
