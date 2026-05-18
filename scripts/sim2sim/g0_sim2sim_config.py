"""Shared G0 sim2sim interface constants.

This module is intentionally lightweight: it can be imported from a normal
Python process for tests and MuJoCo tools. When Isaac Lab is importable, the G0
constants are imported from the project package. When Isaac Lab is not available,
the same constants are read from the source files with Python's AST module so
that ordinary tests do not need to launch Isaac Sim.
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "source" / "g0_robot_lab"
G0_SOURCE = SOURCE_ROOT / "g0_robot_lab" / "assets" / "robots" / "g0" / "g0.py"
G0_ACTUATORS_SOURCE = SOURCE_ROOT / "g0_robot_lab" / "assets" / "robots" / "g0" / "g0_actuators.py"

if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

ACTION_SCALE = 0.12
ISAAC_SIM_DT = 0.005
ISAAC_DECIMATION = 4
CONTROL_DT = ISAAC_SIM_DT * ISAAC_DECIMATION
GAIT_PERIOD = 0.8
POLICY_HISTORY_LENGTH = 5
POLICY_OBS_TERMS = [
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
    "gait_phase",
]
POLICY_OBS_TERM_DIMS = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "velocity_commands": 3,
    "joint_pos_rel": 22,
    "joint_vel_rel": 22,
    "last_action": 22,
    "gait_phase": 2,
}


def get_observation_term_slices() -> dict[str, slice]:
    """Return per-frame observation slices in Isaac policy term order."""

    start = 0
    slices: dict[str, slice] = {}
    for term in POLICY_OBS_TERMS:
        dim = POLICY_OBS_TERM_DIMS[term]
        slices[term] = slice(start, start + dim)
        start += dim
    return slices


def flatten_history_term_major(frames: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """Flatten observation history exactly like Isaac Lab grouped history.

    Isaac Lab keeps a history buffer for each observation term, flattens that
    term's history from oldest to newest, and then concatenates terms in config
    order. It is term-major, not full-frame-major.
    """

    frames = np.asarray(frames, dtype=np.float64)
    expected = (POLICY_HISTORY_LENGTH, get_single_frame_observation_dim())
    if frames.shape != expected:
        raise ValueError(f"Expected history frames shape {expected}, got {frames.shape}")
    return np.concatenate(
        [
            np.concatenate([frame[term_slice] for frame in frames])
            for term_slice in get_observation_term_slices().values()
        ]
    )


def split_policy_observation(obs: np.ndarray) -> dict[str, np.ndarray]:
    """Split a single flattened policy observation into term histories.

    The returned arrays have shape ``(history_length, term_dim)`` for history
    observations and ``(1, term_dim)`` for a single non-history frame.
    """

    obs = np.asarray(obs, dtype=np.float64).reshape(-1)
    frame_width = get_single_frame_observation_dim()
    if obs.shape == (frame_width,):
        return {
            term: obs[term_slice].reshape(1, -1)
            for term, term_slice in get_observation_term_slices().items()
        }
    if obs.shape != (get_policy_observation_dim(),):
        raise ValueError(
            f"Expected observation width {frame_width} or {get_policy_observation_dim()}, got {obs.shape}"
        )

    result: dict[str, np.ndarray] = {}
    offset = 0
    for term in POLICY_OBS_TERMS:
        dim = POLICY_OBS_TERM_DIMS[term]
        width = dim * POLICY_HISTORY_LENGTH
        result[term] = obs[offset : offset + width].reshape(POLICY_HISTORY_LENGTH, dim)
        offset += width
    return result


@dataclass(frozen=True)
class IsaacActuatorSpec:
    """Per-joint actuator parameters resolved from the Isaac G0 configuration."""

    joint_name: str
    servo_type: str
    stiffness: float
    damping: float
    effort_limit_sim: float
    velocity_limit_sim: float
    armature: float


def _literal_from_python_file(path: Path, name: str) -> Any:
    """Return a top-level literal assignment from a Python file."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise RuntimeError(f"Could not find literal assignment {name!r} in {path}")


def _load_actuator_module() -> Any:
    spec = importlib.util.spec_from_file_location("_g0_sim2sim_actuators", G0_ACTUATORS_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load actuator module from {G0_ACTUATORS_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_g0_name(name: str) -> Any:
    if name == "G0_JOINT_SDK_NAMES":
        return G0_JOINT_SDK_NAMES
    if name == "G0_RIGHT_ANGLE_SERVO_JOINT_NAMES":
        return G0_RIGHT_ANGLE_SERVO_JOINT_NAMES
    if name == "G0_STANDARD_SERVO_JOINT_NAMES":
        return G0_STANDARD_SERVO_JOINT_NAMES
    raise KeyError(name)


def _eval_actuator_keyword(node: ast.AST) -> Any:
    """Evaluate the small subset of AST nodes used in the G0 actuator config."""

    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Dict):
        return {
            ast.literal_eval(key): float(_eval_actuator_keyword(value))
            for key, value in zip(node.keys, node.values, strict=True)
        }
    if isinstance(node, ast.Name):
        return _resolve_g0_name(node.id)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "g0_actuators":
        return float(getattr(_g0_actuators, node.attr))
    return ast.literal_eval(node)


def _find_actuator_cfg_nodes() -> dict[str, ast.Call]:
    tree = ast.parse(G0_SOURCE.read_text(encoding="utf-8"), filename=str(G0_SOURCE))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "G0_CFG" for target in node.targets):
            continue
        for keyword in node.value.keywords:
            if keyword.arg == "actuators" and isinstance(keyword.value, ast.Dict):
                groups: dict[str, ast.Call] = {}
                for key, value in zip(keyword.value.keys, keyword.value.values, strict=True):
                    if isinstance(key, ast.Constant) and isinstance(value, ast.Call):
                        groups[str(key.value)] = value
                return groups
    raise RuntimeError(f"Could not find G0_CFG actuator definitions in {G0_SOURCE}")


def _extract_actuator_group(group_name: str, node: ast.Call) -> dict[str, Any]:
    values = {"group_name": group_name}
    for keyword in node.keywords:
        if keyword.arg in {
            "joint_names_expr",
            "effort_limit_sim",
            "velocity_limit_sim",
            "stiffness",
            "damping",
            "armature",
        }:
            values[keyword.arg] = _eval_actuator_keyword(keyword.value)
    missing = {
        "joint_names_expr",
        "effort_limit_sim",
        "velocity_limit_sim",
        "stiffness",
        "damping",
        "armature",
    } - set(values)
    if missing:
        raise RuntimeError(f"Actuator group {group_name!r} is missing keys: {sorted(missing)}")
    return values


def _match_pattern_map(joint_name: str, pattern_map: dict[str, float], field_name: str) -> float:
    for pattern, value in pattern_map.items():
        if re.fullmatch(pattern, joint_name):
            return float(value)
    raise RuntimeError(f"Could not resolve {field_name} for joint {joint_name!r}")


def _load_g0_constants() -> tuple[list[str], dict[str, float]]:
    try:
        from g0_robot_lab.assets.robots.g0.g0 import G0_DEFAULT_JOINT_POS, G0_JOINT_SDK_NAMES

        return list(G0_JOINT_SDK_NAMES), dict(G0_DEFAULT_JOINT_POS)
    except Exception:
        joint_names = _literal_from_python_file(G0_SOURCE, "G0_JOINT_SDK_NAMES")
        default_joint_pos = _literal_from_python_file(G0_SOURCE, "G0_DEFAULT_JOINT_POS")
        return list(joint_names), dict(default_joint_pos)


G0_JOINT_SDK_NAMES, G0_DEFAULT_JOINT_POS = _load_g0_constants()
G0_RIGHT_ANGLE_SERVO_JOINT_NAMES = list(_literal_from_python_file(G0_SOURCE, "G0_RIGHT_ANGLE_SERVO_JOINT_NAMES"))
G0_STANDARD_SERVO_JOINT_NAMES = [
    name for name in G0_JOINT_SDK_NAMES if name not in G0_RIGHT_ANGLE_SERVO_JOINT_NAMES
]

try:
    from g0_robot_lab.assets.robots.g0 import g0_actuators as _g0_actuators
except Exception:
    _g0_actuators = _load_actuator_module()

STANDARD_SERVO_RATED_TORQUE = float(_g0_actuators.STANDARD_SERVO_RATED_TORQUE)
STANDARD_SERVO_PEAK_TORQUE = float(_g0_actuators.STANDARD_SERVO_PEAK_TORQUE)
STANDARD_SERVO_MAX_VELOCITY = float(_g0_actuators.STANDARD_SERVO_MAX_VELOCITY)
RIGHT_ANGLE_SERVO_RATED_TORQUE = float(_g0_actuators.RIGHT_ANGLE_SERVO_RATED_TORQUE)
RIGHT_ANGLE_SERVO_PEAK_TORQUE = float(_g0_actuators.RIGHT_ANGLE_SERVO_PEAK_TORQUE)
RIGHT_ANGLE_SERVO_MAX_VELOCITY = float(_g0_actuators.RIGHT_ANGLE_SERVO_MAX_VELOCITY)
STANDARD_SERVO_ARMATURE = float(_g0_actuators.STANDARD_SERVO_ARMATURE)
RIGHT_ANGLE_SERVO_ARMATURE = float(_g0_actuators.RIGHT_ANGLE_SERVO_ARMATURE)

_ACTUATOR_GROUPS = {
    name: _extract_actuator_group(name, node) for name, node in _find_actuator_cfg_nodes().items()
}


def get_joint_names() -> list[str]:
    """Return the G0 policy/action joint names in deployment order."""

    return list(G0_JOINT_SDK_NAMES)


def get_standard_servo_joint_names() -> list[str]:
    """Return joints assigned to the standard servo group in ``G0_CFG``."""

    return list(G0_STANDARD_SERVO_JOINT_NAMES)


def get_right_angle_servo_joint_names() -> list[str]:
    """Return joints assigned to the right-angle servo group in ``G0_CFG``."""

    return list(G0_RIGHT_ANGLE_SERVO_JOINT_NAMES)


def get_isaac_actuator_specs() -> dict[str, IsaacActuatorSpec]:
    """Return per-joint actuator specs resolved from the source ``G0_CFG``.

    The function parses the Python source instead of importing ``G0_CFG``
    directly. Importing ``G0_CFG`` requires Isaac Sim/pxr, while these sim2sim
    tools must also run in ordinary Python for lightweight tests and reports.
    """

    group_to_servo = {
        "standard_servos": "standard",
        "right_angle_servos": "right_angle",
    }
    specs: dict[str, IsaacActuatorSpec] = {}
    for group_name, group in _ACTUATOR_GROUPS.items():
        servo_type = group_to_servo.get(group_name, group_name)
        for joint_name in group["joint_names_expr"]:
            specs[joint_name] = IsaacActuatorSpec(
                joint_name=joint_name,
                servo_type=servo_type,
                stiffness=_match_pattern_map(joint_name, group["stiffness"], "stiffness"),
                damping=_match_pattern_map(joint_name, group["damping"], "damping"),
                effort_limit_sim=float(group["effort_limit_sim"]),
                velocity_limit_sim=float(group["velocity_limit_sim"]),
                armature=float(group["armature"]),
            )
    missing = [name for name in G0_JOINT_SDK_NAMES if name not in specs]
    if missing:
        raise RuntimeError(f"Missing actuator specs for policy joints: {missing}")
    return {name: specs[name] for name in G0_JOINT_SDK_NAMES}


def get_default_joint_pos_array() -> np.ndarray:
    """Return the default joint position array in ``G0_JOINT_SDK_NAMES`` order."""

    return np.asarray([G0_DEFAULT_JOINT_POS[name] for name in G0_JOINT_SDK_NAMES], dtype=np.float64)


def get_action_dim() -> int:
    """Return the policy action dimension."""

    return len(G0_JOINT_SDK_NAMES)


def validate_action_shape(action: np.ndarray) -> None:
    """Validate that an action has shape ``(action_dim,)``."""

    action = np.asarray(action)
    expected = (get_action_dim(),)
    if action.shape != expected:
        raise ValueError(f"Expected action shape {expected}, got {action.shape}")


def compute_target_joint_pos(action: np.ndarray) -> np.ndarray:
    """Convert a policy action to target joint positions.

    The action is clipped to ``[-1, 1]`` and applied with the Isaac Lab action
    bridge formula:

    ``target_joint_pos = default_joint_pos + ACTION_SCALE * clipped_action``.
    """

    action = np.asarray(action, dtype=np.float64)
    validate_action_shape(action)
    clipped_action = np.clip(action, -1.0, 1.0)
    return get_default_joint_pos_array() + ACTION_SCALE * clipped_action


def get_single_frame_observation_dim() -> int:
    """Return the current per-frame policy observation width."""

    return 3 + 3 + 3 + get_action_dim() + get_action_dim() + get_action_dim() + 2


def get_policy_observation_dim() -> int:
    """Return the flattened policy observation width including history."""

    return get_single_frame_observation_dim() * POLICY_HISTORY_LENGTH
