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
import sys
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


def _load_g0_constants() -> tuple[list[str], dict[str, float]]:
    try:
        from g0_robot_lab.assets.robots.g0.g0 import G0_DEFAULT_JOINT_POS, G0_JOINT_SDK_NAMES

        return list(G0_JOINT_SDK_NAMES), dict(G0_DEFAULT_JOINT_POS)
    except Exception:
        joint_names = _literal_from_python_file(G0_SOURCE, "G0_JOINT_SDK_NAMES")
        default_joint_pos = _literal_from_python_file(G0_SOURCE, "G0_DEFAULT_JOINT_POS")
        return list(joint_names), dict(default_joint_pos)


G0_JOINT_SDK_NAMES, G0_DEFAULT_JOINT_POS = _load_g0_constants()

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


def get_joint_names() -> list[str]:
    """Return the G0 policy/action joint names in deployment order."""

    return list(G0_JOINT_SDK_NAMES)


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

