from __future__ import annotations

import ast
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class JointContractTables:
    joint_order: tuple[str, ...]
    default_joint_pos: dict[str, float]
    right_angle_joints: frozenset[str]
    pos_lower: dict[str, float]
    pos_upper: dict[str, float]
    vel_limit: dict[str, float]
    effort_limit: dict[str, float]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def g0_source() -> Path:
    return repo_root() / "source" / "g0_robot_lab" / "g0_robot_lab" / "assets" / "robots" / "g0" / "g0.py"


def g0_actuators_source() -> Path:
    return repo_root() / "source" / "g0_robot_lab" / "g0_robot_lab" / "assets" / "robots" / "g0" / "g0_actuators.py"


def g0_urdf_source() -> Path:
    return repo_root() / "source" / "g0_robot_lab" / "g0_robot_lab" / "assets" / "robots" / "g0" / "urdf" / "g0.urdf"


def _load_module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _literal_assignment(module: ast.Module, name: str):
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise ValueError(f"Assignment not found: {name}")


def _numeric_assignment_expr(module: ast.Module, name: str) -> ast.expr:
    for node in module.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
    raise ValueError(f"Numeric assignment not found: {name}")


def _eval_numeric_expr(expr: ast.expr, names: dict[str, float]) -> float:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, (int, float)):
        return float(expr.value)
    if isinstance(expr, ast.Name):
        return float(names[expr.id])
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
        if expr.value.id == "math" and expr.attr == "pi":
            return math.pi
    if isinstance(expr, ast.BinOp):
        left = _eval_numeric_expr(expr.left, names)
        right = _eval_numeric_expr(expr.right, names)
        if isinstance(expr.op, ast.Add):
            return left + right
        if isinstance(expr.op, ast.Sub):
            return left - right
        if isinstance(expr.op, ast.Mult):
            return left * right
        if isinstance(expr.op, ast.Div):
            return left / right
        if isinstance(expr.op, ast.Pow):
            return left**right
    raise ValueError(f"Unsupported numeric expression: {ast.dump(expr)}")


def load_g0_joint_contract() -> tuple[list[str], dict[str, float], set[str]]:
    module = _load_module_ast(g0_source())
    joint_order = list(_literal_assignment(module, "G0_JOINT_SDK_NAMES"))
    default_joint_pos = dict(_literal_assignment(module, "G0_DEFAULT_JOINT_POS"))
    right_angle = set(_literal_assignment(module, "G0_RIGHT_ANGLE_SERVO_JOINT_NAMES"))
    return joint_order, default_joint_pos, right_angle


def load_g0_actuator_contract() -> tuple[float, float, float, float]:
    module = _load_module_ast(g0_actuators_source())
    names: dict[str, float] = {}
    for name in (
        "STANDARD_SERVO_MAX_RPM",
        "STANDARD_SERVO_RATED_TORQUE",
        "RIGHT_ANGLE_SPEED_RATIO",
        "RIGHT_ANGLE_TORQUE_RATIO",
        "RIGHT_ANGLE_GEAR_EFFICIENCY",
        "STANDARD_SERVO_MAX_VELOCITY",
        "RIGHT_ANGLE_SERVO_RATED_TORQUE",
        "RIGHT_ANGLE_SERVO_MAX_VELOCITY",
    ):
        names[name] = _eval_numeric_expr(_numeric_assignment_expr(module, name), names)
    return (
        names["STANDARD_SERVO_RATED_TORQUE"],
        names["STANDARD_SERVO_MAX_VELOCITY"],
        names["RIGHT_ANGLE_SERVO_RATED_TORQUE"],
        names["RIGHT_ANGLE_SERVO_MAX_VELOCITY"],
    )


def load_urdf_position_limits() -> tuple[dict[str, float], dict[str, float]]:
    root = ET.parse(g0_urdf_source()).getroot()
    pos_lower: dict[str, float] = {}
    pos_upper: dict[str, float] = {}
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed":
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        name = str(joint.get("name"))
        pos_lower[name] = float(limit.attrib["lower"])
        pos_upper[name] = float(limit.attrib["upper"])
    return pos_lower, pos_upper


def load_joint_contract_tables(joint_order: Sequence[str] | None = None) -> JointContractTables:
    repo_joint_order, default_joint_pos, right_angle = load_g0_joint_contract()
    joint_names = tuple(repo_joint_order if joint_order is None else [str(name) for name in joint_order])
    pos_lower_all, pos_upper_all = load_urdf_position_limits()
    standard_effort, standard_velocity, right_angle_effort, right_angle_velocity = load_g0_actuator_contract()
    vel_limit: dict[str, float] = {}
    effort_limit: dict[str, float] = {}
    for joint_name in joint_names:
        if joint_name in right_angle:
            vel_limit[joint_name] = right_angle_velocity
            effort_limit[joint_name] = right_angle_effort
        else:
            vel_limit[joint_name] = standard_velocity
            effort_limit[joint_name] = standard_effort
    return JointContractTables(
        joint_order=joint_names,
        default_joint_pos={joint_name: float(default_joint_pos[joint_name]) for joint_name in joint_names},
        right_angle_joints=frozenset(right_angle),
        pos_lower={joint_name: float(pos_lower_all[joint_name]) for joint_name in joint_names},
        pos_upper={joint_name: float(pos_upper_all[joint_name]) for joint_name in joint_names},
        vel_limit=vel_limit,
        effort_limit=effort_limit,
    )
