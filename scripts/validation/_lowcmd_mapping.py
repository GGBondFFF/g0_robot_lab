from __future__ import annotations

import ast
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ._safety_filter import SafetyFilter, SafetyFilterError, SafetyLimits, SafetyState


@dataclass(frozen=True)
class FakeMotorCommand:
    motor_id: int
    joint_name: str
    q: float
    dq: float
    kp: float
    kd: float
    tau_ff: float
    effort_limit: float
    source_action_index: int
    source_raw_action: float
    source_clipped_action: float

    @property
    def target_position(self) -> float:
        return self.q

    @property
    def torque_ff(self) -> float:
        return self.tau_ff

    @property
    def source_action_value(self) -> float:
        return self.source_raw_action


@dataclass(frozen=True)
class FakeLowCmd:
    dry_run: bool
    control_dt: float
    motors: tuple[FakeMotorCommand, ...]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _g0_source() -> Path:
    return _repo_root() / "source" / "g0_robot_lab" / "g0_robot_lab" / "assets" / "robots" / "g0" / "g0.py"


def _g0_actuators_source() -> Path:
    return _repo_root() / "source" / "g0_robot_lab" / "g0_robot_lab" / "assets" / "robots" / "g0" / "g0_actuators.py"


def _g0_urdf_source() -> Path:
    return _repo_root() / "source" / "g0_robot_lab" / "g0_robot_lab" / "assets" / "robots" / "g0" / "urdf" / "g0.urdf"


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


def _load_g0_joint_contract() -> tuple[list[str], dict[str, float], set[str]]:
    module = _load_module_ast(_g0_source())
    joint_order = list(_literal_assignment(module, "G0_JOINT_SDK_NAMES"))
    default_joint_pos = dict(_literal_assignment(module, "G0_DEFAULT_JOINT_POS"))
    right_angle = set(_literal_assignment(module, "G0_RIGHT_ANGLE_SERVO_JOINT_NAMES"))
    return joint_order, default_joint_pos, right_angle


def _load_g0_actuator_contract() -> tuple[float, float, float, float]:
    module = _load_module_ast(_g0_actuators_source())
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


def _load_urdf_position_limits() -> tuple[dict[str, float], dict[str, float]]:
    root = ET.parse(_g0_urdf_source()).getroot()
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


def build_default_safety_limits(
    joint_order: Sequence[str],
    *,
    max_target_delta: float = 0.25,
    max_action_delta: float = 1.0,
    max_command_age_s: float = 0.1,
) -> SafetyLimits:
    repo_joint_order, _default_joint_pos, right_angle = _load_g0_joint_contract()
    expected = set(repo_joint_order)
    if set(joint_order) - expected:
        unknown = sorted(set(joint_order) - expected)
        raise ValueError(f"Unknown joints in joint_order: {unknown}")

    standard_effort, standard_velocity, right_angle_effort, right_angle_velocity = _load_g0_actuator_contract()
    pos_lower, pos_upper = _load_urdf_position_limits()
    vel_limit: dict[str, float] = {}
    effort_limit: dict[str, float] = {}
    for joint_name in joint_order:
        if joint_name in right_angle:
            vel_limit[joint_name] = right_angle_velocity
            effort_limit[joint_name] = right_angle_effort
        else:
            vel_limit[joint_name] = standard_velocity
            effort_limit[joint_name] = standard_effort
    return SafetyLimits(
        pos_lower={joint_name: float(pos_lower[joint_name]) for joint_name in joint_order},
        pos_upper={joint_name: float(pos_upper[joint_name]) for joint_name in joint_order},
        vel_limit=vel_limit,
        effort_limit=effort_limit,
        max_target_delta=float(max_target_delta),
        max_action_delta=float(max_action_delta),
        max_command_age_s=float(max_command_age_s),
    )


def _ensure_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise SafetyFilterError(f"{name} must be finite, got {value!r}")
    return value


def map_policy_to_lowcmd_dry_run(
    raw_action: Sequence[float],
    *,
    joint_order: Sequence[str],
    default_joint_pos: Mapping[str, float],
    action_scale: float = 0.12,
    safety: SafetyFilter | None = None,
    limits: SafetyLimits | None = None,
    state: SafetyState | None = None,
    kp_by_joint: Mapping[str, float] | None = None,
    kd_by_joint: Mapping[str, float] | None = None,
    control_dt: float = 0.02,
    now: float | None = None,
    emergency_stop: bool = False,
    hardware_enabled: bool = False,
) -> FakeLowCmd:
    if hardware_enabled:
        raise SafetyFilterError("hardware_enabled=True is forbidden for dry-run mapping.")
    if control_dt <= 0.0:
        raise SafetyFilterError(f"control_dt must be positive, got {control_dt!r}")

    joint_names = tuple(str(name) for name in joint_order)
    limits = limits or build_default_safety_limits(joint_names)
    safety = safety or SafetyFilter()
    state = state or SafetyState()
    kp_by_joint = kp_by_joint or {}
    kd_by_joint = kd_by_joint or {}
    now = float(now if now is not None else (state.last_obs_time if state.last_obs_time is not None else 0.0))

    filtered = safety.apply(
        raw_action=raw_action,
        joint_order=joint_names,
        default_joint_pos=default_joint_pos,
        action_scale=float(action_scale),
        limits=limits,
        state=state,
        now=now,
        emergency_stop=emergency_stop,
        hardware_enabled=False,
    )

    motors: list[FakeMotorCommand] = []
    for index, joint_name in enumerate(joint_names):
        q = _ensure_finite(f"{joint_name}.q", filtered.target_by_joint[joint_name])
        dq = _ensure_finite(f"{joint_name}.dq", filtered.dq_by_joint[joint_name])
        kp = _ensure_finite(f"{joint_name}.kp", kp_by_joint.get(joint_name, 0.0))
        kd = _ensure_finite(f"{joint_name}.kd", kd_by_joint.get(joint_name, 0.0))
        effort_limit = abs(_ensure_finite(f"{joint_name}.effort_limit", limits.effort_limit[joint_name]))
        tau_ff = _ensure_finite(
            f"{joint_name}.tau_ff",
            max(-effort_limit, min(effort_limit, filtered.tau_ff_by_joint[joint_name])),
        )
        source_raw_action = _ensure_finite(
            f"{joint_name}.source_raw_action", filtered.raw_action_by_joint[joint_name]
        )
        source_clipped_action = _ensure_finite(
            f"{joint_name}.source_clipped_action", filtered.clipped_action_by_joint[joint_name]
        )
        motors.append(
            FakeMotorCommand(
                motor_id=index,
                joint_name=joint_name,
                q=q,
                dq=dq,
                kp=kp,
                kd=kd,
                tau_ff=tau_ff,
                effort_limit=effort_limit,
                source_action_index=index,
                source_raw_action=source_raw_action,
                source_clipped_action=source_clipped_action,
            )
        )

    if len(motors) != len(joint_names):
        raise SafetyFilterError(f"Expected {len(joint_names)} motors, got {len(motors)}.")

    return FakeLowCmd(dry_run=True, control_dt=float(control_dt), motors=tuple(motors))
