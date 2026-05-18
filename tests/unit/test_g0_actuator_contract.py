from __future__ import annotations

import math

import pytest

from tests.conftest import G0_ACTUATORS_SOURCE
from tests.helpers.static_contract import eval_numeric_expr, load_module_ast, numeric_assignment_expr


pytestmark = pytest.mark.unit


def _numeric_values() -> dict[str, float]:
    module = load_module_ast(G0_ACTUATORS_SOURCE)
    values: dict[str, float] = {"math.pi": math.pi}
    ordered_names = [
        "STANDARD_SERVO_RATED_TORQUE",
        "STANDARD_SERVO_PEAK_TORQUE",
        "STANDARD_SERVO_MAX_RPM",
        "STANDARD_SERVO_MAX_VELOCITY",
        "RIGHT_ANGLE_SPEED_RATIO",
        "RIGHT_ANGLE_TORQUE_RATIO",
        "RIGHT_ANGLE_GEAR_EFFICIENCY",
        "RIGHT_ANGLE_SERVO_RATED_TORQUE",
        "RIGHT_ANGLE_SERVO_PEAK_TORQUE",
        "RIGHT_ANGLE_SERVO_MAX_VELOCITY",
        "STANDARD_SERVO_DAMPING",
        "STANDARD_SERVO_FRICTION",
        "STANDARD_SERVO_ARMATURE",
        "RIGHT_ANGLE_SERVO_DAMPING",
        "RIGHT_ANGLE_SERVO_FRICTION",
        "RIGHT_ANGLE_SERVO_ARMATURE",
    ]
    for name in ordered_names:
        values[name] = eval_numeric_expr(numeric_assignment_expr(module, name), values)
    return values


def test_standard_servo_speed_conversion():
    values = _numeric_values()
    assert values["STANDARD_SERVO_MAX_RPM"] == 300.0
    assert values["STANDARD_SERVO_MAX_VELOCITY"] == pytest.approx(300.0 * 2.0 * math.pi / 60.0)


def test_right_angle_servo_ratios():
    values = _numeric_values()
    assert values["RIGHT_ANGLE_SPEED_RATIO"] == pytest.approx(6.0 / 7.0)
    assert values["RIGHT_ANGLE_TORQUE_RATIO"] == pytest.approx(7.0 / 6.0)
    assert values["RIGHT_ANGLE_GEAR_EFFICIENCY"] == pytest.approx(1.0)


def test_right_angle_servo_limits_follow_ratio_contract():
    values = _numeric_values()
    assert values["RIGHT_ANGLE_SERVO_RATED_TORQUE"] == pytest.approx(
        values["STANDARD_SERVO_RATED_TORQUE"]
        * values["RIGHT_ANGLE_TORQUE_RATIO"]
        * values["RIGHT_ANGLE_GEAR_EFFICIENCY"]
    )
    assert values["RIGHT_ANGLE_SERVO_PEAK_TORQUE"] == pytest.approx(
        values["STANDARD_SERVO_PEAK_TORQUE"]
        * values["RIGHT_ANGLE_TORQUE_RATIO"]
        * values["RIGHT_ANGLE_GEAR_EFFICIENCY"]
    )
    assert values["RIGHT_ANGLE_SERVO_MAX_VELOCITY"] == pytest.approx(
        values["STANDARD_SERVO_MAX_VELOCITY"] * values["RIGHT_ANGLE_SPEED_RATIO"]
    )


def test_right_angle_armature_damping_friction_follow_ratio_contract():
    values = _numeric_values()
    torque_ratio = values["RIGHT_ANGLE_TORQUE_RATIO"]
    assert values["RIGHT_ANGLE_SERVO_DAMPING"] == pytest.approx(values["STANDARD_SERVO_DAMPING"] * torque_ratio**2)
    assert values["RIGHT_ANGLE_SERVO_FRICTION"] == pytest.approx(values["STANDARD_SERVO_FRICTION"] * torque_ratio)
    assert values["RIGHT_ANGLE_SERVO_ARMATURE"] == pytest.approx(values["STANDARD_SERVO_ARMATURE"] * torque_ratio**2)
