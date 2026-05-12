from __future__ import annotations

import math

# -----------------------------------------------------------------------------
# Standard servo hardware constants
# -----------------------------------------------------------------------------

STANDARD_SERVO_RATED_TORQUE = 0.5
"""Rated output torque of the standard servo, in N*m."""

STANDARD_SERVO_PEAK_TORQUE = 1.0
"""Peak output torque of the standard servo, in N*m."""

STANDARD_SERVO_MAX_RPM = 300.0
"""Maximum output speed of the standard servo, in r/min."""

STANDARD_SERVO_MAX_VELOCITY = STANDARD_SERVO_MAX_RPM * 2.0 * math.pi / 60.0
"""Maximum output speed of the standard servo, in rad/s. 300 r/min = 31.416 rad/s."""


# -----------------------------------------------------------------------------
# Right-angle servo hardware constants
# -----------------------------------------------------------------------------
# The right-angle servo is the standard servo plus an external reduction gear.
#
# Current confirmed assumption:
#   output_speed = input_speed * 6 / 7
#   output_torque = input_torque * 7 / 6
#
# If the manufacturer defines 6/7 in the opposite convention, swap these ratios.

RIGHT_ANGLE_SPEED_RATIO = 6.0 / 7.0
"""Output speed ratio of the right-angle servo relative to the standard servo."""

RIGHT_ANGLE_TORQUE_RATIO = 7.0 / 6.0
"""Output torque ratio of the right-angle servo relative to the standard servo."""

RIGHT_ANGLE_GEAR_EFFICIENCY = 1.0
"""Gear efficiency. Keep 1.0 before measurement. Replace later if measured."""

RIGHT_ANGLE_SERVO_RATED_TORQUE = (
    STANDARD_SERVO_RATED_TORQUE * RIGHT_ANGLE_TORQUE_RATIO * RIGHT_ANGLE_GEAR_EFFICIENCY
)
"""Rated output torque of the right-angle servo, in N*m."""

RIGHT_ANGLE_SERVO_PEAK_TORQUE = (
    STANDARD_SERVO_PEAK_TORQUE * RIGHT_ANGLE_TORQUE_RATIO * RIGHT_ANGLE_GEAR_EFFICIENCY
)
"""Peak output torque of the right-angle servo, in N*m."""

RIGHT_ANGLE_SERVO_MAX_VELOCITY = STANDARD_SERVO_MAX_VELOCITY * RIGHT_ANGLE_SPEED_RATIO
"""Maximum output speed of the right-angle servo, in rad/s."""


# -----------------------------------------------------------------------------
# Armature placeholders, damping and friction
# -----------------------------------------------------------------------------
# These are not final identified values.
# They should be replaced by motor rotor inertia and gear-side reflected inertia.

STANDARD_SERVO_DAMPING = 0.1
STANDARD_SERVO_FRICTION = 0.05
STANDARD_SERVO_ARMATURE = 1.0e-3
"""Initial placeholder armature for standard servos."""

RIGHT_ANGLE_SERVO_DAMPING = STANDARD_SERVO_DAMPING * RIGHT_ANGLE_TORQUE_RATIO**2
RIGHT_ANGLE_SERVO_FRICTION = STANDARD_SERVO_FRICTION * RIGHT_ANGLE_TORQUE_RATIO
RIGHT_ANGLE_SERVO_ARMATURE = STANDARD_SERVO_ARMATURE * RIGHT_ANGLE_TORQUE_RATIO**2
"""Initial placeholder armature for right-angle servos."""