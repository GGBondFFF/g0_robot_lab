# G0 Actuator Parameters

This document summarizes the current actuator parameter design for the G0 Isaac Lab baseline.

Current files:

```text
source/g0_robot_lab/g0_robot_lab/assets/robots/g0/g0.py
source/g0_robot_lab/g0_robot_lab/assets/robots/g0/g0_actuators.py
```

These values are a debug baseline, not final hardware identification results.

## Standard Servo Constants

Current standard servo constants:

```text
rated torque = 0.5 N*m
peak torque = 1.0 N*m
max speed = 300 r/min
max speed = 31.416 rad/s
```

The speed conversion is:

```text
300 * 2*pi / 60 = 31.416 rad/s
```

## Right-Angle Servo Assumption

The current right-angle servo model is understood as:

```text
output_speed = standard_speed * 6 / 7
output_torque = standard_torque * 7 / 6
```

With the current standard servo constants, this gives approximately:

```text
right-angle rated torque = 0.5833 N*m
right-angle peak torque = 1.1667 N*m
right-angle max speed = 26.928 rad/s
```

This ratio convention should be replaced if the manufacturer defines the gearbox ratio differently or if bench measurements show different output behavior.

## Current Servo Groups

Right-angle servo joints:

```text
l_elbow_pitch_joint
r_elbow_pitch_joint
l_knee_pitch_joint
r_knee_pitch_joint
l_ankle_pitch_joint
r_ankle_pitch_joint
```

Standard servo joints:

```text
l_hip_pitch_joint
l_hip_roll_joint
l_hip_yaw_joint
l_ankle_roll_joint
r_hip_pitch_joint
r_hip_roll_joint
r_hip_yaw_joint
r_ankle_roll_joint
waist_yaw_joint
waist_roll_joint
l_shoulder_pitch_joint
l_shoulder_roll_joint
l_shoulder_yaw_joint
r_shoulder_pitch_joint
r_shoulder_roll_joint
r_shoulder_yaw_joint
```

The current right-angle ankle assumption is `ankle_pitch`. If the real hardware uses right-angle servos on `ankle_roll`, the grouping must be updated in a dedicated code change.

## PD And Physical Constants

`g0_actuators.py` stores current torque, velocity, damping, friction, and armature placeholders. The armature/damping/friction values are not final identified hardware values.

`g0.py` assigns Isaac Lab implicit actuator groups and PD gains. PD gains are controller tuning parameters, while `effort_limit_sim` and `velocity_limit_sim` should continue to reflect the current hardware parameter model.

## Test Coverage

The static unit validation tier covers the actuator constants and ratio contracts:

```bash
python -m pytest tests/unit -m "unit"
```

These tests check the standard servo speed conversion, right-angle speed and torque ratios, derived torque and velocity limits, and reflected damping/friction/armature placeholders. They do not launch Isaac Sim and do not contact hardware.

## Debug-Only Effort Scale

Some debug scripts support an `--effort-scale` argument. That scale is only for diagnosis.

Do not treat debug-only effort scale as a real hardware parameter. In particular:

- do not write `--effort-scale` results back into `g0_actuators.py`
- do not permanently inflate torque limits to hide standing or contact problems
- do not use higher effort limits as proof that the hardware can produce those torques

The current zero-action debug result suggests that even when temporary effort scaling reduces saturation, the robot can still fail from geometry, contact, COM, height, or inertia issues.

The 500-step zero-action standing check is an explicit release gate. It may report a physical-readiness failure on the current baseline, and that result should be treated as deployment readiness feedback rather than a default unit, deployment dry-run, or Isaac smoke failure.
