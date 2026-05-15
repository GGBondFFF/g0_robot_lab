# G0 Zero-Action Standing Pose Sweep

## Purpose

This debug pass checks whether `G0-Velocity-v0` can hold a zero-action standing pose before PPO training. The run uses one environment, headless simulation, fixed zero velocity command, disabled reset randomization, disabled command randomization, and at least 500 simulation steps per candidate.

The current sweep intentionally uses two-decimal initial pose values. G0 is a small robot, so the standing baseline should not depend on `0.0001 rad` pose precision that would be fragile in real deployment.

## Hardware Boundary

`g0_actuators.py` contains hardware constants: rated torque, peak torque, velocity limit, and armature. This sweep does not enlarge those constants.

The current documented hardware baseline keeps `STANDARD_SERVO_RATED_TORQUE = 0.5`. The `--effort-scale` option is runtime debug only and does not write back to `g0_actuators.py`.

## Script

Main entry:

```bash
TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p scripts/debug/sweep_g0_standing_pose.py \
  --task G0-Velocity-v0 \
  --headless \
  --mode fixed \
  --steps 500 \
  --print-torque-every 10 \
  --effort-scale 1.0
```

`scripts/debug/sweep_g0_standing_pose.py` is a compatibility wrapper. The implementation lives in `scripts/debug/sweep_zero_action_standing_pose.py`.

Useful options:

```bash
--mode fixed
--root_z 0.23 --hip 0.20 --knee 0.35 --ankle 0.14
--mode sweep
--candidate-set quick
--candidate-set grid
--root_z_values 0.22,0.23,0.24
--hip_values 0.18,0.19,0.20,0.21
--knee_values 0.33,0.34,0.35,0.36
--ankle_values 0.13,0.14,0.15,0.16
--print-torque-every 10
--effort-scale 1.0
```

The script reports per candidate:

- root height, hip pitch, knee pitch, ankle pitch
- survive steps and termination reason
- final and minimum root height
- final pitch, max absolute pitch, and pitch slope
- max hip/knee/ankle torque ratio
- foot force imbalance and final left/right foot force
- initial and final left/right foot height
- per-joint torque saturation summary

It also writes full per-step per-joint torque traces to:

```text
logs/zero_action_torque_trace_<timestamp>.csv
```

## Torque Logging

Every simulation step records:

- step
- joint name
- joint position and velocity
- applied torque if available
- computed torque if available
- selected torque used for ratio calculation
- effort limit
- `torque_ratio = abs(torque) / effort_limit`
- whether `torque_ratio > 0.90`
- root height and root pitch
- termination reason once triggered

Console output is throttled by `--print-torque-every`. Each print shows the top 8 joints by torque ratio so the falling process can be inspected without flooding the terminal.

## Test Results

Fixed two-decimal candidate:

```text
root_z=0.23 hip=0.20 knee=0.35 ankle=0.14 effort_scale=1.0
trace=logs/zero_action_torque_trace_20260512_154804.csv
survive=91/500
reason=bad_orientation
final_root_z=0.1262
min_root_z=0.1262
final_pitch=70.137 deg
pitch_slope=0.54495 deg/step
max_hip_ratio=0.397
max_knee_ratio=0.597
max_ankle_ratio=1.000
```

Torque saturation summary for that fixed candidate:

```text
l_ankle_pitch_joint max_abs_torque=1.1667 max_ratio=1.000 sat_steps=26 first_sat_step=57
r_ankle_pitch_joint max_abs_torque=1.1667 max_ratio=1.000 sat_steps=27 first_sat_step=57
l_ankle_roll_joint  max_abs_torque=0.9019 max_ratio=0.902 sat_steps=2  first_sat_step=60
earliest ratio>0.90 joint: l_ankle_pitch_joint at step 57
```

Effort scale fixed tests on the same candidate:

| effort_scale | trace | survive | reason | final pitch | max ankle ratio | conclusion |
| ---: | --- | ---: | --- | ---: | ---: | --- |
| 1.0 | `logs/zero_action_torque_trace_20260512_154804.csv` | 91 | bad_orientation | 70.137 | 1.000 | ankle pitch saturates early |
| 1.2 | `logs/zero_action_torque_trace_20260512_154844.csv` | 91 | bad_orientation | 70.137 | 1.000 | saturation delayed but fall unchanged |
| 1.5 | `logs/zero_action_torque_trace_20260512_154917.csv` | 91 | bad_orientation | 70.137 | 0.959 | saturation reduced but fall unchanged |
| 2.0 | `logs/zero_action_torque_trace_20260512_154940.csv` | 91 | bad_orientation | 70.137 | 0.720 | no saturation, fall unchanged |

Quick two-decimal sweep at `effort_scale=1.0`:

| root_z | hip | knee | ankle | survive | reason | final z | final pitch | hip ratio | knee ratio | ankle ratio |
| ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.23 | 0.20 | 0.34 | 0.14 | 116 | bad_orientation | 0.120 | 71.823 | 0.410 | 0.656 | 1.000 |
| 0.22 | 0.20 | 0.35 | 0.14 | 95 | bad_orientation | n/a | 69.612 | n/a | n/a | n/a |
| 0.23 | 0.20 | 0.35 | 0.14 | 91 | bad_orientation | 0.126 | 70.137 | 0.397 | 0.597 | 1.000 |
| 0.23 | 0.19 | 0.35 | 0.14 | 79 | bad_orientation | n/a | n/a | n/a | n/a | n/a |
| 0.24 | 0.20 | 0.35 | 0.14 | 76 | base_height | n/a | -64.943 | n/a | n/a | n/a |
| 0.23 | 0.20 | 0.35 | 0.15 | 74 | bad_orientation | n/a | n/a | n/a | n/a | n/a |

Quick two-decimal sweep at `effort_scale=2.0`:

```text
trace=logs/zero_action_torque_trace_20260512_155047.csv
best root_z=0.23 hip=0.20 knee=0.34 ankle=0.14
survive=116/500
reason=bad_orientation
final_root_z=0.120
final_pitch=71.823 deg
max_hip_ratio=0.205
max_knee_ratio=0.328
max_ankle_ratio=0.751
```

The same best candidate still fails even after saturation is removed by `effort_scale=2.0`.

Disable-termination observation on the same best candidate:

```text
root_z=0.23 hip=0.20 knee=0.34 ankle=0.14 effort_scale=2.0
trace=logs/zero_action_torque_trace_20260512_164624.csv
first would-terminate step=115
first would-terminate reason=bad_orientation
final_root_z=0.0468
final_pitch=85.136 deg
max_abs_pitch=86.502 deg
```

The robot continues to collapse after the original termination point. By step 120 the root is near the ground and the body is about `83 deg` pitched with about `-90 deg` roll. This confirms the earlier failure is a real fall, not just a termination threshold artifact.

Initial geometry check:

```text
total_mass=1.352610 kg
COM=(-0.00202, 0.00005, 0.26771)
support_center=(0.00792, 0.00158)
com_dx=-0.00994
com_dy=-0.00153
l_foot_lowest_z=0.00081
r_foot_lowest_z=0.00133
left/right foot force z=0.00000, 0.00000
```

The initial COM projection is near the support center, but total mass is not close to the expected `1.9 kg`, and the feet are close to but not carrying contact force at reset.

## Current Conclusion

No two-decimal candidate has passed 500 zero-action steps yet.

Current best two-decimal candidate:

```text
root_z=0.23
hip_pitch=0.20
knee_pitch=0.34
ankle_pitch=0.14
survive=116/500
reason=bad_orientation
```

The first visible torque bottleneck at nominal effort is ankle pitch saturation. However, `effort_scale=2.0` removes long saturation and the robot still falls in essentially the same way. That means the problem is not explained by ankle/knee/hip torque limit alone.

Next checks should prioritize:

- COM and link inertial parameters
- foot collision shape and effective contact patch
- foot link/contact frame height
- root initial height and initial foot penetration or suspension
- whether the default pose places the support polygon behind the COM

Formal PPO training should remain blocked until zero-action standing reaches the 500-step criterion without deleting terminations or weakening the task to hide the fall.
