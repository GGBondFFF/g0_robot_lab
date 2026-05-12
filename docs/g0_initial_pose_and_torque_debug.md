# G0 Initial Pose And Torque Debug

## Why Log Torque In Real Time

The previous zero-action sweeps only showed that G0 fell, not why it fell. Per-joint torque logging lets us see whether the fall starts because a key joint reaches its effort limit, because the contact/COM geometry is already wrong, or because the pose is dynamically unstable even when torque margin remains.

The key signal is `torque_ratio = abs(torque) / effort_limit`.

- `ratio > 0.90` early and repeatedly on ankle, knee, or hip means that joint is likely torque limited.
- `ratio < 0.90` while the robot still tips over means the failure is more likely pose, COM, inertia, contact patch, or root height.
- One short saturation spike is less important than many consecutive saturated steps before termination.

## Debug-Only Effort Scale

`--effort-scale` temporarily scales actuator `effort_limit_sim` inside the debug script only. It is used to answer the question: "Would the same pose stand if the torque limit were higher?"

It is not a hardware parameter and is not written to `g0_actuators.py`.

The script blocks values above `2.0` on purpose. If the robot still fails at `2.0`, the next step is not more torque; the next step is asset/contact/inertia inspection.

## Two-Decimal Initial Pose Rule

Initial pose values are now limited to two decimals. G0 is about 49 cm tall and about 1.9 kg, so a useful standing pose should not require `0.0001 rad` tuning precision. The debug grid therefore uses:

```text
root_z:      0.22, 0.23, 0.24
hip_pitch:   0.18, 0.19, 0.20, 0.21
knee_pitch:  0.33, 0.34, 0.35, 0.36
ankle_pitch: 0.13, 0.14, 0.15, 0.16
```

The sign mirroring is handled by the script according to the G0 left/right pitch axes.

## Commands

Fixed torque trace:

```bash
TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p scripts/debug/sweep_g0_standing_pose.py \
  --task G0-Velocity-v0 \
  --headless \
  --mode fixed \
  --steps 500 \
  --print-torque-every 10 \
  --effort-scale 1.0
```

Effort scale debug:

```bash
for scale in 1.0 1.2 1.5 2.0; do
  TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p scripts/debug/sweep_g0_standing_pose.py \
    --task G0-Velocity-v0 \
    --headless \
    --mode fixed \
    --steps 500 \
    --print-torque-every 10 \
    --effort-scale "$scale"
done
```

Quick two-decimal sweep:

```bash
TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p scripts/debug/sweep_g0_standing_pose.py \
  --task G0-Velocity-v0 \
  --headless \
  --mode sweep \
  --candidate-set quick \
  --steps 500 \
  --print-torque-every 0 \
  --effort-scale 1.0 \
  --top_k 6
```

Full grid:

```bash
TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p scripts/debug/sweep_g0_standing_pose.py \
  --task G0-Velocity-v0 \
  --headless \
  --mode sweep \
  --candidate-set grid \
  --steps 500 \
  --print-torque-every 0 \
  --effort-scale 1.0
```

## Test Rounds

| round | root_z | hip | knee | ankle | effort_scale | trace |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fixed nominal | 0.23 | 0.20 | 0.35 | 0.14 | 1.0 | `logs/zero_action_torque_trace_20260512_154804.csv` |
| fixed scale | 0.23 | 0.20 | 0.35 | 0.14 | 1.2 | `logs/zero_action_torque_trace_20260512_154844.csv` |
| fixed scale | 0.23 | 0.20 | 0.35 | 0.14 | 1.5 | `logs/zero_action_torque_trace_20260512_154917.csv` |
| fixed scale | 0.23 | 0.20 | 0.35 | 0.14 | 2.0 | `logs/zero_action_torque_trace_20260512_154940.csv` |
| quick sweep | six requested two-decimal candidates | | | | 1.0 | `logs/zero_action_torque_trace_20260512_155017.csv` |
| quick sweep | six requested two-decimal candidates | | | | 2.0 | `logs/zero_action_torque_trace_20260512_155047.csv` |

## Fixed Candidate Results

| effort_scale | survive | reason | final_root_z | final_pitch | max hip ratio | max knee ratio | max ankle ratio |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1.0 | 91/500 | bad_orientation | 0.1262 | 70.137 | 0.397 | 0.597 | 1.000 |
| 1.2 | 91/500 | bad_orientation | 0.1262 | 70.137 | n/a | n/a | 1.000 |
| 1.5 | 91/500 | bad_orientation | 0.1262 | 70.137 | n/a | n/a | 0.959 |
| 2.0 | 91/500 | bad_orientation | 0.1262 | 70.137 | n/a | n/a | 0.720 |

At nominal effort the earliest saturation is:

```text
l_ankle_pitch_joint at step 57
r_ankle_pitch_joint at step 57
```

Nominal key joint max values:

```text
l_ankle_pitch_joint max_abs_torque=1.1667 max_ratio=1.000 sat_steps=26/91
r_ankle_pitch_joint max_abs_torque=1.1667 max_ratio=1.000 sat_steps=27/91
l_ankle_roll_joint  max_abs_torque=0.9019 max_ratio=0.902 sat_steps=2/91
knee pitch joints   max_ratio about 0.60
hip pitch joints    max_ratio about 0.40
```

## Quick Sweep Results

Best candidate at `effort_scale=1.0`:

```text
root_z=0.23 hip=0.20 knee=0.34 ankle=0.14
survive=116/500
reason=bad_orientation
final_root_z=0.120
final_pitch=71.823 deg
max_hip_ratio=0.410
max_knee_ratio=0.656
max_ankle_ratio=1.000
```

The same candidate at `effort_scale=2.0`:

```text
root_z=0.23 hip=0.20 knee=0.34 ankle=0.14
survive=116/500
reason=bad_orientation
final_root_z=0.120
final_pitch=71.823 deg
max_hip_ratio=0.205
max_knee_ratio=0.328
max_ankle_ratio=0.751
```

This is the most important result: the fall remains even when the torque ratios are comfortably below saturation. The ankle is the first nominal-limit bottleneck, but torque limit alone is not the root cause.

## Current Best Pose

Current best two-decimal pose:

```text
root_z=0.23
hip_pitch=0.20
knee_pitch=0.34
ankle_pitch=0.14
```

This pose is now the simple baseline in `g0.py`, but it is not declared stable. It only survives 116/500 steps in the current debug setup.

## 500-Step Status

No candidate passed 500 zero-action steps.

No minimum usable `effort_scale` has been found for standing. `effort_scale=2.0` removes long saturation for the best quick-sweep candidate, but the robot still tips over and terminates on bad orientation.

## Disable-Termination Observation

`--disable-terminations` was added as a debug-only mode. It removes `base_height` and `bad_orientation` from the environment termination manager, but still logs when those conditions would have triggered.

Command:

```bash
TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p scripts/debug/sweep_g0_standing_pose.py \
  --task G0-Velocity-v0 \
  --headless \
  --mode fixed \
  --steps 500 \
  --root_z 0.23 \
  --hip 0.20 \
  --knee 0.34 \
  --ankle 0.14 \
  --effort-scale 2.0 \
  --print-torque-every 10 \
  --disable-terminations
```

Result:

```text
trace=logs/zero_action_torque_trace_20260512_164624.csv
first would-terminate step=115
first would-terminate reason=bad_orientation
final_root_z=0.0468
min_root_z=0.0463
final_pitch=85.136 deg
max_abs_pitch=86.502 deg
max_hip_ratio=0.205
max_knee_ratio=0.993
max_ankle_ratio=1.000
```

Natural fall trend:

```text
step 0:   COM x=0.0022, root_z=0.233, pitch=-1.49 deg, foot force=0/0
step 50:  COM x=0.0167, root_z=0.232, pitch=0.49 deg
step 80:  COM x=0.0427, root_z=0.230, pitch=8.22 deg
step 100: COM x=0.1108, root_z=0.222, pitch=27.24 deg
step 110: COM x=0.2122, root_z=0.190, pitch=49.98 deg
step 120: COM x=0.3421, root_z=0.047, pitch=83.35 deg, roll=-90.34 deg
```

This confirms the robot rotates and collapses forward/sideways even when formal termination is disabled. It is not merely an overly strict termination issue.

## Next Inspection Targets

Before PPO training, inspect:

- COM and body inertial parameters in the robot asset
- foot collision geometry and contact patch size
- foot link frame height versus actual sole collision height
- root initial height and whether feet start penetrated or suspended
- support polygon relative to COM in the two-decimal default pose

Do not solve this by deleting termination, weakening rewards, or permanently inflating hardware constants.
