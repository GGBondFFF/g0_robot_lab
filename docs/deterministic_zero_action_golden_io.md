# Deterministic Zero-Action Golden I/O

## Context

- Branch: `structure/mujoco-sim2sim-layout`
- Test time: `2026-05-15 17:13 CST`
- Task: `G0-Velocity-v0`
- Goal: make Isaac zero-action golden I/O cleaner for MuJoCo dynamics comparison.

This pass does not change `velocity_env_cfg.py`. All deterministic settings are applied only to the in-memory `env_cfg` instance inside `scripts/sim2sim/dump_isaac_golden_io.py`.

## Command

```bash
TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p scripts/sim2sim/dump_isaac_golden_io.py \
  --task G0-Velocity-v0 \
  --steps 100 \
  --num_envs 1 \
  --output logs/sim2sim/isaac_zero_action_deterministic_golden_io.npz \
  --zero-action \
  --headless \
  --deterministic-zero \
  --command 0.0 0.0 0.0
```

Result:

```text
Saved Isaac golden I/O: logs/sim2sim/isaac_zero_action_deterministic_golden_io.npz
```

## Deterministic Overrides

The script records the applied overrides in `deterministic_overrides`.

Current overrides:

```text
deterministic_zero: True
command: (0.0, 0.0, 0.0)
root_pos: (0.0, 0.0, 0.23)
root_yaw: 0.0
zero_joint_velocity: True
reset_base pose_range x/y/yaw: fixed
reset_base velocity_range: all zero
reset_robot_joints velocity_range: zero
command ranges: fixed to command
physics material randomization: fixed ranges, one bucket
base mass randomization: zero mass delta
push_robot: disabled
policy observation corruption: disabled
seed: 0
```

The command was successfully fixed:

```text
command max abs: 0.0
```

## Exported Fields

The deterministic Isaac golden file contains:

```text
obs
action
target_joint_pos
joint_pos
joint_vel
root_pos
root_height
root_quat
base_ang_vel
projected_gravity
command
joint_acc
contact_force
foot_contact_force
left_foot_contact_force
right_foot_contact_force
foot_contact_force_norm
left_foot_contact_force_norm
right_foot_contact_force_norm
default_joint_pos
joint_names
action_scale
sim_dt
decimation
control_dt
deterministic_zero
deterministic_overrides
```

Foot force export succeeded:

```text
contact_force shape: (100, 23, 3)
foot_contact_force shape: (100, 2, 3)
left_foot_contact_force shape: (100, 3)
right_foot_contact_force shape: (100, 3)
foot_contact_force_norm shape: (100,)
```

Isaac foot force norm summary:

```text
min / mean / max: 0.856135 / 8.72885 / 11.8376
```

## Notes

- This deterministic mode is for sim2sim diagnostics only.
- It does not modify the training/play baseline config files.
- The exported contact forces come from the existing `contact_forces` sensor over `Robot/.*`, filtered to `l_foot_link` and `r_foot_link`.
- The result still must not be interpreted as policy performance. It is a cleaner zero-action dynamics reference.
