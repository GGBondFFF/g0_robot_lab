# Observation And Action Interface

This document records the current `G0-Velocity-v0` observation and action interface. It is especially important for later MuJoCo sim2sim and hardware deployment.

Current config file:

```text
source/g0_robot_lab/g0_robot_lab/tasks/locomotion/robots/g0/velocity_env_cfg.py
```

## Policy Observation

The policy observation group currently contains these terms:

```text
base_ang_vel
projected_gravity
velocity_commands
joint_pos_rel
joint_vel_rel
last_action
gait_phase
```

Current policy observation settings:

```text
history_length = 5
enable_corruption = True
concatenate_terms = True
```

Important notes:

- `base_ang_vel` is scaled by `0.2` and has uniform noise.
- `projected_gravity` has uniform noise.
- `velocity_commands` comes from the generated `base_velocity` command.
- `joint_pos_rel` and `joint_vel_rel` use the robot joint order from Isaac Lab's articulation/action interface.
- `last_action` is the previous policy action.
- `gait_phase` currently uses `period=0.8`.

The critic observation group additionally includes:

```text
base_lin_vel
```

and otherwise mirrors the main locomotion state terms without the same corruption settings.

## Action Interface

The current action is joint position control:

```text
mdp.JointPositionActionCfg
```

Current action settings:

```text
asset_name = "robot"
joint_names = list(G0_JOINT_SDK_NAMES)
scale = 0.12
use_default_offset = True
preserve_order = True
```

The action dimension is the number of G0 SDK joints:

```text
22
```

## Deployment Policy IO Contract

The raw RSL-RL training checkpoint, for example `model_9999.pt`, is not the final deployment inference artifact. It is used for checkpoint identity and load sanity checks.

Deployment inference artifacts are exported by `scripts/rsl_rl/play.py` into the run directory's `exported/` folder:

```text
exported/policy.pt
exported/policy.onnx
```

The exported policy IO contract is:

```text
observation dim = 385
action dim = 22
```

This comes from 77 policy observation values per frame and `history_length = 5`. The exported `policy.pt` and `policy.onnx` release-gate tests validate `385 -> 22` when the artifacts exist. ONNX validation requires `onnxruntime` installed in `g0_isaaclab`.

## Joint Order

Policy actions are ordered by:

```text
G0_JOINT_SDK_NAMES
```

Current order:

```text
l_hip_pitch_joint
l_hip_roll_joint
l_hip_yaw_joint
l_knee_pitch_joint
l_ankle_pitch_joint
l_ankle_roll_joint
r_hip_pitch_joint
r_hip_roll_joint
r_hip_yaw_joint
r_knee_pitch_joint
r_ankle_pitch_joint
r_ankle_roll_joint
waist_yaw_joint
waist_roll_joint
l_shoulder_pitch_joint
l_shoulder_roll_joint
l_shoulder_yaw_joint
l_elbow_pitch_joint
r_shoulder_pitch_joint
r_shoulder_roll_joint
r_shoulder_yaw_joint
r_elbow_pitch_joint
```

`preserve_order=True` is required so this SDK/deployment-friendly order remains the policy action order.

## Target Joint Position Formula

Because `use_default_offset=True`, the policy action is interpreted as an offset around the default joint pose:

```text
target_joint_pos = default_joint_pos + action_scale * policy_action
```

With the current scale:

```text
target_joint_pos = default_joint_pos + 0.12 * policy_action
```

The `default_joint_pos` values come from `G0_CFG.init_state.joint_pos`.

## Sim2sim And Deployment Notes

For MuJoCo sim2sim and hardware deployment, the following must be kept aligned exactly:

- joint order
- default joint pose
- action scale
- action clipping/normalization convention
- observation term order
- observation history length
- projected gravity convention
- base angular velocity frame and scaling
- generated command order and scaling
- gait phase convention
- control dt and decimation

Small mismatches in joint order, default pose, or observation order can make a valid Isaac Lab policy fail immediately in another simulator.
