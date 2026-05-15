# Isaac Lab To MuJoCo Sim2sim

This document is the working flow for moving a `G0-Velocity-v0` policy from Isaac Lab into MuJoCo for sim2sim validation.

## Sim2sim Goal

The goal is to verify that a policy trained in Isaac Lab sees the same observation/action interface in MuJoCo, then gradually align dynamics until short rollouts are explainable.

The current branch builds the framework. It does not claim that MuJoCo walking is already correct.

## Why Exporting Policy Is Not Enough

A policy export only preserves network weights. It does not preserve:

- joint order
- default joint pose
- action scale
- observation order
- observation history
- projected gravity convention
- base angular velocity frame
- control dt and decimation
- actuator PD and torque limits
- contact geometry and friction

Any mismatch can make a valid Isaac Lab policy fail immediately in MuJoCo.

## Current Isaac Lab Policy Interface

- Task: `G0-Velocity-v0`
- Robot config: `G0_CFG`
- Action type: joint position action
- Joint order: `G0_JOINT_SDK_NAMES`
- Action scale: `0.12`
- Action formula: `target_joint_pos = default_joint_pos + 0.12 * clipped_policy_action`
- History length: `5`
- Isaac sim dt: `0.005`
- Isaac decimation: `4`
- Control dt: `0.02`

Policy observation terms:

```text
base_ang_vel
projected_gravity
velocity_commands
joint_pos_rel
joint_vel_rel
last_action
gait_phase
```

## Step 1: Export Policy From Isaac Lab

Run play with a trained checkpoint so the existing play path exports `policy.pt` and `policy.onnx`.

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task G0-Velocity-v0 \
  --num_envs 32 \
  --checkpoint <checkpoint-path>
```

Use the exported TorchScript policy first:

```text
logs/rsl_rl/g0_velocity/<run>/exported/policy.pt
```

ONNX rollout in the new MuJoCo script is still TODO.

## Step 2: Export Isaac Golden I/O

Zero-action golden data:

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/sim2sim/dump_isaac_golden_io.py \
  --task G0-Velocity-v0 \
  --steps 100 \
  --num_envs 1 \
  --output logs/sim2sim/isaac_golden_io.npz \
  --zero-action \
  --headless
```

Policy golden data using exported TorchScript:

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/sim2sim/dump_isaac_golden_io.py \
  --task G0-Velocity-v0 \
  --checkpoint logs/rsl_rl/g0_velocity/<run>/exported/policy.pt \
  --steps 100 \
  --num_envs 1 \
  --output logs/sim2sim/isaac_golden_io.npz \
  --headless
```

The dump includes observation, action, target joint position, joint state, root state when available, command, default pose, joint names, action scale, and timing metadata.

## Step 3: Prepare MuJoCo Model

Current file:

```text
mujoco/g0.xml
```

This first XML is an editable scaffold for interface validation. It is not a full converted dynamics model.

Future work should convert or manually clean up the URDF:

```text
source/g0_robot_lab/g0_robot_lab/assets/robots/g0/urdf/g0.urdf
```

and preserve all policy joint names.

## Step 4: Replicate Joint Order And Default Pose

Shared constants live in:

```text
scripts/sim2sim/g0_sim2sim_config.py
```

Check:

```bash
python scripts/sim2sim/validate_sim2sim_setup.py
```

The MuJoCo model must contain all 22 joints in `G0_JOINT_SDK_NAMES`. The software bridge builds qpos/qvel/actuator indices by name, not by guessed position.

## Step 5: Replicate Action Bridge

The action bridge is:

```text
target_joint_pos = default_joint_pos + 0.12 * clipped_policy_action
```

The helper is:

```text
scripts/sim2sim/g0_sim2sim_config.py::compute_target_joint_pos
```

Run:

```bash
python -m pytest tests/sim2sim/test_action_bridge.py -q
```

## Step 6: Replicate Observation Builder

The first MuJoCo observation builder is in:

```text
scripts/sim2sim/g0_mujoco_interface.py
```

It wires joint position, joint velocity, command, last action, and gait phase. `base_ang_vel` and `projected_gravity` are present but still require exact frame verification against Isaac Lab.

## Step 7: Run Zero-Action MuJoCo Check

```bash
python scripts/sim2sim/play_mujoco_g0.py \
  --model mujoco/g0.xml \
  --steps 1000 \
  --command 0.0 0.0 0.0 \
  --zero-action \
  --record-rollout logs/sim2sim/mujoco_rollout.npz
```

Zero action should map to `G0_DEFAULT_JOINT_POS`.

## Step 8: Run Policy MuJoCo Rollout

```bash
python scripts/sim2sim/play_mujoco_g0.py \
  --model mujoco/g0.xml \
  --policy logs/rsl_rl/g0_velocity/<run>/exported/policy.pt \
  --steps 1000 \
  --command 0.0 0.0 0.0 \
  --device cpu \
  --record-rollout logs/sim2sim/mujoco_rollout.npz
```

Do not interpret poor walking as policy failure until interface and dynamics alignment are checked.

## Step 9: Compare Isaac And MuJoCo Rollouts

```bash
python scripts/sim2sim/compare_isaac_mujoco_rollout.py \
  --isaac logs/sim2sim/isaac_golden_io.npz \
  --mujoco logs/sim2sim/mujoco_rollout.npz \
  --output logs/sim2sim/compare_report.md
```

The report compares joint position, joint velocity, action, target joint position, root height, and command when available.

## Common Failure Causes

- joint order is wrong
- action scale is wrong
- default pose is wrong
- projected gravity coordinate frame is wrong
- base angular velocity frame is wrong
- actuator PD is inconsistent
- torque limit is inconsistent
- velocity limit is inconsistent
- foot contact geometry is inconsistent
- friction is inconsistent
- control dt or decimation is inconsistent

## Current TODO

- Replace placeholder `mujoco/g0.xml` with a full URDF/MJCF-derived model.
- Verify projected gravity exactly against Isaac Lab.
- Verify base angular velocity exactly against Isaac Lab.
- Align actuator PD gains.
- Align torque and velocity limits.
- Align contact patch and friction.
- Add ONNX policy execution if needed.

## Recommended Command List

```bash
python -m pytest tests/sim2sim -q
python scripts/sim2sim/validate_sim2sim_setup.py
/home/lz/IsaacLab/isaaclab.sh -p scripts/sim2sim/dump_isaac_golden_io.py --zero-action --headless
python scripts/sim2sim/play_mujoco_g0.py --zero-action --record-rollout logs/sim2sim/mujoco_rollout.npz
python scripts/sim2sim/compare_isaac_mujoco_rollout.py
```
