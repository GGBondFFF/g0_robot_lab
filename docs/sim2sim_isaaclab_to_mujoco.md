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

## Step 1: Pin The Checkpoint

All sim2sim validation commands must use the explicitly pinned absolute checkpoint path. Do not use `latest`, an exported `policy.pt`, or any default checkpoint selection while doing checkpoint identity audits.

```bash
CKPT=/home/lz/g0_robot_lab/g0_robot_lab/logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/model_9999.pt
```

The sim2sim runners accept this raw RSL-RL checkpoint directly and record path, filename, SHA256, run folder, task, command, dimensions, joint names, and action scale in rollout artifacts.

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

Policy golden data using the pinned checkpoint:

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/sim2sim/dump_isaac_golden_io.py \
  --task G0-Velocity-v0 \
  --checkpoint /home/lz/g0_robot_lab/g0_robot_lab/logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/model_9999.pt \
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
  --policy /home/lz/g0_robot_lab/g0_robot_lab/logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/model_9999.pt \
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
It also reports per-observation-term history diffs and the worst per-joint action/target/state diffs.

## Step 10: Replay Isaac Actions Or Targets In MuJoCo

Replay processed Isaac target joint positions without running the policy:

```bash
python scripts/sim2sim/play_mujoco_g0.py \
  --model mujoco/g0.xml \
  --steps 100 \
  --replay logs/sim2sim/isaac_golden_io.npz \
  --replay-field target_joint_pos \
  --record-rollout logs/sim2sim/mujoco_replay_isaac_target.npz
```

Replay Isaac policy actions through the MuJoCo action bridge:

```bash
python scripts/sim2sim/play_mujoco_g0.py \
  --model mujoco/g0.xml \
  --steps 100 \
  --replay logs/sim2sim/isaac_golden_io.npz \
  --replay-field action \
  --record-rollout logs/sim2sim/mujoco_replay_isaac_action.npz
```

Interpretation:

- If replaying `target_joint_pos` fails, debug MuJoCo model/control/dynamics/contact before blaming policy.
- If replaying `action` differs from replaying `target_joint_pos`, debug action scale, clipping, default pose, or joint order.
- If both replay modes are reasonable but live policy rollout fails, debug observation history, frame convention, and policy input scaling.

## Current Audit Snapshot

Checkpoint identity audit policy:

```text
/home/lz/g0_robot_lab/g0_robot_lab/logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/model_9999.pt
```

Key results from this branch:

```text
pytest tests/sim2sim: 23 passed
validate_sim2sim_setup: OK, all 22 policy joints found
actuator alignment table: 22/22 first-pass rows aligned
zero-action MuJoCo target-default max error: 0
zero-action MuJoCo root height over 100 steps: min 0.027547, final 0.040076
zero-action deterministic Isaac root height over 100 steps: min 0.130989, final 0.231955
policy cmd=(0.1,0,0) Isaac 1000 steps: root height min 0.227384, final 0.230000
policy cmd=(0.1,0,0) MuJoCo 1000 steps: root height min 0.033228, final 0.035982
```

The current stage conclusion is:

```text
A. The checked policy is not the primary demonstrated failure.
```

The strongest evidence is that Isaac stays upright for 1000 policy steps with the same fixed command where MuJoCo eventually collapses, while zero-action target replay is exactly aligned at the action/target level but diverges strongly in root height. This points first to MuJoCo model/contact/control/dynamics and longer-horizon observation drift, not to a broken export or a clearly bad Isaac policy.

Do not retrain solely because MuJoCo policy rollout fails until the model/contact/control gaps below are resolved.

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

- Replace the current first-pass `mujoco/g0.xml` with a fully audited MJCF model or document every deviation from the Isaac USD/URDF.
- Verify projected gravity against controlled Isaac/MuJoCo root-state samples.
- Verify base angular velocity against controlled Isaac/MuJoCo root-state samples and MuJoCo gyro sensors.
- Align the control backend more carefully: MuJoCo position actuator, explicit PD torque, Isaac implicit actuator damping, and effort clipping are not proven equivalent.
- Align or explicitly model torque and velocity limits.
- Audit contact patch, friction, solver settings, mass/inertia, and root initialization.
- Add ONNX policy execution if needed.

## Unitree Mujoco Recommendation

`unitree_mujoco` is useful as an architecture reference, especially for a future SDK-style low-level interface:

- It already has C++ and Python MuJoCo simulators.
- It publishes/subscribes Unitree SDK2 low-level messages.
- `LowCmd` carries `q`, `dq`, `kp`, `kd`, and `tau`; the bridge applies `tau + kp*(q-q_sensor) + kd*(dq-dq_sensor)`.
- `LowState`, `SportModeState`, and IMU messages expose motor state, base pose/velocity, and IMU-style feedback.
- `unitree_robots` organizes supported robot MJCFs with actuators and sensors in hardware/SDK order.

This branch now contains a minimal Python Unitree-style G0 prototype rather than only a feasibility note:

```text
scripts/unitree_mujoco_g0/
mujoco/unitree_mujoco_g0/
```

The generated MJCF is derived from the current URDF-derived `mujoco/g0.xml`; it is not a fake interface-only model. The generator replaces position actuators with Unitree-style torque motors, adds joint position/velocity/actuator-force sensors in `G0_JOINT_SDK_NAMES` order, adds an IMU site/sensors, and writes:

```text
mujoco/unitree_mujoco_g0/g0.xml
mujoco/unitree_mujoco_g0/scene.xml
```

The Python bridge implements explicit LowCmd-like fields:

```text
q, dq, tau, kp, kd
```

and applies the Unitree control law:

```text
ctrl = tau + kp * (q_des - q_sensor) + kd * (dq_des - dq_sensor)
```

with Isaac/G0 actuator stiffness, damping, and effort limits.

Commands:

```bash
conda run -n g0_isaaclab python scripts/unitree_mujoco_g0/generate_g0_unitree_mjcf.py

conda run -n g0_isaaclab python scripts/unitree_mujoco_g0/run_unitree_mujoco_g0_zero_action.py \
  --steps 100 \
  --record-rollout logs/sim2sim/unitree_g0_zero_action_100.npz

conda run -n g0_isaaclab python scripts/unitree_mujoco_g0/run_unitree_mujoco_g0_replay_target.py \
  --steps 1000 \
  --golden logs/sim2sim/isaac_model_9999_cmd01.npz \
  --record-rollout logs/sim2sim/unitree_g0_replay_policy_target_cmd01_1000.npz

conda run -n g0_isaaclab python scripts/unitree_mujoco_g0/run_unitree_mujoco_g0_policy.py \
  --steps 1000 \
  --policy /home/lz/g0_robot_lab/g0_robot_lab/logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/model_9999.pt \
  --command 0.1 0 0 \
  --record-rollout logs/sim2sim/unitree_g0_policy_cmd01_1000.npz

conda run -n g0_isaaclab python scripts/unitree_mujoco_g0/compare_isaac_unitree_mujoco_rollout.py \
  --isaac logs/sim2sim/isaac_model_9999_cmd01.npz \
  --custom logs/sim2sim/custom_mujoco_policy_model_9999_cmd01.npz \
  --unitree logs/sim2sim/unitree_g0_policy_cmd01_1000.npz \
  --output logs/sim2sim/compare_threeway_policy_cmd01_1000.md
```

Prototype results:

```text
zero-action 100 steps:
  Isaac final root height: 0.231955
  custom MuJoCo final root height: 0.040076
  unitree-style final root height: 0.040035

replay Isaac policy target, cmd=(0.1,0,0), 1000 steps:
  Isaac final root height: 0.230000
  custom MuJoCo final root height: 0.040639
  unitree-style final root height: 0.045913

live policy, cmd=(0.1,0,0), 1000 steps:
  Isaac final root height: 0.230000
  custom MuJoCo final root height: 0.035982
  unitree-style final root height: 0.034414
  custom action saturation ratio: 0.450955
  unitree-style action saturation ratio: 0.270182
  unitree-style torque saturation ratio: 0.002182
```

Interpretation:

- Unitree-style LowCmd control does not fix zero-action collapse.
- Replaying Isaac target positions through LowCmd torque control is only slightly better in root height than custom replay and still collapses.
- Live policy shows some mid-rollout improvement and lower action saturation than the custom position-actuator rollout, but it still collapses by 1000 steps.
- Because replayed Isaac targets also collapse, the main remaining blocker is still G0 MJCF/contact/dynamics fidelity, not policy observation drift alone.

For this G0 policy validation, do not fully migrate yet:

- G0 is not one of the official Unitree robots in that repo.
- The current policy is Isaac Lab/RSL-RL, not a controller already written around Unitree SDK2 messages.
- The Python prototype proves the LowCmd-style chain is easy to host locally, but it does not solve the model/contact mismatch by itself.
- A full unitree_mujoco fork is most valuable after the G0 MJCF/contact model is made credible.

Recommended path:

```text
short term: continue current custom sim2sim branch until model/contact/control are explainable
long term: borrow or migrate toward unitree_mujoco's LowCmd/LowState architecture if real low-level interface validation becomes the goal
```

## Recommended Command List

```bash
python -m pytest tests/sim2sim -q
python scripts/sim2sim/validate_sim2sim_setup.py
/home/lz/IsaacLab/isaaclab.sh -p scripts/sim2sim/dump_isaac_golden_io.py --zero-action --headless
python scripts/sim2sim/play_mujoco_g0.py --zero-action --record-rollout logs/sim2sim/mujoco_rollout.npz
python scripts/sim2sim/compare_isaac_mujoco_rollout.py
```
