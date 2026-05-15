# Unitree-Style Sim2sim Gap Analysis

## Context

- Branch: `structure/mujoco-sim2sim-layout`
- Task: `G0-Velocity-v0`
- Reference repositories:
  - `https://github.com/unitreerobotics/unitree_mujoco`
  - `https://github.com/unitreerobotics/unitree_rl_lab`

This note compares the Unitree deployment organization with the current G0 MuJoCo sim2sim stack. The goal of this pass is to create a deploy-style validation entrypoint, not to tune walking quality.

## Unitree Sim2sim Chain

Unitree's sim2sim organization is not just "call a policy inside MuJoCo." The chain is closer to:

```text
Isaac Lab training
  -> exported policy.onnx
  -> exported params/deploy.yaml
  -> deploy controller
  -> LowCmd q / dq / kp / kd / tau fields
  -> unitree_mujoco
  -> LowState motor state and IMU feedback
  -> deploy controller
```

In `unitree_mujoco`, the bridge converts each motor command into MuJoCo torque control:

```text
ctrl = tau + kp * (q_des - q) + kd * (dq_des - dq)
```

In `unitree_rl_lab`, `export_deploy_cfg.py` writes deployment metadata such as `joint_ids_map`, `step_dt`, `stiffness`, `damping`, `default_joint_pos`, `actions`, `observations`, and `commands`. The C++ deploy controller loads `params/deploy.yaml` and `exported/policy.onnx`, constructs observations, runs the ONNX policy, processes actions through an IsaacLab-style `ActionManager`, and writes the processed joint-position targets into `LowCmd.motor_cmd[i].q`. Gains and feed-forward fields are LowCmd fields, not policy outputs.

## Current G0 Chain Before This Pass

The G0 project already had a useful direct sim2sim scaffold:

```text
Isaac Lab G0-Velocity-v0
  -> deterministic golden I/O
  -> g0_sim2sim_config.py constants
  -> play_mujoco_g0.py
  -> MuJoCo position actuator rollout
```

Important existing pieces:

- `mujoco/g0.xml` is a URDF-derived working model.
- Foot collision remains URDF-derived STL mesh collision.
- No foot box, capsule, or sphere has been added.
- Self-collision filtering is aligned with Isaac `enabled_self_collisions=False`.
- First-pass actuator stiffness, effort, damping, and armature alignment exists.
- Deterministic zero-action Isaac golden I/O exists.
- Command mismatch diagnostics have been addressed.
- Foot contact forces can be exported from both Isaac and MuJoCo.
- Zero-action velocity-limit diagnostics report `0/22` exceeded joints.
- Zero-action dynamics compare tooling exists.

## Gaps Against Unitree

- `deploy.yaml` was missing.
- A deploy-style MuJoCo runner was missing.
- `play_mujoco_g0.py` was a direct runner, not a deploy controller analogue.
- The recorded control chain did not explicitly expose `q_des`, `dq_des`, `kp`, `kd`, `tau_ff`, and `tau_cmd`.
- Policy action to `q_des` was implemented as a helper constant path, not exported as deployment metadata.
- Observation/action construction still has hand-written Python pieces rather than a deploy controller consuming only `deploy.yaml`.
- There is no DDS `LowCmd` / `LowState` bridge.
- There is no C++ ONNX deploy controller for G0.

## Filled In This Pass

- Added `scripts/sim2sim/export_g0_deploy_cfg.py`.
- Added `scripts/sim2sim/run_g0_mujoco_deploy.py`.
- Added `scripts/sim2sim/check_g0_deploy_rollout.py`.
- Added deploy rollout recording for:
  - `policy_action`
  - `processed_action`
  - `target_joint_pos`
  - `pd_q_des`
  - `pd_dq_des`
  - `pd_kp`
  - `pd_kd`
  - `pd_tau_ff`
  - `pd_tau_cmd`
  - `pd_tau_cmd_clipped`
- Added `position` mode, which keeps the current position actuator backend but records the Unitree-style PD command.
- Added `pd_torque` mode, which generates `mujoco/g0_pd_torque.xml` with motor actuators and writes clipped torque commands to MuJoCo.
- Added rollout checks for shape, target formula, effort clipping, joint order, finite values, root/contact ranges, velocity-limit exceedance, and zero-action metrics.

## Explicitly Not Done

- No DDS bridge.
- No C++ ONNX `OrtRunner` deploy controller for G0.
- No real robot control path.
- No formal foot collision simplification.
- No foot box, capsule, or sphere.
- No physics tuning of kp, damping, friction, solver, or root height.
- No claim that policy rollout quality is validated.

## Remaining Difference

The project can now start MuJoCo sim2sim validation in a Unitree-style organization, but it is still not identical to Unitree's production chain. The main remaining differences are the missing DDS bridge, missing C++ ONNX deploy process, Python-only runner, observation frame conventions that still need controlled diagnostics, and first-pass `pd_torque` behavior that has not been matched against Unitree's bridge timing or sensor ordering beyond the command formula.
