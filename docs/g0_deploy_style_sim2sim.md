# G0 Deploy-Style MuJoCo Sim2sim

## Why Unitree Style

Unitree's sim2sim stack separates the learned policy from the low-level motor command bridge. The policy produces actions, the deploy controller converts those actions into target joint positions, and the MuJoCo bridge applies motor-like PD torque from `LowCmd` fields.

```text
Isaac Lab -> deploy.yaml + policy.onnx -> deploy controller -> LowCmd -> unitree_mujoco -> LowState
```

The G0 first pass now follows the same organization in Python:

```text
Isaac Lab -> g0 deploy.yaml -> policy.pt -> run_g0_mujoco_deploy.py -> MuJoCo rollout
```

## What This Adds

`scripts/sim2sim/export_g0_deploy_cfg.py` exports:

- `task`
- `joint_names`
- `joint_ids_map`
- `step_dt`
- `sim_dt`
- `decimation`
- `default_joint_pos`
- `stiffness`
- `damping`
- `effort_limit_sim`
- `velocity_limit_sim`
- `armature`
- `actions.JointPositionAction`
- `observations.terms`
- `observations.policy_obs_dim`
- `commands.base_velocity`
- `metadata`

`scripts/sim2sim/run_g0_mujoco_deploy.py` implements:

```text
obs
  -> policy_action
  -> clipped/processed action
  -> target_q = offset + scale * clipped_action
  -> q_des = target_q
  -> dq_des = 0
  -> tau_ff = 0
  -> tau_cmd = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
  -> effort-limit clipping
  -> MuJoCo control
```

No feed-forward torque is added in this pass.

## Commands

Export deploy config:

```bash
TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p scripts/sim2sim/export_g0_deploy_cfg.py \
  --task G0-Velocity-v0 \
  --output logs/sim2sim/g0_deploy/params/deploy.yaml \
  --headless
```

Zero-action position rollout:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/run_g0_mujoco_deploy.py \
  --model mujoco/g0.xml \
  --deploy-cfg logs/sim2sim/g0_deploy/params/deploy.yaml \
  --steps 200 \
  --command 0.0 0.0 0.0 \
  --zero-action \
  --control-mode position \
  --record-rollout logs/sim2sim/g0_deploy/mujoco_deploy_zero_action_position_rollout.npz
```

Check rollout:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/check_g0_deploy_rollout.py \
  --rollout logs/sim2sim/g0_deploy/mujoco_deploy_zero_action_position_rollout.npz \
  --deploy-cfg logs/sim2sim/g0_deploy/params/deploy.yaml \
  --output logs/sim2sim/g0_deploy/deploy_zero_action_position_check.md
```

PD torque mode:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/run_g0_mujoco_deploy.py \
  --model mujoco/g0.xml \
  --deploy-cfg logs/sim2sim/g0_deploy/params/deploy.yaml \
  --steps 200 \
  --command 0.0 0.0 0.0 \
  --zero-action \
  --control-mode pd_torque \
  --record-rollout logs/sim2sim/g0_deploy/mujoco_deploy_zero_action_pd_torque_rollout.npz
```

Policy rollout:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/run_g0_mujoco_deploy.py \
  --model mujoco/g0.xml \
  --deploy-cfg logs/sim2sim/g0_deploy/params/deploy.yaml \
  --policy logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/exported/policy.pt \
  --steps 200 \
  --command 0.0 0.0 0.0 \
  --device cpu \
  --control-mode position \
  --record-rollout logs/sim2sim/g0_deploy/mujoco_deploy_policy_position_rollout.npz
```

## Results From This Pass

Deploy export:

- Output: `logs/sim2sim/g0_deploy/params/deploy.yaml`
- Joint count: `22`
- Step dt: `0.02`
- Sim dt: `0.005`
- Decimation: `4`
- Policy obs dim: `385`
- Action scale: `0.12` for all 22 joints

Zero-action position mode:

- Rollout: `logs/sim2sim/g0_deploy/mujoco_deploy_zero_action_position_rollout.npz`
- Check: `logs/sim2sim/g0_deploy/deploy_zero_action_position_check.md`
- Result: OK
- `max_target_default_abs_err`: `0`
- `max_abs_policy_action`: `0`
- `max_abs_tau_cmd`: `0.578934`
- `max_abs_joint_vel`: `3.03837`
- `max_abs_root_ang_vel`: `0.000142146`

Zero-action pd_torque mode:

- Rollout: `logs/sim2sim/g0_deploy/mujoco_deploy_zero_action_pd_torque_rollout.npz`
- Check: `logs/sim2sim/g0_deploy/deploy_zero_action_pd_torque_check.md`
- Result: OK
- `max_target_default_abs_err`: `0`
- `max_abs_policy_action`: `0`
- `max_abs_tau_cmd`: `0.761774`
- `max_abs_joint_vel`: `3.68799`
- `max_abs_root_ang_vel`: `0.00019535`

Policy rollout:

- `policy.pt`: `logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/exported/policy.pt`
- Position mode: completed 200 deploy steps and check OK.
- PD torque mode: completed 200 deploy steps and check OK.
- This validates startup and command recording, not walking quality.

## Validation Matrix

The deploy validation matrix runs all combinations of:

- zero-action and policy rollout
- position and pd_torque control mode
- commands `[0.0, 0.0, 0.0]`, `[0.05, 0.0, 0.0]`, `[0.1, 0.0, 0.0]`, `[0.0, 0.0, 0.1]`

Command:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/run_g0_deploy_validation_matrix.py \
  --model mujoco/g0.xml \
  --deploy-cfg logs/sim2sim/g0_deploy/params/deploy.yaml \
  --policy logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/exported/policy.pt \
  --steps 200 \
  --output-dir logs/sim2sim/g0_deploy/validation_matrix
```

Latest 500-step result:

```text
Validation matrix OK: 16/16 cases OK
```

Summary report:

```text
docs/g0_deploy_sim2sim_validation_matrix_report.md
logs/sim2sim/g0_deploy/validation_matrix/validation_matrix_summary.md
```

Main 500-step signals:

- All 16 rollouts ran and passed checker.
- All zero-action cases tripped the early-fall heuristic.
- Some policy cases also tripped the early-fall heuristic at 500 steps after the corrected `base_ang_vel` convention.
- Policy action saturation is present; the highest 500-step case is `policy_position_c0_c0_c0`.
- Raw PD torque saturation is present but remains below the 5% smoke threshold.
- No case exceeded `velocity_limit_sim`.
- Current status is `sim2sim started`, not smoke pass and not credible pass.

Credibility criteria:

```text
docs/g0_sim2sim_credibility_criteria.md
```

Root-frame diagnostics:

```text
docs/root_frame_convention_diagnostics.md
```

Zero-action collapse analysis:

```text
docs/zero_action_collapse_analysis.md
```

## Position Mode Versus PD Torque Mode

`position` mode keeps the existing `mujoco/g0.xml` position actuators. It sends `q_des` to the MuJoCo position actuator and records the LowCmd-style torque that would be produced by the deploy bridge.

`pd_torque` mode generates `mujoco/g0_pd_torque.xml` with motor actuators and writes `pd_tau_cmd_clipped` directly into `data.ctrl`. This is closer to the Unitree bridge formula, but it is still a first Python implementation rather than Unitree's DDS/C++ bridge.

## Current Status

This is now a real starting point for G0 MuJoCo sim2sim verification because the deployed artifacts and runtime shape match the Unitree organization:

- deploy config exists
- policy loading exists for TorchScript `policy.pt`
- ONNX loading is implemented when `onnxruntime` is installed
- action processing is explicit
- `q_des`, `dq_des`, `kp`, `kd`, `tau_ff`, `tau_cmd`, and clipped torque are recorded
- zero-action and policy deploy rollouts run

## Still Different From Unitree

- No DDS `LowCmd` / `LowState` bridge.
- No C++ ONNX `OrtRunner` deploy controller.
- Python runner owns the loop.
- Observation frame conventions still require controlled diagnostics.
- `pd_torque` timing and sensing are first-pass approximations.
- Contact, friction, and solver behavior are not tuned here.

## Next Steps

1. Export and verify `policy.onnx`.
2. Split the deploy controller logic into a reusable Python controller mirroring Unitree's C++ manager structure.
3. Add optional DDS bridge experiments only after the Python loop is stable.
4. Run controlled frame diagnostics for `base_ang_vel` and `projected_gravity`.
5. Continue contact/friction/solver validation without changing foot mesh geometry.
