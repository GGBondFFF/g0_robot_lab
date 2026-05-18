# G0 Sim2sim Scripts

This directory contains the Isaac Lab to MuJoCo sim2sim validation scaffold for `G0-Velocity-v0`.

## Files

- `g0_sim2sim_config.py`: shared G0 joint order, default pose, action scale, timing, and action bridge helpers.
- `g0_mujoco_interface.py`: MuJoCo model loader, joint/action index bridge, action application, and observation skeleton.
- `play_mujoco_g0.py`: MuJoCo rollout runner for zero-action or exported TorchScript policy.
- `dump_isaac_golden_io.py`: Isaac Lab golden I/O dumper. Must be launched through Isaac Lab.
- `compare_isaac_mujoco_rollout.py`: compares Isaac and MuJoCo `.npz` rollouts and writes a Markdown report.
- `compare_first_frame_observation.py`: compares first-frame observation terms instead of only flattened rollout arrays.
- `compare_zero_action_dynamics.py`: compares zero-action state, root, contact, and acceleration diagnostics.
- `check_mujoco_velocity_limits.py`: reports observed MuJoCo joint velocities against Isaac `velocity_limit_sim`.
- `inspect_g0_urdf_for_mujoco.py`: read-only URDF readiness inspection for MuJoCo migration.
- `inspect_g0_usd_collision.py`: USD/PhysX foot collision inspection. Must be launched through Isaac Lab.
- `inspect_mujoco_collision.py`: MuJoCo foot geom and initial contact inspection.
- `generate_g0_mujoco_from_urdf.py`: generates the current URDF-derived `mujoco/g0.xml` working model.
- `export_g0_actuator_alignment_table.py`: writes an Isaac-vs-MuJoCo actuator parameter table.
- `export_g0_deploy_cfg.py`: exports a Unitree-style `deploy.yaml` from the Isaac Lab G0 env.
- `run_g0_mujoco_deploy.py`: deploy-style MuJoCo runner that records LowCmd-like PD fields.
- `check_g0_deploy_rollout.py`: checks deploy rollout shape, action processing, PD command clipping, and zero-action metrics.
- `run_g0_deploy_validation_matrix.py`: runs the deploy matrix over zero-action/policy, position/pd_torque, and fixed velocity commands.
- `validate_sim2sim_setup.py`: quick structure and interface validator.

## Validate Setup

```bash
python scripts/sim2sim/validate_sim2sim_setup.py
```

Validate the preserved placeholder model:

```bash
python scripts/sim2sim/validate_sim2sim_setup.py --model mujoco/g0_interface_placeholder.xml
```

If ordinary Python cannot import the project package, run through Isaac Lab:

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/sim2sim/validate_sim2sim_setup.py
```

If MuJoCo is not installed, the XML load check is skipped with a warning.

After installing DeepMind MuJoCo in the `g0_isaaclab` environment, the expected validation summary is:

```text
OK: MuJoCo model has all 22 joints
Summary: OK
```

## Dump Isaac Golden I/O

## Inspect And Generate MuJoCo Model

Inspect URDF migration readiness:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/inspect_g0_urdf_for_mujoco.py
```

Regenerate the URDF-derived working model:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/generate_g0_mujoco_from_urdf.py
```

The generated `mujoco/g0.xml` is a working model, not final dynamics. The preserved interface scaffold is `mujoco/g0_interface_placeholder.xml`.

The generator maps Isaac `G0_CFG` actuator parameters into MJCF:

```text
stiffness -> position actuator kp
effort_limit_sim -> position actuator forcerange
damping -> joint damping, first-pass approximation
armature -> joint armature
```

Export the current alignment table:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/export_g0_actuator_alignment_table.py
```

Expected current summary:

```text
aligned rows: 22/22
```

`velocity_limit_sim` is recorded in the table but is not yet an exactly equivalent MuJoCo actuator limit.

Inspect contact fidelity:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/inspect_mujoco_collision.py --model mujoco/g0.xml --steps 1

TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p scripts/sim2sim/inspect_g0_usd_collision.py --headless
```

`mujoco/g0.xml` is generated with Isaac-style self-collision filtering:

```text
ground: contype=1, conaffinity=2
robot:  contype=2, conaffinity=1
```

MuJoCo contacts are enabled when `(contype1 & conaffinity2) || (contype2 & conaffinity1)` is non-zero, so this keeps robot-ground contact and disables robot-robot self-contact. Foot geoms remain URDF-derived mesh geoms; no formal foot box is added.

## Dump Isaac Golden I/O

Zero-action dump:

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/sim2sim/dump_isaac_golden_io.py \
  --task G0-Velocity-v0 \
  --steps 100 \
  --num_envs 1 \
  --output logs/sim2sim/isaac_golden_io.npz \
  --zero-action \
    --headless
```

Deterministic zero-action dump:

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

`--deterministic-zero` temporarily freezes the in-memory env config for diagnostics:

```text
reset_base x/y/yaw fixed
reset_base velocity fixed to zero
reset_robot_joints velocity fixed to zero
base_velocity command range fixed to --command
physics material randomization fixed
base mass randomization zeroed
push_robot disabled
policy observation corruption disabled
```

It also exports foot contact diagnostics when available: `contact_force`, `foot_contact_force`, left/right foot contact force vectors, and left/right/total foot contact force norms.

Policy dump using the pinned absolute raw RSL-RL checkpoint:

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/sim2sim/dump_isaac_golden_io.py \
  --task G0-Velocity-v0 \
  --checkpoint /home/lz/g0_robot_lab/g0_robot_lab/logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/model_9999.pt \
  --steps 100 \
  --num_envs 1 \
  --output logs/sim2sim/isaac_golden_io.npz \
  --headless
```

Sim2sim checkpoint identity audits must not use `latest`, a relative checkpoint path, or exported `policy.pt`.

## Run MuJoCo Rollout

Zero-action rollout:

```bash
python scripts/sim2sim/play_mujoco_g0.py \
  --model mujoco/g0.xml \
  --steps 1000 \
  --command 0.0 0.0 0.0 \
  --zero-action \
  --record-rollout logs/sim2sim/mujoco_rollout.npz
```

Pinned checkpoint policy rollout:

```bash
python scripts/sim2sim/play_mujoco_g0.py \
  --model mujoco/g0.xml \
  --policy /home/lz/g0_robot_lab/g0_robot_lab/logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/model_9999.pt \
  --steps 1000 \
  --command 0.0 0.0 0.0 \
  --device cpu \
  --record-rollout logs/sim2sim/mujoco_rollout.npz
```

ONNX rollout is intentionally left as TODO.

## Unitree-Style Deploy Rollout

Export G0 deploy metadata from Isaac Lab:

```bash
TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p scripts/sim2sim/export_g0_deploy_cfg.py \
  --task G0-Velocity-v0 \
  --output logs/sim2sim/g0_deploy/params/deploy.yaml \
  --headless
```

Run zero-action deploy rollout in the current position-actuator backend:

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

Check it:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/check_g0_deploy_rollout.py \
  --rollout logs/sim2sim/g0_deploy/mujoco_deploy_zero_action_position_rollout.npz \
  --deploy-cfg logs/sim2sim/g0_deploy/params/deploy.yaml \
  --output logs/sim2sim/g0_deploy/deploy_zero_action_position_check.md
```

Run the first torque-backend pass:

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

`pd_torque` mode generates `mujoco/g0_pd_torque.xml` and writes:

```text
tau_cmd = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
```

with `tau_ff = 0` and effort-limit clipping. The formal `mujoco/g0.xml` and foot mesh collision are not modified.

Run the deploy validation matrix:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/run_g0_deploy_validation_matrix.py \
  --model mujoco/g0.xml \
  --deploy-cfg logs/sim2sim/g0_deploy/params/deploy.yaml \
  --policy /home/lz/g0_robot_lab/g0_robot_lab/logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/model_9999.pt \
  --steps 200 \
  --output-dir logs/sim2sim/g0_deploy/validation_matrix
```

The matrix writes:

```text
logs/sim2sim/g0_deploy/validation_matrix/<case>.npz
logs/sim2sim/g0_deploy/validation_matrix/<case>_check.md
logs/sim2sim/g0_deploy/validation_matrix/<case>_run.log
logs/sim2sim/g0_deploy/validation_matrix/validation_matrix_summary.md
docs/g0_deploy_sim2sim_validation_matrix_report.md
```

The current 16-case matrix result is:

```text
500-step matrix: 16/16 runner OK
500-step matrix: 16/16 checker OK
velocity-limit exceeded: 0/16 cases
zero-action cases: early-fall heuristic true
some policy cases: early-fall heuristic true at 500 steps
current status: started, not smoke pass, not credible pass
```

Run controlled root-frame diagnostics:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/diagnose_root_frame_conventions.py

TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p scripts/sim2sim/dump_isaac_controlled_root_state.py \
  --task G0-Velocity-v0 \
  --output logs/sim2sim/root_frame/isaac_controlled_root_state.npz \
  --headless

TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/dump_mujoco_controlled_root_state.py \
  --model mujoco/g0.xml \
  --output logs/sim2sim/root_frame/mujoco_controlled_root_state.npz

TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/compare_controlled_root_frame.py \
  --isaac logs/sim2sim/root_frame/isaac_controlled_root_state.npz \
  --mujoco logs/sim2sim/root_frame/mujoco_controlled_root_state.npz \
  --output logs/sim2sim/root_frame/controlled_root_frame_report.md
```

Current controlled root-frame result:

```text
quaternion order: wxyz
projected_gravity aligned: True
base_ang_vel aligned: True
```

Analyze 500-step policy failure windows:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/analyze_g0_deploy_failure_windows.py \
  --matrix-dir logs/sim2sim/g0_deploy/validation_matrix_500 \
  --deploy-cfg logs/sim2sim/g0_deploy/params/deploy.yaml \
  --output logs/sim2sim/g0_deploy/failure_window_analysis.md
```

Outputs:

```text
logs/sim2sim/g0_deploy/failure_window_analysis.md
docs/g0_policy_failure_window_analysis.md
```

Current result:

```text
policy cases analyzed: 8
failed: 5
stable: 3
most common precursor: action saturation first
velocity limit: not a primary suspect
torque saturation: not a primary suspect
```

## Compare Rollouts

```bash
python scripts/sim2sim/compare_isaac_mujoco_rollout.py \
  --isaac logs/sim2sim/isaac_golden_io.npz \
  --mujoco logs/sim2sim/mujoco_rollout.npz \
  --output logs/sim2sim/compare_report.md
```

First zero-action validation command set:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/play_mujoco_g0.py \
  --model mujoco/g0.xml \
  --steps 100 \
  --command 0.0 0.0 0.0 \
  --zero-action \
  --record-rollout logs/sim2sim/mujoco_zero_action_rollout.npz

TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p scripts/sim2sim/dump_isaac_golden_io.py \
  --task G0-Velocity-v0 \
  --steps 100 \
  --num_envs 1 \
  --output logs/sim2sim/isaac_zero_action_golden_io.npz \
  --zero-action \
  --headless

TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/compare_isaac_mujoco_rollout.py \
  --isaac logs/sim2sim/isaac_zero_action_golden_io.npz \
  --mujoco logs/sim2sim/mujoco_zero_action_rollout.npz \
  --output logs/sim2sim/compare_zero_action_report.md
```

First-frame term report:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/compare_first_frame_observation.py \
  --isaac logs/sim2sim/isaac_zero_action_golden_io.npz \
  --mujoco logs/sim2sim/mujoco_zero_action_rollout.npz \
  --output logs/sim2sim/first_frame_observation_report.md
```

Expected zero-action interface result: `action` and `target_joint_pos` compare with zero error. Joint/root state differences are expected while `mujoco/g0.xml` remains an uncalibrated URDF-derived working model.

After the first actuator alignment pass, the latest 100-step zero-action rollout did not reproduce the previous MuJoCo `QACC` warning. This is a useful regression signal, not proof that dynamics are final.

Velocity-limit diagnostic:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/check_mujoco_velocity_limits.py \
  --rollout logs/sim2sim/mujoco_zero_action_rollout.npz \
  --output logs/sim2sim/mujoco_velocity_limit_report.md
```

Current zero-action result: `0/22` joints exceeded Isaac `velocity_limit_sim`; worst ratio was about `0.0967`.

Zero-action dynamics diagnostic:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/compare_zero_action_dynamics.py \
  --isaac logs/sim2sim/isaac_zero_action_golden_io.npz \
  --mujoco logs/sim2sim/mujoco_zero_action_rollout.npz \
  --output logs/sim2sim/zero_action_dynamics_compare_report.md
```

The MuJoCo rollout now records `qacc`, policy-order `joint_acc`, `contact_count`, `max_contact_force_norm`, `foot_ground_contact_count`, and `root_height`. These are diagnostics only and do not change control behavior.

It also records:

```text
left_foot_contact_force_norm
right_foot_contact_force_norm
total_foot_contact_force_norm
foot_contact_force_norm
```

## Current TODOs

- Calibrate the URDF-derived `mujoco/g0.xml` into a full dynamics model.
- Verify projected gravity frame against Isaac Lab.
- Verify base angular velocity frame and scaling against Isaac Lab.
- Validate actuator velocity-limit semantics.
- Align contact solver parameters, friction, and mass/inertia.
- Regenerate Isaac golden I/O with optional acceleration/contact diagnostics before making contact-force conclusions.
