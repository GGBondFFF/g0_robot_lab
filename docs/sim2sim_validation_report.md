# G0 MuJoCo Sim2sim Validation Report

## Context

- Branch: `structure/mujoco-sim2sim-layout`
- Task: `G0-Velocity-v0`
- Test time: `2026-05-15 13:44:12 CST`
- Goal: validate the Isaac Lab to MuJoCo sim2sim pipeline, not MuJoCo walking quality.

## Commands Run

Framework tests:

```bash
TERM=xterm conda run -n g0_isaaclab python -m pytest tests/sim2sim -q
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/validate_sim2sim_setup.py
TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p -m pytest tests/sim2sim -q
```

MuJoCo API diagnosis:

```bash
TERM=xterm conda run -n g0_isaaclab python -c "import mujoco; print('mujoco file:', getattr(mujoco, '__file__', None)); print('has MjModel:', hasattr(mujoco, 'MjModel')); print('has MjData:', hasattr(mujoco, 'MjData'))"
TERM=xterm conda run -n g0_isaaclab python -m pip show mujoco || true
TERM=xterm conda run -n g0_isaaclab python -m pip list | grep -i mujoco || true
find . -maxdepth 3 \( -iname "mujoco.py" -o -iname "mujoco" \)
```

MuJoCo package install:

```bash
TERM=xterm conda run -n g0_isaaclab python -m pip install -U mujoco
```

Zero-action MuJoCo rollout:

```bash
mkdir -p logs/sim2sim
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/play_mujoco_g0.py \
  --model mujoco/g0.xml \
  --steps 100 \
  --command 0.0 0.0 0.0 \
  --zero-action \
  --record-rollout logs/sim2sim/mujoco_zero_action_rollout.npz
```

Isaac Lab golden I/O:

```bash
TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p scripts/sim2sim/dump_isaac_golden_io.py \
  --task G0-Velocity-v0 \
  --steps 100 \
  --num_envs 1 \
  --output logs/sim2sim/isaac_zero_action_golden_io.npz \
  --zero-action \
  --headless
```

Compare report:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/compare_isaac_mujoco_rollout.py \
  --isaac logs/sim2sim/isaac_zero_action_golden_io.npz \
  --mujoco logs/sim2sim/mujoco_zero_action_rollout.npz \
  --output logs/sim2sim/compare_zero_action_report.md
```

Policy rollout:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/play_mujoco_g0.py \
  --model mujoco/g0.xml \
  --policy /home/lz/g0_robot_lab/g0_robot_lab/logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/model_9999.pt \
  --steps 100 \
  --command 0.0 0.0 0.0 \
  --device cpu \
  --record-rollout logs/sim2sim/mujoco_policy_rollout.npz
```

## Test Results

`pytest` with conda Python:

```text
9 passed, 1 skipped in 0.95s
```

`pytest` through Isaac Lab launcher:

```text
9 passed, 1 skipped in 0.98s
```

The skipped test was the optional MuJoCo XML load test before the correct DeepMind MuJoCo package was installed. After installation, `validate_sim2sim_setup.py` performed the real XML load check.

## MuJoCo Python API Diagnosis

Before installing DeepMind MuJoCo, `import mujoco` resolved to a namespace module from the local `./mujoco` directory:

```text
mujoco file: None
has MjModel: False
has MjData: False
pip show mujoco: package not found
find result: ./mujoco
```

This was an environment/package issue, not a sim2sim code logic issue.

After installing `mujoco==3.8.1`:

```text
mujoco file: /home/lz/miniconda3/envs/g0_isaaclab/lib/python3.11/site-packages/mujoco/__init__.py
MjModel: <class 'mujoco._structs.MjModel'>
MjData: <class 'mujoco._structs.MjData'>
```

## MuJoCo XML Load Result

Initial load failed with:

```text
Error: more than 6 dofs in body 'base_link'
```

Cause: the first placeholder XML placed a freejoint and all 22 hinge joints on the same MuJoCo body. MuJoCo does not allow that body DOF layout.

Fix: `mujoco/g0.xml` was minimally changed into a nested placeholder body chain. This is only to make the interface scaffold loadable. It is not a real G0 kinematic/dynamics model.

Final validation:

```text
OK: scripts/sim2sim exists
OK: mujoco/g0.xml exists
OK: joint/action dimension is 22
OK: default_joint_pos has shape (22,)
OK: action scale is 0.12
OK: control_dt is 0.02
OK: MuJoCo model has all 22 joints
Summary: OK
```

## Zero-Action MuJoCo Rollout

Output:

```text
logs/sim2sim/mujoco_zero_action_rollout.npz
```

Saved keys and important shapes:

```text
action: (100, 22)
target_joint_pos: (100, 22)
joint_pos: (100, 22)
joint_vel: (100, 22)
root_pos: (100, 3)
root_quat: (100, 4)
obs: (100, 385)
joint_names: (22,)
default_joint_pos: (22,)
action_scale: scalar
control_dt: scalar
```

Zero-action bridge check:

```text
target_equals_default: True
max_target_default_abs_err: 0.0
```

This confirms the current MuJoCo action bridge implements:

```text
target_joint_pos = default_joint_pos + 0.12 * clipped_policy_action
```

## Isaac Lab Golden I/O

Output:

```text
logs/sim2sim/isaac_zero_action_golden_io.npz
```

Saved keys and important shapes:

```text
obs: (100, 385)
action: (100, 22)
target_joint_pos: (100, 22)
joint_pos: (100, 22)
joint_vel: (100, 22)
root_pos: (100, 3)
root_quat: (100, 4)
base_ang_vel: (100, 3)
projected_gravity: (100, 3)
command: (100, 3)
default_joint_pos: (22,)
joint_names: (22,)
action_scale: scalar
sim_dt: scalar
decimation: scalar
control_dt: scalar
```

Isaac Lab warnings were ordinary runtime warnings and did not block export.

## Compare Report

Output:

```text
logs/sim2sim/compare_zero_action_report.md
```

Main result:

```text
action mean/max abs error: 0 / 0
target_joint_pos mean/max abs error: 0 / 0
joint_pos mean/max abs error: 0.483795 / 1.36278
joint_vel mean/max abs error: 0.449632 / 42.6161
command mean/max abs error: 0.0391715 / 0.0723783
root_height mean/max abs error: 0.143505 / 0.143896
```

Interpretation:

- `action` and `target_joint_pos` are aligned for zero-action.
- `joint_pos`, `joint_vel`, and `root_height` differences are expected because `mujoco/g0.xml` is still a placeholder dynamics model.
- `command` differs because Isaac Lab's command generator is active even in zero-action policy mode, while MuJoCo rollout used an explicit zero command.
- Missing keys are expected: MuJoCo rollout does not yet store `base_ang_vel`, `projected_gravity`, or `decimation`; Isaac does not store `time`.

## Policy Rollout

Pinned checkpoint used for checkpoint identity audits:

```text
/home/lz/g0_robot_lab/g0_robot_lab/logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/model_9999.pt
```

Output:

```text
logs/sim2sim/mujoco_policy_rollout.npz
```

Result:

```text
Finished 100 MuJoCo control steps.
```

This only validates that the TorchScript policy can run through the MuJoCo scaffold. It does not validate walking quality because dynamics/contact/observation frame details are not fully aligned.

## Confirmed Aligned

## Deploy-Style Sim2sim Follow-Up

This pass added a Unitree-style deploy entrypoint while preserving the existing direct MuJoCo runner.

Commands run:

```bash
TERM=xterm conda run -n g0_isaaclab python -m pytest tests/sim2sim -q
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/validate_sim2sim_setup.py --model mujoco/g0.xml
TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p scripts/sim2sim/export_g0_deploy_cfg.py \
  --task G0-Velocity-v0 \
  --output logs/sim2sim/g0_deploy/params/deploy.yaml \
  --headless
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/run_g0_mujoco_deploy.py \
  --model mujoco/g0.xml \
  --deploy-cfg logs/sim2sim/g0_deploy/params/deploy.yaml \
  --steps 200 \
  --command 0.0 0.0 0.0 \
  --zero-action \
  --control-mode position \
  --record-rollout logs/sim2sim/g0_deploy/mujoco_deploy_zero_action_position_rollout.npz
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/check_g0_deploy_rollout.py \
  --rollout logs/sim2sim/g0_deploy/mujoco_deploy_zero_action_position_rollout.npz \
  --deploy-cfg logs/sim2sim/g0_deploy/params/deploy.yaml \
  --output logs/sim2sim/g0_deploy/deploy_zero_action_position_check.md
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/run_g0_mujoco_deploy.py \
  --model mujoco/g0.xml \
  --deploy-cfg logs/sim2sim/g0_deploy/params/deploy.yaml \
  --steps 200 \
  --command 0.0 0.0 0.0 \
  --zero-action \
  --control-mode pd_torque \
  --record-rollout logs/sim2sim/g0_deploy/mujoco_deploy_zero_action_pd_torque_rollout.npz
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/check_g0_deploy_rollout.py \
  --rollout logs/sim2sim/g0_deploy/mujoco_deploy_zero_action_pd_torque_rollout.npz \
  --deploy-cfg logs/sim2sim/g0_deploy/params/deploy.yaml \
  --output logs/sim2sim/g0_deploy/deploy_zero_action_pd_torque_check.md
```

Policy rollout commands also completed for:

```text
/home/lz/g0_robot_lab/g0_robot_lab/logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/model_9999.pt
```

Results:

```text
pytest: 11 passed
validate_sim2sim_setup.py: Summary OK
deploy.yaml: exported, obs_dim=385, joints=22
zero-action position rollout: completed 200 steps, check OK
zero-action pd_torque rollout: completed 200 steps, check OK
policy position rollout: completed 200 steps, check OK
policy pd_torque rollout: completed 200 steps, check OK
```

The deploy runner now records the LowCmd-style fields:

```text
pd_q_des
pd_dq_des
pd_kp
pd_kd
pd_tau_ff
pd_tau_cmd
pd_tau_cmd_clipped
```

Control equation:

```text
tau_cmd = tau_ff + kp * (q_des - q) + kd * (dq_des - dq)
```

with `tau_ff = 0`. Raw `tau_cmd` can exceed `effort_limit_sim`; `pd_tau_cmd_clipped` is clipped and checked against the effort limits.

This is now a deploy-style MuJoCo sim2sim validation entrypoint. It is still not a Unitree DDS/C++ ONNX deployment stack.

## Deploy Validation Matrix

The first deploy-style validation matrix has been added and run.

Command:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/run_g0_deploy_validation_matrix.py \
  --model mujoco/g0.xml \
  --deploy-cfg logs/sim2sim/g0_deploy/params/deploy.yaml \
  --policy /home/lz/g0_robot_lab/g0_robot_lab/logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/model_9999.pt \
  --steps 200 \
  --output-dir logs/sim2sim/g0_deploy/validation_matrix
```

Result:

```text
500-step validation matrix OK: 16/16 cases OK
```

Coverage:

```text
action kind: zero-action, policy
control mode: position, pd_torque
commands:
  [0.0, 0.0, 0.0]
  [0.05, 0.0, 0.0]
  [0.1, 0.0, 0.0]
  [0.0, 0.0, 0.1]
```

500-step summary:

- All 16 rollouts completed and passed `check_g0_deploy_rollout.py`.
- Checker now reports action saturation, torque saturation, velocity-limit exceedance, root height min/max/final, finite root/quaternion/gravity diagnostics, contact count, foot force ranges, and early-fall heuristic.
- Zero-action cases all fell below the `root_height < 0.12` heuristic. This is a default-pose/contact/dynamics signal, not a policy failure.
- Several policy cases also fell below the `root_height < 0.12` heuristic at 500 steps after the corrected `base_ang_vel` convention.
- Policy action saturation is present. The highest 500-step matrix value was `0.4053` in `policy_position_c0_c0_c0`.
- Raw PD torque saturation is present but below the 5% smoke threshold. The highest 500-step matrix value was `0.01655` in `policy_position_c0p05_c0_c0`.
- No case exceeded `velocity_limit_sim`.
- Foot contact force maxima ranged from about `13.6` in zero-action cases to about `22.1` in policy cases.
- Current status: `sim2sim started`.
- Current status: not smoke pass, because policy root-height and action-saturation criteria are not met.
- Current status: not credible pass, because 1000-step stability and deeper actuator/contact explanations are still missing.

Controlled root-frame diagnostics:

```text
quaternion order: wxyz
projected_gravity max abs error: 2.32911e-08
base_ang_vel max abs error: 5.5907e-08
root_height max abs error: 4.17233e-09
```

The controlled diagnostics found and fixed a MuJoCo-side `base_ang_vel` frame mismatch. `projected_gravity` did not require a formula change.

Primary report:

```text
docs/g0_deploy_sim2sim_validation_matrix_report.md
docs/root_frame_convention_diagnostics.md
docs/g0_sim2sim_credibility_criteria.md
docs/zero_action_collapse_analysis.md
```

Policy failure-window analysis:

```text
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/analyze_g0_deploy_failure_windows.py \
  --matrix-dir logs/sim2sim/g0_deploy/validation_matrix_500 \
  --deploy-cfg logs/sim2sim/g0_deploy/params/deploy.yaml \
  --output logs/sim2sim/g0_deploy/failure_window_analysis.md
```

Result:

```text
Analyzed 8 policy cases
failed: 5
stable: 3
most common pre-failure signal: action saturation first
```

Failed policy cases:

```text
policy_position_c0_c0_c0: failure step 324
policy_position_c0p05_c0_c0: failure step 355
policy_pd_torque_c0_c0_c0: failure step 421
policy_pd_torque_c0_c0_c0p1: failure step 414
policy_pd_torque_c0p05_c0_c0: failure step 449
```

Stable policy cases:

```text
policy_position_c0p1_c0_c0
policy_position_c0_c0_c0p1
policy_pd_torque_c0p1_c0_c0
```

Interpretation:

- Action saturation is the leading current suspect.
- Torque saturation is not the main suspect.
- Velocity limits are not the main suspect.
- Contact loss is not the earliest precursor in failed policy cases.
- Suspicious joints are concentrated around `r_knee_pitch_joint`, `l_hip_pitch_joint`, `r_hip_pitch_joint`, `l_ankle_pitch_joint`, and `l_knee_pitch_joint`.

Generated artifacts:

```text
logs/sim2sim/g0_deploy/validation_matrix/*.npz
logs/sim2sim/g0_deploy/validation_matrix/*_check.md
logs/sim2sim/g0_deploy/validation_matrix/*_run.log
logs/sim2sim/g0_deploy/validation_matrix/validation_matrix_summary.md
```

- sim2sim framework files are present.
- `G0_JOINT_SDK_NAMES` dimension is 22.
- default joint position dimension is 22.
- action scale is `0.12`.
- control dt is `0.02`.
- MuJoCo XML can be loaded by DeepMind MuJoCo Python API.
- MuJoCo model exposes all 22 expected joint names.
- MuJoCo actuator index mapping can be built.
- zero-action target joint positions equal `G0_DEFAULT_JOINT_POS`.
- Isaac golden I/O export works.
- MuJoCo rollout export works.
- compare report generation works.
- TorchScript policy rollout path works.

## Not Yet Aligned

- `mujoco/g0.xml` is not a complete robot model.
- MuJoCo body hierarchy is now URDF-derived, but still not validated against Isaac Lab USD.
- mass and inertia are now URDF-derived, but still need hardware and Isaac Lab sanity checks.
- actuator PD is placeholder.
- torque limits are placeholder.
- velocity limits are placeholder.
- foot contact geometry is placeholder.
- friction/contact settings are placeholder.
- `projected_gravity` is not yet precisely matched in MuJoCo.
- `base_ang_vel` frame is not yet precisely matched in MuJoCo.
- Isaac command generator behavior is not yet mirrored in MuJoCo.

## Next TODO

1. Calibrate the URDF-derived `mujoco/g0.xml` into a stable dynamics model.
2. Preserve exact joint names and actuator names while replacing the physical model.
3. Store MuJoCo `base_ang_vel`, `projected_gravity`, and `decimation` in rollout files.
4. Add a deterministic zero-command mode for Isaac golden I/O or mirror Isaac command sampling in MuJoCo.
5. Compare first-frame observations term by term.
6. Align actuator PD, torque limits, velocity limits, friction, and foot contact geometry.
7. Only after interface and dynamics alignment, interpret policy rollout quality.

## URDF-Derived Model Migration Update

Second validation time:

```text
2026-05-15 14:21:31 CST
```

The previous loadable placeholder model is preserved as:

```text
mujoco/g0_interface_placeholder.xml
```

The current `mujoco/g0.xml` is now generated from the source URDF with:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/generate_g0_mujoco_from_urdf.py
```

The generated working model keeps the URDF-derived body tree, mesh geoms, inertials, joint axes, and joint limits, then adds a floating root, ground plane, and 22 policy position actuators.

URDF inspection passed:

```text
link_count: 23
joint_count: 22
movable_joint_count: 22
mesh_count: 46
missing_policy_joints: []
extra_movable_joints: []
missing_meshes: []
links_missing_inertial: []
movable_without_limit: []
```

Latest validation:

```text
pytest: 11 passed
validate mujoco/g0.xml: Summary OK, all 22 joints found
validate mujoco/g0_interface_placeholder.xml: Summary OK, all 22 joints found
```

Latest MuJoCo zero-action rollout:

```text
Saved MuJoCo rollout: logs/sim2sim/mujoco_zero_action_rollout.npz
Finished 100 MuJoCo control steps.
WARNING: Nan, Inf or huge value in QACC at DOF 0. The simulation is unstable. Time = 0.0600.
```

This warning is a dynamics/contact/actuator calibration issue in the new URDF-derived working model. It is not interpreted as policy failure.

The zero-action action bridge remains exact:

```text
target_joint_pos == default_joint_pos
max_target_default_abs_err: 0.0
```

Latest short rollout comparison:

```text
action mean/max abs error: 0 / 0
target_joint_pos mean/max abs error: 0 / 0
joint_pos mean/max abs error: 111.666 / 16816.7
joint_vel mean/max abs error: 18139.1 / 1.04043e+06
root_height mean/max abs error: 0.00533956 / 0.09883
```

Large joint position/velocity errors are expected while the URDF-derived MuJoCo model is dynamically unstable.

First-frame observation term report:

```text
joint_names: ok
default_joint_pos: 0 / 0
action: 0 / 0
target_joint_pos: 0 / 0
obs_shape: 0 / 0
obs_last_action: 0 / 0
projected_gravity: about 0.00206 / 0.00349
base_ang_vel: about 5.22 / 9.73
```

The MuJoCo observation builder was changed from frame-major history to term-major history to better match Isaac Lab. `projected_gravity` and `base_ang_vel` are now recorded in MuJoCo rollout files, but their frame conventions still require controlled-orientation diagnostics.

## Collision Filtering Update

The formal `mujoco/g0.xml` now uses Isaac-style robot self-collision filtering generated by:

```text
scripts/sim2sim/generate_g0_mujoco_from_urdf.py
```

Filtering:

```text
ground: contype=1, conaffinity=2
robot:  contype=2, conaffinity=1
```

This preserves robot-ground contact and disables robot-robot internal contacts under MuJoCo's contact rule:

```text
(contype1 & conaffinity2) || (contype2 & conaffinity1)
```

Inspection after regeneration:

```text
foot_ground_contacts: 3
non_ground_self_contacts: 0
base_link_torso_link_contact: False
isaac_style_self_collision_disabled: True
```

Foot mesh collision remains unchanged:

```text
l_foot_link: mesh geom using l_foot_link.STL
r_foot_link: mesh geom using r_foot_link.STL
```

No foot box was added.

Zero-action rollout still reports a QACC instability warning:

```text
WARNING: Nan, Inf or huge value in QACC at DOF 3. The simulation is unstable. Time = 0.0450.
```

Since internal self-collision is now filtered out, the next debugging target should be actuator PD/limits and initial foot-ground penetration/contact solver settings.

## Actuator Alignment Validation Update

Test time: `2026-05-15 15:48:50 CST`

The generator now maps Isaac `G0_CFG` actuator parameters into `mujoco/g0.xml`:

```text
stiffness -> position actuator kp
effort_limit_sim -> position actuator forcerange
damping -> joint damping, first-pass approximation
armature -> joint armature
```

Validation commands:

```bash
TERM=xterm conda run -n g0_isaaclab python -m pytest tests/sim2sim -q
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/generate_g0_mujoco_from_urdf.py
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/validate_sim2sim_setup.py --model mujoco/g0.xml
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/export_g0_actuator_alignment_table.py
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/inspect_mujoco_collision.py --model mujoco/g0.xml --steps 1
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/play_mujoco_g0.py \
  --model mujoco/g0.xml \
  --steps 100 \
  --command 0.0 0.0 0.0 \
  --zero-action \
  --record-rollout logs/sim2sim/mujoco_zero_action_rollout.npz
```

Results:

```text
pytest: 11 passed
validate: Summary OK
actuator alignment table: 22/22 rows aligned
foot_ground_contacts: 3
non_ground_self_contacts: 0
base_link_torso_link_contact: False
zero-action rollout: saved successfully
QACC warning: not observed in this 100-step zero-action run
```

Zero-action rollout check:

```text
action shape: (100, 22)
target_joint_pos shape: (100, 22)
joint_pos shape: (100, 22)
target_equals_default: True
max_target_default_abs_err: 0.0
max_abs_action: 0.0
has_nan_inf_joint_pos: False
```

Latest compare against the existing Isaac golden I/O:

```text
action mean/max abs error: 0 / 0
target_joint_pos mean/max abs error: 0 / 0
joint_pos mean/max abs error: 0.0100061 / 0.0607303
joint_vel mean/max abs error: 0.0523146 / 3.18014
root_height mean/max abs error: 0.18667 / 0.192377
```

Foot collision geometry was not modified. The formal model still uses the URDF-derived foot mesh geoms and no formal foot box/capsule/sphere was added.

Remaining non-equivalent items:

- `velocity_limit_sim` is documented but not yet enforced with a verified MuJoCo equivalent.
- Isaac implicit actuator damping is only approximately represented by MJCF joint damping.
- Contact solver/friction, mass/inertia, root initialization, `projected_gravity`, and `base_ang_vel` still need separate alignment passes.

## Velocity-Limit And Zero-Action Dynamics Diagnostics

Test time: `2026-05-15 15:48:50 CST`

This pass added diagnostics only. No formal foot collision geometry, actuator gains, damping, friction, or solver parameters were tuned.

Commands:

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/check_mujoco_velocity_limits.py \
  --rollout logs/sim2sim/mujoco_zero_action_rollout.npz \
  --output logs/sim2sim/mujoco_velocity_limit_report.md

TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/compare_zero_action_dynamics.py \
  --isaac logs/sim2sim/isaac_zero_action_golden_io.npz \
  --mujoco logs/sim2sim/mujoco_zero_action_rollout.npz \
  --output logs/sim2sim/zero_action_dynamics_compare_report.md
```

Velocity-limit result:

```text
exceeded joints: 0/22
worst ratio: 0.0967143 on l_hip_pitch_joint
```

Zero-action dynamics result:

```text
target_joint_pos mean/max abs error: 0 / 0
joint_pos mean/max abs error: 0.0100061 / 0.0607303
joint_vel mean/max abs error: 0.0523146 / 3.18014
root_quat mean/max abs error: 0.237017 / 0.696053
base_ang_vel mean/max abs error: 0.214718 / 4.77339
projected_gravity mean/max abs error: 0.240835 / 0.999848
root_height mean/max abs error: 0.0723206 / 0.192692
```

MuJoCo-side diagnostic ranges:

```text
qacc min/mean/max: -160.078 / -0.168248 / 358.986
joint_acc min/mean/max: -160.078 / -0.219228 / 164.065
contact_count min/mean/max: 0 / 3.37 / 9
max_contact_force_norm min/mean/max: 0 / 10.0759 / 66.815
foot_ground_contact_count min/mean/max: 0 / 2.23 / 5
```

No QACC warning was observed in this 100-step zero-action rollout. `qacc` and `joint_acc` were finite.

Interpretation:

- Velocity limits are not being hit in zero-action.
- The largest remaining differences are root/frame-sensitive terms: `root_quat`, `projected_gravity`, `base_ang_vel`, and `root_height`.
- Contact and acceleration terms are currently MuJoCo-side only in the files compared here. Isaac golden export now attempts optional `root_height`, `joint_acc`, and contact-force fields, but the existing golden file should be regenerated if cross-simulator contact/acceleration comparison is needed.

## Deterministic Zero-Action Golden I/O Update

Test time: `2026-05-15 17:13 CST`

The Isaac golden dumper now supports deterministic zero-action export without modifying `velocity_env_cfg.py`:

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

In-memory deterministic overrides:

```text
reset_base pose x/y/yaw: fixed to 0
reset_base velocity: all zero
reset_robot_joints velocity: zero
command range: fixed to 0 0 0
physics material randomization: fixed ranges
base mass randomization: zero delta
push_robot interval event: disabled
policy observation corruption: disabled
seed: 0
```

The deterministic file successfully exports:

```text
root_height
joint_acc
contact_force
foot_contact_force
left_foot_contact_force
right_foot_contact_force
foot_contact_force_norm
left_foot_contact_force_norm
right_foot_contact_force_norm
```

Command is now exactly aligned:

```text
command mean/max abs error: 0 / 0
```

Deterministic dynamics compare:

```text
joint_pos mean/max abs error: 0.0108194 / 0.074113
joint_vel mean/max abs error: 0.0563463 / 3.12355
root_quat mean/max abs error: 0.0902682 / 0.694091
base_ang_vel mean/max abs error: 0.207238 / 4.84762
projected_gravity mean/max abs error: 0.242127 / 0.999301
root_height mean/max abs error: 0.072338 / 0.198734
joint_acc mean/max abs error: 4.02243 / 165.262
foot_contact_force_norm mean/max abs error: 4.28525 / 9.19268
```

MuJoCo zero-action rollout still completed without a QACC warning in this pass.

Interpretation:

- Command sampling is no longer a source of mismatch in this deterministic report.
- Remaining differences are dominated by root orientation/frame terms, root height/contact settling, and acceleration/contact-force behavior.
- This is still a simulator/model alignment report, not a policy-quality report.
