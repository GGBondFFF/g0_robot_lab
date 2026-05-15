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
  --policy logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/exported/policy.pt \
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

Existing exported policy used:

```text
logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/exported/policy.pt
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
