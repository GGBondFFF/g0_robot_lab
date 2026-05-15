# G0 Sim2sim Scripts

This directory contains the Isaac Lab to MuJoCo sim2sim validation scaffold for `G0-Velocity-v0`.

## Files

- `g0_sim2sim_config.py`: shared G0 joint order, default pose, action scale, timing, and action bridge helpers.
- `g0_mujoco_interface.py`: MuJoCo model loader, joint/action index bridge, action application, and observation skeleton.
- `play_mujoco_g0.py`: MuJoCo rollout runner for zero-action or exported TorchScript policy.
- `dump_isaac_golden_io.py`: Isaac Lab golden I/O dumper. Must be launched through Isaac Lab.
- `compare_isaac_mujoco_rollout.py`: compares Isaac and MuJoCo `.npz` rollouts and writes a Markdown report.
- `compare_first_frame_observation.py`: compares first-frame observation terms instead of only flattened rollout arrays.
- `inspect_g0_urdf_for_mujoco.py`: read-only URDF readiness inspection for MuJoCo migration.
- `inspect_g0_usd_collision.py`: USD/PhysX foot collision inspection. Must be launched through Isaac Lab.
- `inspect_mujoco_collision.py`: MuJoCo foot geom and initial contact inspection.
- `generate_g0_mujoco_from_urdf.py`: generates the current URDF-derived `mujoco/g0.xml` working model.
- `export_g0_actuator_alignment_table.py`: writes an Isaac-vs-MuJoCo actuator parameter table.
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

Policy dump using exported TorchScript `policy.pt`:

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/sim2sim/dump_isaac_golden_io.py \
  --task G0-Velocity-v0 \
  --checkpoint logs/rsl_rl/g0_velocity/<run>/exported/policy.pt \
  --steps 100 \
  --num_envs 1 \
  --output logs/sim2sim/isaac_golden_io.npz \
  --headless
```

Raw RSL-RL checkpoints should first be exported with `scripts/rsl_rl/play.py`.

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

TorchScript policy rollout:

```bash
python scripts/sim2sim/play_mujoco_g0.py \
  --model mujoco/g0.xml \
  --policy logs/rsl_rl/g0_velocity/<run>/exported/policy.pt \
  --steps 1000 \
  --command 0.0 0.0 0.0 \
  --device cpu \
  --record-rollout logs/sim2sim/mujoco_rollout.npz
```

ONNX rollout is intentionally left as TODO.

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

## Current TODOs

- Calibrate the URDF-derived `mujoco/g0.xml` into a full dynamics model.
- Verify projected gravity frame against Isaac Lab.
- Verify base angular velocity frame and scaling against Isaac Lab.
- Validate actuator velocity-limit semantics.
- Align contact solver parameters, friction, and mass/inertia.
