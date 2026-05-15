# G0 Sim2sim Scripts

This directory contains the Isaac Lab to MuJoCo sim2sim validation scaffold for `G0-Velocity-v0`.

## Files

- `g0_sim2sim_config.py`: shared G0 joint order, default pose, action scale, timing, and action bridge helpers.
- `g0_mujoco_interface.py`: MuJoCo model loader, joint/action index bridge, action application, and observation skeleton.
- `play_mujoco_g0.py`: MuJoCo rollout runner for zero-action or exported TorchScript policy.
- `dump_isaac_golden_io.py`: Isaac Lab golden I/O dumper. Must be launched through Isaac Lab.
- `compare_isaac_mujoco_rollout.py`: compares Isaac and MuJoCo `.npz` rollouts and writes a Markdown report.
- `validate_sim2sim_setup.py`: quick structure and interface validator.

## Validate Setup

```bash
python scripts/sim2sim/validate_sim2sim_setup.py
```

If ordinary Python cannot import the project package, run through Isaac Lab:

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/sim2sim/validate_sim2sim_setup.py
```

If MuJoCo is not installed, the XML load check is skipped with a warning.

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

## Current TODOs

- Replace placeholder `mujoco/g0.xml` with a full URDF/MJCF-derived model.
- Verify projected gravity frame against Isaac Lab.
- Verify base angular velocity frame and scaling against Isaac Lab.
- Align actuator PD, torque limits, velocity limits, contact, and friction.

