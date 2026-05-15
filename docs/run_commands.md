# Run Commands

This document records the current common commands for `g0_robot_lab`.

Project root:

```bash
cd /home/lz/g0_robot_lab/g0_robot_lab
```

Current task:

```text
G0-Velocity-v0
```

Isaac Lab launcher:

```text
/home/lz/IsaacLab/isaaclab.sh -p
```

## Install Editable Package

```bash
cd /home/lz/g0_robot_lab/g0_robot_lab
/home/lz/IsaacLab/isaaclab.sh -p -m pip install -e source/g0_robot_lab
```

## List Or Check Task Registration

```bash
/home/lz/IsaacLab/isaaclab.sh -p -c "
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({'headless': True})
simulation_app = app_launcher.app

import gymnasium as gym
import g0_robot_lab
import g0_robot_lab.tasks

matched = [env_id for env_id in gym.registry.keys() if 'G0' in env_id or 'g0' in env_id]
print('Matched env ids:')
for env_id in matched:
    print(env_id)

simulation_app.close()
"
```

Expected task id:

```text
G0-Velocity-v0
```

## Smoke Test

Use one environment and one training iteration to check the train path without doing meaningful training:

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task G0-Velocity-v0 \
  --num_envs 1 \
  --max_iterations 1
```

## Small Headless Test

Use this for a small baseline run after config changes:

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task G0-Velocity-v0 \
  --num_envs 32 \
  --max_iterations 100 \
  --headless
```

This is still a debug-scale test, not a large training run.

## Play Checkpoint

Template:

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task G0-Velocity-v0 \
  --num_envs 32 \
  --checkpoint <checkpoint-path>
```

Example using the latest saved checkpoint:

```bash
CKPT=$(find logs/rsl_rl/g0_velocity -name "model_*.pt" | sort | tail -1)

/home/lz/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task G0-Velocity-v0 \
  --num_envs 32 \
  --checkpoint "$CKPT"
```

Headless play with video:

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task G0-Velocity-v0 \
  --num_envs 1 \
  --checkpoint "$CKPT" \
  --headless \
  --video \
  --video_length 100
```

`play.py` is also the current export path for `policy.pt` and `policy.onnx`.

## Zero-Action And Debug Commands

These commands are documented for debugging only. They are not a request to change the scripts in this documentation pass.

Fixed zero-action standing trace:

```bash
TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p scripts/debug/sweep_g0_standing_pose.py \
  --task G0-Velocity-v0 \
  --headless \
  --mode fixed \
  --steps 500 \
  --print-torque-every 10 \
  --effort-scale 1.0
```

Effort-scale diagnosis:

```bash
for scale in 1.0 1.2 1.5 2.0; do
  TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p scripts/debug/sweep_g0_standing_pose.py \
    --task G0-Velocity-v0 \
    --headless \
    --mode fixed \
    --steps 500 \
    --print-torque-every 10 \
    --effort-scale "$scale"
done
```

Initial geometry inspection:

```bash
TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p scripts/debug/inspect_g0_initial_geometry.py \
  --task G0-Velocity-v0 \
  --headless \
  --root_z 0.23 \
  --hip 0.20 \
  --knee 0.34 \
  --ankle 0.14
```

## Guardrails

- Do not run large training jobs from `main` while zero-action standing is still being debugged.
- Do not mix in `humanoid_lab_v0` commands or checkpoints.
- Do not create `scripts/sim2sim/` or `mujoco/` on `main`; keep sim2sim implementation for a later branch.
