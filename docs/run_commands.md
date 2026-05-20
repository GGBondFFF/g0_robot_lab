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

All simulation-related commands must run inside the `g0_isaaclab` conda environment. This includes Isaac Lab, `AppLauncher`, `SimulationApp`, `gym.make` for `G0-Velocity-v0`, `pytest tests/isaaclab`, and commands that import `isaaclab`, `pxr`, `omni`, or runtime task registration.

Use this shell setup before Isaac Lab or simulation commands:

```bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate g0_isaaclab
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

## Validation Commands

Default static unit tier:

```bash
python -m pytest tests/unit -m "unit"
```

Deployment dry-run tier:

```bash
python -m pytest tests/deployment -m "deployment_dryrun and hardware_forbidden"
```

Offline LowCmd mapping validation:

```bash
python scripts/validation/validate_g0_lowcmd_mapping.py \
  --mode offline-contract \
  --emit-json logs/validation/lowcmd_mapping_offline.json
```

This Phase C validator is offline only. It does not import Isaac, does not start `AppLauncher`, does not send real LowCmd, does not connect to hardware, and does not indicate real-robot readiness.

Isaac Lab headless smoke tier:

```bash
/home/lz/IsaacLab/isaaclab.sh -p -m pytest tests/isaaclab -m "isaaclab"
```

Release-gate tier:

```bash
/home/lz/IsaacLab/isaaclab.sh -p -m pytest tests -m "release_gate"
```

Policy rollout safety release gate:

```bash
/home/lz/IsaacLab/isaaclab.sh -p -m pytest -q \
  tests/isaaclab/test_release_gate_policy_rollout_safety.py \
  -m "release_gate"
```

The Isaac Lab smoke tier is selected by `pytest.mark.isaaclab` and uses a combined runtime smoke test to avoid repeated `gym.make`/`env.close` cycles in one `SimulationApp` session. It does not include release-gate tests.

Release gates are explicit deployment-readiness checks. The policy export release gate passed in the current implementation. The zero-action 500-step release gate is explicitly selectable with `-m "release_gate"` and may report a physical-readiness failure; that result is a deployment readiness signal, not a default smoke failure.

The deployment dry-run tier uses fake LowCmd objects and fake transports only. Hardware transport blocking is marker-scoped, so unit and Isaac Lab tests are not affected by a global socket monkeypatch.

The offline LowCmd mapping validator also uses dry-run-only fake commands and the pure-Python safety/mapping core only. It must not be interpreted as permission to enter any real motor command path.

## Policy Rollout Safety Validation (Isaac Lab only)

This diagnostic runs Isaac Lab policy rollout validation only. It does not send LowCmd or touch real hardware. The separate pytest release gate covers the conservative 500-step hard checks, while raw action clipping, effort, joint margin, and target-delta signals remain diagnostic here.

These commands must be run from the repository root with the `g0_isaaclab` conda environment active, or through the Isaac Lab runtime wrapper.

500 steps:

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/validation/validate_g0_policy_rollout_in_isaac.py \
  --task G0-Velocity-v0 \
  --checkpoint logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/model_9999.pt \
  --headless \
  --steps 500 \
  --num-envs 1
```

1000 steps:

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/validation/validate_g0_policy_rollout_in_isaac.py \
  --task G0-Velocity-v0 \
  --checkpoint logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/model_9999.pt \
  --headless \
  --steps 1000 \
  --num-envs 1
```

2000 steps:

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/validation/validate_g0_policy_rollout_in_isaac.py \
  --task G0-Velocity-v0 \
  --checkpoint logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/model_9999.pt \
  --headless \
  --steps 2000 \
  --num-envs 1
```

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
- Do not send real LowCmd or motor commands from tests.
- Real hardware command paths require a separately approved bring-up procedure.
