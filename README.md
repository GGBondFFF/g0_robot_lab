# g0_robot_lab

`g0_robot_lab` is a G0 humanoid locomotion project built on Isaac Lab.

The current main task is:

```text
G0-Velocity-v0
```

The project can already run train and play inside Isaac Lab. The current work on `main` is a protected baseline for G0 locomotion debugging: actuator parameters, zero-action standing, reward/termination/reset tuning, and preparation for later Isaac Lab to MuJoCo sim2sim work.

Do not mix this project with `humanoid_lab_v0`, and do not move the current task back to the old `manager_based` template layout. The active task tree is `tasks/locomotion/`.

## Current Status

- Project: `g0_robot_lab`
- Main task: `G0-Velocity-v0`
- Simulator: Isaac Lab
- Robot config: `G0_CFG`
- Train entry point: `scripts/rsl_rl/train.py`
- Play entry point: `scripts/rsl_rl/play.py`
- Current baseline: train/play runnable, with zero-action standing and locomotion stability still under active debug

This repository should keep the current `main` branch runnable. Large training runs, MuJoCo sim2sim scaffolding, and non-documentation refactors should happen on later branches.

## Directory Overview

```text
g0_robot_lab/
├── README.md
├── docs/
├── scripts/
│   └── rsl_rl/
│       ├── train.py
│       └── play.py
└── source/
    └── g0_robot_lab/
        └── g0_robot_lab/
            ├── assets/
            │   └── robots/
            │       └── g0/
            │           ├── g0.py
            │           ├── g0_actuators.py
            │           ├── urdf/
            │           ├── usd/
            │           └── meshes/
            └── tasks/
                └── locomotion/
                    ├── __init__.py
                    ├── agents/
                    ├── mdp/
                    └── robots/
                        └── g0/
                            └── velocity_env_cfg.py
```

Key paths:

- `source/g0_robot_lab/g0_robot_lab/tasks/locomotion/`: current locomotion task package
- `source/g0_robot_lab/g0_robot_lab/tasks/locomotion/__init__.py`: Gym task registration for `G0-Velocity-v0`
- `source/g0_robot_lab/g0_robot_lab/tasks/locomotion/robots/g0/velocity_env_cfg.py`: G0 velocity environment config
- `source/g0_robot_lab/g0_robot_lab/assets/robots/g0/g0.py`: `G0_CFG`, joint names, default pose, actuator groups
- `source/g0_robot_lab/g0_robot_lab/assets/robots/g0/g0_actuators.py`: current actuator hardware constants and debug baseline values

## Install

Install Isaac Lab first, then install this package in editable mode from the repository root:

```bash
cd /home/lz/g0_robot_lab/g0_robot_lab
/home/lz/IsaacLab/isaaclab.sh -p -m pip install -e source/g0_robot_lab
```

## Common Commands

All simulation-related commands must run inside the `g0_isaaclab` conda environment. This includes Isaac Lab, `AppLauncher`, `SimulationApp`, `gym.make` for `G0-Velocity-v0`, `pytest tests/isaaclab`, and any command that imports `isaaclab`, `pxr`, `omni`, or runtime task registration.

Activate the environment first:

```bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate g0_isaaclab
```

List or check the registered G0 task:

```bash
/home/lz/IsaacLab/isaaclab.sh -p -c "
from isaaclab.app import AppLauncher
app_launcher = AppLauncher({'headless': True})
simulation_app = app_launcher.app

import gymnasium as gym
import g0_robot_lab
import g0_robot_lab.tasks

print([env_id for env_id in gym.registry.keys() if 'G0' in env_id])
simulation_app.close()
"
```

Smoke test:

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task G0-Velocity-v0 \
  --num_envs 1 \
  --max_iterations 1
```

Small headless test:

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task G0-Velocity-v0 \
  --num_envs 32 \
  --max_iterations 100 \
  --headless
```

Play a checkpoint:

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task G0-Velocity-v0 \
  --num_envs 32 \
  --checkpoint <checkpoint-path>
```

More command details are in [docs/run_commands.md](docs/run_commands.md).

## Validation Tiers

Run the fast static unit tier without Isaac Sim:

```bash
python -m pytest tests/unit -m "unit"
```

Run the offline deployment dry-run tier with fake LowCmd and fake transport checks:

```bash
python -m pytest tests/deployment -m "deployment_dryrun and hardware_forbidden"
```

Run the Isaac Lab headless smoke tier inside `g0_isaaclab`:

```bash
/home/lz/IsaacLab/isaaclab.sh -p -m pytest tests/isaaclab -m "isaaclab"
```

Run explicit release gates before deployment work:

```bash
/home/lz/IsaacLab/isaaclab.sh -p -m pytest tests -m "release_gate"
```

The Isaac Lab smoke tier uses one combined runtime smoke test so it does not create multiple `gym.make`/`env.close` cycles in a single `SimulationApp` session. Release gates are selected separately with `-m "release_gate"` and are not part of the default smoke tier.

## Current Debug Focus

The current baseline is not considered a final walking policy. Before large-scale training, keep checking:

- zero-action standing
- default joint pose and root initial height
- reward, termination, and reset behavior
- foot collision/contact patch
- COM, mass, and inertia consistency
- action scale and observation/action ordering for later sim2sim

See:

- [docs/project_structure.md](docs/project_structure.md)
- [docs/observation_action_interface.md](docs/observation_action_interface.md)
- [docs/g0_actuator_parameters.md](docs/g0_actuator_parameters.md)
- [docs/zero_action_standing_debug.md](docs/zero_action_standing_debug.md)
- [docs/sim2sim_isaaclab_to_mujoco.md](docs/sim2sim_isaaclab_to_mujoco.md)
- [docs/pre_deployment_validation.md](docs/pre_deployment_validation.md)
