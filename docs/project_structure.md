# g0_robot_lab Project Structure

This document records the current `g0_robot_lab` structure on `main`.

The active project is `g0_robot_lab`, not `humanoid_lab_v0`. The active task is `G0-Velocity-v0`. Do not move current work back into the old Isaac Lab template `tasks/manager_based/` structure.

## Current Main Paths

```text
g0_robot_lab/
├── scripts/
│   └── rsl_rl/
│       ├── train.py
│       └── play.py
├── source/
│   └── g0_robot_lab/
│       └── g0_robot_lab/
│           ├── assets/
│           │   └── robots/
│           │       └── g0/
│           │           ├── __init__.py
│           │           ├── g0.py
│           │           ├── g0_actuators.py
│           │           ├── urdf/
│           │           ├── usd/
│           │           └── meshes/
│           └── tasks/
│               └── locomotion/
│                   ├── __init__.py
│                   ├── agents/
│                   │   ├── __init__.py
│                   │   └── rsl_rl_ppo_cfg.py
│                   ├── mdp/
│                   │   ├── __init__.py
│                   │   ├── curriculums.py
│                   │   ├── events.py
│                   │   ├── observations.py
│                   │   ├── rewards.py
│                   │   └── commands/
│                   └── robots/
│                       └── g0/
│                           ├── __init__.py
│                           └── velocity_env_cfg.py
└── docs/
```

## Task Registration

Current task id:

```text
G0-Velocity-v0
```

Registration location:

```text
source/g0_robot_lab/g0_robot_lab/tasks/locomotion/__init__.py
```

Registered entries:

```text
env_cfg_entry_point:
g0_robot_lab.tasks.locomotion.robots.g0.velocity_env_cfg:G0RobotLabEnvCfg

play_env_cfg_entry_point:
g0_robot_lab.tasks.locomotion.robots.g0.velocity_env_cfg:G0RobotLabPlayEnvCfg

rsl_rl_cfg_entry_point:
g0_robot_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:PPORunnerCfg
```

The environment entry point still uses Isaac Lab's `ManagerBasedRLEnv`, but the project should not use the old local `tasks/manager_based/` template directory as the current task layout.

## Robot Asset Package

Current robot asset directory:

```text
source/g0_robot_lab/g0_robot_lab/assets/robots/g0/
```

Important files:

- `g0.py`: defines `G0_CFG`, joint name lists, default standing pose, actuator groups, and articulation settings.
- `g0_actuators.py`: defines current actuator constants for standard and right-angle servos.
- `urdf/g0.urdf`: G0 URDF source.
- `usd/g0.usd`: Isaac Lab USD asset used by `G0_CFG`.
- `meshes/`: STL meshes referenced by the robot asset.

Current robot configuration:

```text
G0_CFG
```

## Locomotion Task Package

Current main task directory:

```text
source/g0_robot_lab/g0_robot_lab/tasks/locomotion/
```

Important subdirectories:

- `agents/`: RSL-RL PPO runner config.
- `mdp/`: task MDP functions, rewards, events, observations, command helpers, and curriculum helpers.
- `robots/g0/`: G0-specific environment config.

Current environment config:

```text
source/g0_robot_lab/g0_robot_lab/tasks/locomotion/robots/g0/velocity_env_cfg.py
```

This file is now the active G0 velocity locomotion config. It should not be described as still being a Cartpole template.

## Action Interface

Current action config is in `ActionsCfg`:

```text
mdp.JointPositionActionCfg
joint_names=list(G0_JOINT_SDK_NAMES)
scale=0.12
use_default_offset=True
preserve_order=True
```

That means policy actions are ordered by:

```text
G0_JOINT_SDK_NAMES
```

and target joint positions are computed as:

```text
target_joint_pos = default_joint_pos + 0.12 * policy_action
```

The `preserve_order=True` setting is important for Isaac Lab to keep the SDK/deployment-friendly joint order instead of reordering names internally.

## Train And Play Entrypoints

Training:

```text
scripts/rsl_rl/train.py
```

Playback and export:

```text
scripts/rsl_rl/play.py
```

`play.py` can export `policy.pt` and `policy.onnx` under the selected RSL-RL log directory. Later sim2sim work should reuse that export path, but the MuJoCo code framework should be created on a separate branch.

## Current Main-Branch Rule

Keep `main` as the runnable Isaac Lab baseline:

- Do not mix in `humanoid_lab_v0`.
- Do not restore the old template `tasks/manager_based/` task layout.
- Do not add MuJoCo sim2sim code or placeholder non-Markdown files on `main`.
- Keep `G0-Velocity-v0` train/play runnable while actuator, standing, reward, termination, and reset debug continues.
