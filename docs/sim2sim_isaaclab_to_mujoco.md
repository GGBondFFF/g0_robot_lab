# Sim2sim Plan: Isaac Lab To MuJoCo

This document records the current sim2sim plan only. It does not add a MuJoCo code framework on `main`.

Current validation tests treat sim2sim references as interface snapshots only. This branch must not restore `mujoco/` or `scripts/sim2sim/`.

## Current Main-Branch Scope

The current `main` branch keeps the Isaac Lab `G0-Velocity-v0` baseline runnable.

Do not create these implementation directories on `main` during documentation-only work:

```text
scripts/sim2sim/
mujoco/
```

MuJoCo sim2sim should be implemented later on a new branch.

## Expected Future Structure

Later, a sim2sim branch may add:

```text
scripts/sim2sim/
mujoco/
```

Possible responsibilities:

- `scripts/sim2sim/`: policy loading, observation construction, action application, rollout scripts, and comparison tools.
- `mujoco/`: MuJoCo model assets, XML files, actuator settings, contact/friction settings, and sim2sim-specific notes.

This document is only a planning note for that future branch.

The deployment dry-run tier includes snapshot checks that these implementation directories remain absent while the documentation still records the interface anchors needed for future sim2sim work:

```bash
python -m pytest tests/deployment -m "deployment_dryrun and hardware_forbidden"
```

## Policy Export

The current Isaac Lab `play.py` path can export:

```text
policy.pt
policy.onnx
```

Future sim2sim work should reuse those exported policies instead of inventing a separate export path first.

The raw RSL-RL checkpoint such as `model_9999.pt` is not the deployment inference artifact. The deployment artifacts are:

```text
exported/policy.pt
exported/policy.onnx
```

The release-gate policy artifact contract checks the exported policy IO shape `385 -> 22`; ONNX validation requires `onnxruntime` installed in `g0_isaaclab`.

## Critical Alignment Items

Sim2sim is not just exporting a policy. The hard part is aligning the policy interface and dynamics assumptions.

The following must match between Isaac Lab, MuJoCo, and eventual hardware deployment:

- joint order
- default joint pose
- action scale
- action offset convention
- observation order
- observation history length
- projected gravity convention
- base angular velocity frame and scale
- command vector order and scale
- gait phase convention, if used by policy
- control dt
- decimation
- actuator PD gains
- torque limits
- velocity limits
- contact geometry
- contact patch
- friction
- mass and inertial parameters

## Current Isaac Lab Interface Anchors

Current task id:

```text
G0-Velocity-v0
```

Current action order:

```text
G0_JOINT_SDK_NAMES
```

Current action scale:

```text
0.12
```

Current target position convention:

```text
target_joint_pos = default_joint_pos + action_scale * policy_action
```

Current policy observation terms:

```text
base_ang_vel
projected_gravity
velocity_commands
joint_pos_rel
joint_vel_rel
last_action
gait_phase
```

Current policy history length:

```text
5
```

Current simulation timing in the Isaac Lab config:

```text
sim.dt = 0.005
decimation = 4
control dt = 0.02
```

## Recommended Future Workflow

1. Keep Isaac Lab train/play baseline runnable on `main`.
2. Create a dedicated sim2sim branch.
3. Export `policy.pt` or `policy.onnx` from `play.py`.
4. Build a MuJoCo model with matching joint names, default pose, actuator limits, and contact settings.
5. Reconstruct observations in the exact Isaac Lab policy order.
6. Apply actions using the same joint order and target position formula.
7. Compare short rollouts from identical initial states before attempting long motion tests.
