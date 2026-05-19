# Pre-Deployment Validation

This document summarizes the validation workflow for the G0 deployment-preparation branch. All simulation-related commands must run inside the `g0_isaaclab` conda environment.

Activate the environment before Isaac Lab or runtime task-registration commands:

```bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate g0_isaaclab
```

## Default Unit Tier

Fast static tests that do not launch Isaac Sim:

```bash
python -m pytest tests/unit -m "unit"
```

This tier checks G0 joint order, default pose, actuator ratio contracts, velocity environment action/observation contracts, policy dimensions, and documentation anchors.

## Deployment Dry-Run Tier

Offline deployment and LowCmd dry-run tests:

```bash
python -m pytest tests/deployment -m "deployment_dryrun and hardware_forbidden"
```

This tier uses fake LowCmd objects and fake transports only. Hardware transport blocking is marker-scoped, not global, so unit and Isaac Lab tests still see the normal socket class. No test should send real LowCmd or motor commands.

## Isaac Lab Headless Tier

Headless Isaac Lab smoke tests:

```bash
/home/lz/IsaacLab/isaaclab.sh -p -m pytest tests/isaaclab -m "isaaclab"
```

The smoke tier uses a combined runtime smoke test to avoid repeated `gym.make`/`env.close` cycles in one `SimulationApp` session. It validates task registration, one-env reset, finite observations/state, action dimension, and action manager order. Release-gate tests are not selected by this marker.

## Release Gate Tier

Explicit deployment-readiness gates:

```bash
/home/lz/IsaacLab/isaaclab.sh -p -m pytest tests -m "release_gate"
```

The policy export release gate passed in the current implementation. It runs `scripts/rsl_rl/play.py` with `G0_ALLOW_HARDWARE=0` in the subprocess environment and verifies that deployment inference artifacts can be exported from the raw RSL-RL checkpoint when the checkpoint exists.

The zero-action 500-step release gate is explicitly selectable with `-m "release_gate"`. It may report a physical-readiness failure on the current baseline; that is a deployment readiness signal, not a default smoke failure.

## Policy Artifacts

`model_9999.pt` is a raw RSL-RL training checkpoint. It is not the final deployment inference artifact.

Deployment inference artifacts are:

```text
exported/policy.pt
exported/policy.onnx
```

The exported policy IO contract is:

```text
observation dim 385 -> action dim 22
```

TorchScript validation uses `policy.pt`. ONNX validation uses `policy.onnx` and requires `onnxruntime` installed in `g0_isaaclab`.

## Hardware Safety

Validation tests must not connect to real hardware and must not send real LowCmd or motor commands. Real hardware command paths require a separately approved bring-up procedure.

## Sim2sim Scope

Sim2sim references are interface snapshots only on this branch. Do not restore or create:

```text
mujoco/
scripts/sim2sim/
```
