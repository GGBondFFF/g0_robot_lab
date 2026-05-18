# Pre-Deployment Validation and Test Strategy Design

## Context

Branch: `validation/isaac-lowcmd-dryrun`

Repository: `GGBondFFF/g0_robot_lab`

This branch is intended as deployment preparation before real G0 robot work. It is intentionally identical to `main` at the start of this design pass. The repository currently has no `tests/` tree, no `conftest.py`, and no low-level hardware command implementation in the active codebase.

The active Isaac Lab task is `G0-Velocity-v0`. The active robot config is `G0_CFG`. The main train/play entry points are:

- `scripts/rsl_rl/train.py`
- `scripts/rsl_rl/play.py`

The current deployment-sensitive anchors are:

- `source/g0_robot_lab/g0_robot_lab/assets/robots/g0/g0.py`
- `source/g0_robot_lab/g0_robot_lab/assets/robots/g0/g0_actuators.py`
- `source/g0_robot_lab/g0_robot_lab/tasks/locomotion/robots/g0/velocity_env_cfg.py`
- `docs/observation_action_interface.md`
- `docs/g0_actuator_parameters.md`
- `docs/sim2sim_isaaclab_to_mujoco.md`

Current known contract values:

- task id: `G0-Velocity-v0`
- action joint order: `G0_JOINT_SDK_NAMES`
- action dimension: `22`
- action scale: `0.12`
- action offset convention: `target_joint_pos = default_joint_pos + action_scale * policy_action`
- action config: `use_default_offset=True`, `preserve_order=True`
- policy observation terms: `base_ang_vel`, `projected_gravity`, `velocity_commands`, `joint_pos_rel`, `joint_vel_rel`, `last_action`, `gait_phase`
- policy observation history length: `5`
- simulation timing: `sim.dt = 0.005`, `decimation = 4`, control dt `0.02`
- right-angle servo ratio assumption: speed `6/7`, torque `7/6`

## Goals

Create a validation strategy that can catch deployment-breaking mismatches before any real robot command path is used.

The strategy must cover:

1. Pure Python and unit tests that run without Isaac Sim when possible.
2. Isaac Lab headless smoke and integration tests.
3. Policy observation/action interface contract tests.
4. G0 joint order, default pose, action scale, and `preserve_order` checks.
5. Actuator parameter and right-angle servo ratio checks.
6. Sim2sim/deployment dry-run checks for low-level command generation.
7. Safety guardrails that prevent real hardware commands during tests.
8. Documentation updates needed for this validation branch.

## Non-Goals

This design does not implement tests yet.

This design does not add MuJoCo sim2sim code to `main`.

This design does not require real hardware, Unitree SDK connectivity, serial devices, UDP sockets, or motor power.

This design does not make current zero-action standing a default CI blocker while the documented baseline still fails before 500 steps. It defines that check as a deployment gate and diagnostic test tier.

## Testing Architecture Options

### Option A: Isaac-First End-to-End Suite

Most validation would run through Isaac Lab headless environments. Contract checks would instantiate `G0-Velocity-v0`, inspect the managers, run zero-action or policy rollouts, and infer the interface from runtime behavior.

Benefits:

- High confidence that the instantiated Isaac Lab task behaves as expected.
- Catches registry, environment construction, action manager, sensor, and observation manager issues together.
- Closest to training/play reality.

Tradeoffs:

- Slow and GPU/display/runtime dependent.
- Hard to run in normal local development or lightweight CI.
- Failures can be noisy because import, simulator startup, assets, physics, and policy contracts are coupled.
- Poor fit for simple constants such as right-angle servo ratios or joint list partitions.

### Option B: Pure-Unit Static Contract Suite

Most validation would avoid Isaac Lab entirely. Tests would parse Python files, inspect constants, and verify documentation snapshots without launching the simulator.

Benefits:

- Fast, deterministic, and suitable for default CI.
- Can run on machines without Isaac Sim.
- Good at catching accidental edits to joint order, default pose, action scale, actuator ratios, and docs drift.

Tradeoffs:

- Static parsing can miss runtime behavior, especially Isaac Lab manager resolution.
- It does not prove that `preserve_order=True` survives through the action manager at runtime.
- It does not prove that the gym task registers, resets, or steps.
- It can become brittle if tests overfit source formatting instead of public constants.

### Option C: Layered Contract Pyramid

Use a three-tier suite:

- `unit`: fast, no Isaac Sim app launch, focused on pure contracts and static safety checks.
- `isaaclab`: headless smoke/integration tests that launch Isaac Lab only where runtime behavior matters.
- `deployment_dryrun`: offline low-level command tests using fake policy outputs and fake transports; never sends hardware commands.

Benefits:

- Keeps default validation fast while still covering runtime Isaac behavior.
- Separates deployment contracts from physics stability diagnostics.
- Makes safety guardrails testable without real hardware.
- Supports incremental adoption: unit tests first, headless smoke tests second, dry-run deployment tests third.

Tradeoffs:

- Requires clear pytest markers and developer discipline.
- Some constants may need to be moved into a pure module later if direct imports still pull in Isaac Lab.
- More initial test harness work than a single end-to-end script.

## Recommendation

Use Option C: Layered Contract Pyramid.

Default local and CI validation should run `unit` tests without Isaac Sim. These tests should fail quickly on contract drift in joint order, default pose, actuator ratios, action scale, and safety assumptions.

Isaac Lab headless tests should run separately with an explicit marker and command because they require the Isaac Lab Python environment and simulator startup. These tests should cover task registration, environment reset, observation/action spaces, action manager joint order, and minimal stepping.

Deployment dry-run tests should remain offline by design. They should validate the low-level command generation path using fake policies, fake robot state, fake clocks, and fake transports. Any real hardware send path must be unreachable unless explicitly enabled outside pytest.

## Proposed Test Layout

Future implementation should add:

```text
tests/
├── conftest.py
├── unit/
│   ├── test_g0_joint_contract.py
│   ├── test_g0_default_pose_contract.py
│   ├── test_g0_actuator_contract.py
│   ├── test_velocity_env_static_contract.py
│   └── test_docs_contract.py
├── isaaclab/
│   ├── test_task_registration_headless.py
│   ├── test_env_reset_and_spaces_headless.py
│   ├── test_action_manager_order_headless.py
│   └── test_zero_action_smoke_headless.py
└── deployment/
    ├── test_policy_io_contract.py
    ├── test_lowcmd_generation_dryrun.py
    ├── test_lowcmd_transport_guardrails.py
    └── test_sim2sim_interface_snapshot.py
```

Pytest markers should include:

```text
unit
isaaclab
deployment_dryrun
slow
release_gate
hardware_forbidden
```

Recommended commands:

```bash
python -m pytest tests/unit -m "unit"
```

```bash
/home/lz/IsaacLab/isaaclab.sh -p -m pytest tests/isaaclab -m "isaaclab"
```

```bash
python -m pytest tests/deployment -m "deployment_dryrun and hardware_forbidden"
```

Release-gate command:

```bash
/home/lz/IsaacLab/isaaclab.sh -p -m pytest tests -m "isaaclab or deployment_dryrun or release_gate"
```

## Pure Python Unit Tests

Pure tests should run without launching `isaaclab.app.AppLauncher`.

If importing `g0_robot_lab.assets.robots.g0.g0` requires Isaac Lab, the implementation should choose one of these approaches:

- Prefer extracting deployment-sensitive constants into a pure module such as `g0_robot_lab/assets/robots/g0/g0_contract.py`.
- Otherwise use AST-based tests for the first validation pass, then refactor to a pure module in a later step.

Unit tests should verify:

- `G0_JOINT_SDK_NAMES` has exactly 22 names.
- `G0_JOINT_SDK_NAMES` contains no duplicates.
- `G0_DEFAULT_JOINT_POS` covers every SDK joint exactly once.
- `G0_STANDARD_SERVO_JOINT_NAMES` and `G0_RIGHT_ANGLE_SERVO_JOINT_NAMES` partition the SDK joint set.
- right-angle servo joints are exactly the documented six joints unless a deliberate hardware correction is made.
- standard servo count is 16 and right-angle servo count is 6.
- default pose keeps the documented mirrored pitch signs:
  - left hip pitch negative, right hip pitch positive
  - left knee pitch negative, right knee pitch positive
  - left ankle pitch positive, right ankle pitch negative
  - left elbow pitch positive, right elbow pitch negative
- `STANDARD_SERVO_MAX_VELOCITY == STANDARD_SERVO_MAX_RPM * 2*pi / 60`
- `RIGHT_ANGLE_SERVO_RATED_TORQUE == STANDARD_SERVO_RATED_TORQUE * 7/6 * RIGHT_ANGLE_GEAR_EFFICIENCY`
- `RIGHT_ANGLE_SERVO_MAX_VELOCITY == STANDARD_SERVO_MAX_VELOCITY * 6/7`
- `RIGHT_ANGLE_SERVO_ARMATURE == STANDARD_SERVO_ARMATURE * (7/6)^2`
- `ActionsCfg.joint_pos.scale == 0.12`
- `ActionsCfg.joint_pos.use_default_offset is True`
- `ActionsCfg.joint_pos.preserve_order is True`
- `ActionsCfg.joint_pos.joint_names == list(G0_JOINT_SDK_NAMES)`

## Policy Observation and Action Contract Tests

The policy interface is deployment-critical and should be validated as a contract, not as an incidental detail.

Static unit tests should verify the configured policy observation term order:

```text
base_ang_vel
projected_gravity
velocity_commands
joint_pos_rel
joint_vel_rel
last_action
gait_phase
```

Expected per-frame policy dimensions:

```text
base_ang_vel: 3
projected_gravity: 3
velocity_commands: 3
joint_pos_rel: 22
joint_vel_rel: 22
last_action: 22
gait_phase: 2
total per frame: 77
history_length: 5
expected flattened policy observation dimension: 385
```

Expected action contract:

```text
action_dim: 22
joint_order: G0_JOINT_SDK_NAMES
scale: 0.12
offset: G0_DEFAULT_JOINT_POS in runtime joint order
formula: target_joint_pos = default_joint_pos + 0.12 * policy_action
```

Isaac Lab headless tests should instantiate one environment and verify runtime values:

- `env.action_space.shape == (22,)` for one environment after wrapper normalization, or equivalent vectorized shape before wrapper.
- action manager total action dimension is 22.
- resolved action joint order equals `G0_JOINT_SDK_NAMES`.
- policy observation group reports the expected flattened dimension.
- a zero action produces targets equal to default joint positions.
- a one-hot action at index `i` affects only `G0_JOINT_SDK_NAMES[i]` by `0.12`.

## Isaac Lab Headless Smoke and Integration Tests

Headless tests should be explicit and minimal.

Pytest commands should not depend on forwarding `--headless` through pytest. Instead, the Isaac Lab test fixture should create the simulator with an explicit headless launcher configuration, such as `AppLauncher({"headless": True})`, and should close the app after the test session.

Required smoke tests:

- `test_task_registration_headless`: launch Isaac Lab headless, import `g0_robot_lab.tasks`, assert `G0-Velocity-v0` is registered.
- `test_env_reset_and_spaces_headless`: construct `G0RobotLabEnvCfg` with `num_envs=1`, call `gym.make`, reset, inspect observation/action spaces, close env.
- `test_action_manager_order_headless`: reuse the logic from `scripts/debug/debug_runtime_joint_order.py` in a pytest-friendly helper and assert action order equals `G0_JOINT_SDK_NAMES`.
- `test_zero_action_smoke_headless`: disable randomization, use one env, step zero action for a small number of steps such as 10, and assert no exception, finite observations, finite root pose, and finite joint state.

Release-gate Isaac tests:

- `test_zero_action_standing_release_gate`: run the fixed-condition zero-action standing check for 500 steps with nominal effort limits.
- `test_train_one_iteration_headless`: run `scripts/rsl_rl/train.py --task G0-Velocity-v0 --num_envs 1 --max_iterations 1 --headless`.
- `test_play_export_contract_headless`: when a checkpoint is available, run `play.py` headless long enough to export `policy.pt` and `policy.onnx`, then validate policy input/output shapes.

Current baseline note:

The documented zero-action standing result does not yet pass 500 steps. Therefore the 500-step zero-action check should be marked `release_gate` and not included in default CI until the physical baseline is fixed.

## Sim2sim and Deployment Dry-Run Tests

The repository currently documents future MuJoCo sim2sim work but does not include `scripts/sim2sim/` or `mujoco/`. The dry-run strategy should prepare for deployment without adding real hardware sends.

Future deployment code should separate:

- policy loading
- observation construction
- action scaling and clipping
- low-level command frame generation
- transport/publisher side effects

Dry-run tests should use fake inputs and assert outputs:

- `test_policy_io_contract`: feed a fake or exported policy with a 385-dimensional observation and assert a 22-dimensional action.
- `test_lowcmd_generation_zero_action`: generate a low-level command from zero action and assert target positions equal default pose.
- `test_lowcmd_generation_one_hot_action`: generate one-hot actions and assert only the matching SDK joint target changes by `0.12`.
- `test_lowcmd_generation_clipping`: assert policy actions are clipped or rejected according to the selected deployment convention.
- `test_lowcmd_joint_order_snapshot`: assert low-level command joint order equals `G0_JOINT_SDK_NAMES`.
- `test_sim2sim_interface_snapshot`: assert sim2sim observation order, history length, command vector order, gait phase convention, and control dt match Isaac Lab docs.

The low-level command generator should be testable as a pure function:

```text
lowcmd = build_lowcmd(
    policy_action,
    default_joint_pos,
    joint_order=G0_JOINT_SDK_NAMES,
    action_scale=0.12,
    dry_run=True,
)
```

The dry-run object should include enough inspectable data for tests:

```text
joint_name
target_position
kp
kd
torque_ff
source_action_index
source_action_value
```

## Hardware Safety Guardrails

Tests must never send commands to real hardware.

Required guardrails:

- All deployment tests default to dry-run mode.
- Any real hardware transport must require an explicit runtime opt-in such as `G0_ALLOW_HARDWARE=1`.
- Pytest should set `G0_ALLOW_HARDWARE=0`.
- Pytest fixtures should monkeypatch or block known transport side effects.
- Tests should fail if a hardware send function is called.
- Real hardware tests should not exist in this repository until a separate hardware bring-up procedure is approved.

Suggested blocked operations in tests:

```text
socket.socket(...).send
socket.socket(...).sendto
serial.Serial(...).write
Unitree SDK publisher write/send methods
filesystem writes to /dev/tty*
network sends to robot control ports
```

Recommended fixtures:

- `forbid_hardware_transports`: monkeypatch socket, serial, and Unitree SDK send paths to raise `AssertionError`.
- `dryrun_required`: assert `G0_ALLOW_HARDWARE` is unset or set to `0`.
- `fake_lowcmd_transport`: captures command frames in memory for assertions.

No test command should require robot power, robot network, motor enable, or a connected controller.

## Documentation Updates

After implementation, update or add:

- `README.md`: add a short validation section with default unit and Isaac headless commands.
- `docs/run_commands.md`: add test commands and explain marker tiers.
- `docs/observation_action_interface.md`: add the test-enforced policy/action contract and expected dimensions.
- `docs/g0_actuator_parameters.md`: state that actuator constants are covered by unit tests.
- `docs/sim2sim_isaaclab_to_mujoco.md`: add the dry-run interface contract and clarify that sim2sim code remains separate from `main` unless intentionally added on this validation branch.
- `docs/pre_deployment_validation.md`: summarize the full validation workflow, release gates, and hardware safety rules.

## Acceptance Criteria

Default unit validation passes without Isaac Sim:

```bash
python -m pytest tests/unit -m "unit"
```

Isaac Lab smoke validation passes in the Isaac Lab environment:

```bash
/home/lz/IsaacLab/isaaclab.sh -p -m pytest tests/isaaclab -m "isaaclab"
```

Deployment dry-run validation passes without hardware:

```bash
python -m pytest tests/deployment -m "deployment_dryrun and hardware_forbidden"
```

The test suite fails if:

- action joint order drifts away from `G0_JOINT_SDK_NAMES`
- action scale changes without test updates
- `preserve_order=True` is removed
- default pose no longer covers the SDK joints
- standard/right-angle servo groups overlap or miss a joint
- right-angle servo ratios change without an intentional test update
- low-level command generation can send through a real transport under pytest
- policy observation/action dimensions drift without an intentional contract update

## Open Implementation Notes

The first implementation step should decide whether to extract pure constants into a small contract module or use AST-based tests initially. Extraction is cleaner long-term, but AST-based tests can provide immediate protection without changing implementation behavior.

The current debug scripts contain useful inspection logic, but pytest tests should factor shared helpers instead of importing scripts that launch `AppLauncher` at module import time.

Headless tests must always close the simulation app and environment, even on assertion failure.

Release-gate tests should be documented separately from default CI because zero-action standing is still an active physical consistency issue.
