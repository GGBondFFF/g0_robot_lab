# LowCmd Mapping Safety Audit

## Purpose

This document records the Phase A through Phase E audit status for the deployment-facing G0 dry-run mapping chain on branch `validation/lowcmd-mapping-safety-audit`.

The audited chain is:

`raw policy action -> clipped action [-1, 1] -> target joint position -> safety filter -> FakeLowCmd dry-run command`

The purpose of this work is to validate the software-side contract for a future deployment path without entering any real hardware path.

## Scope

Included in scope:

- joint order and mapping-table audit
- pure-Python safety filter and LowCmd-style dry-run mapping core
- offline contract validation
- Isaac Lab policy sampling through the dry-run mapping chain
- deployment, unit, Isaac non-release, and Isaac release-gate regression checks

Out of scope:

- real LowCmd transmission
- real hardware communication
- motor power or real command timing
- hardware-side motor ID confirmation
- hardware-side direction sign confirmation
- real emergency-stop validation on hardware
- MuJoCo sim2sim restoration or continuation
- reward, PPO, checkpoint, or robot-asset changes

## No-Hardware Warning

This audit does not authorize real robot deployment.

This audit does not authorize LowCmd transmission.

This audit does not confirm motor IDs on hardware.

This audit does not confirm joint direction signs on hardware.

This audit does not validate real communication timing.

This audit does not validate emergency stop on real hardware.

Passing Phase E still does not indicate real-robot readiness.

The next stage after Phase E is hardware-free communication rehearsal, not real motor command.

## Relation To Policy Rollout Safety Validation

The earlier Isaac Lab policy rollout safety validation established that the fixed checkpoint can satisfy the current simulation-only rollout contract, but also showed that raw policy actions can exceed `[-1, 1]`.

That earlier result is the reason this audit treats clipping as mandatory before any LowCmd-compatible representation is produced. The current audit therefore focuses on proving the software contract from clipped policy output into a fake LowCmd dry-run command, while explicitly preventing any hardware path.

## Mapping Table Status

The current 22-row mapping table remains a source-backed software draft, not a hardware-confirmed deployment table.

Current status:

- joint order is sourced from `G0_JOINT_SDK_NAMES`
- default pose is sourced from `G0_DEFAULT_JOINT_POS`
- action scale is sourced from `JointPositionActionCfg(scale=0.12, use_default_offset=True, preserve_order=True)`
- effort and velocity limits are sourced from `g0_actuators.py`
- position limits in this audit are sourced from the URDF `<limit>` values
- `motor_id = index` is treated as a software convention only
- direction/sign notes remain software-side interpretation only

## Mapping Assumptions Requiring Future Hardware Confirmation

### Motor ID

`motor_id = index` is the current software convention only.

This convention is consistent with the SDK-ordered 22-joint action contract and is used throughout the dry-run validator and deployment tests, but it is not hardware-confirmed by an authoritative repository source.

Real robot deployment still requires hardware-side motor ID confirmation.

### Direction Sign

The mirrored default standing pose and URDF axis conventions strongly suggest the current software sign interpretation, but the audit does not claim that those signs are hardware-confirmed.

Real robot deployment still requires low-gain single-joint sign validation.

### Position Limits

This audit uses URDF limit values for the dry-run mapping tables and validators.

`soft_joint_pos_limit_factor = 0.9` exists in `g0.py`, but this audit does not claim that the current table is the final hardware-side enforcement source.

## Initial 22-Joint Mapping Table Draft

| index | motor_id | joint_name | default_q | action_scale | servo_type | effort_limit | velocity_limit | position_limit_lower | position_limit_upper | sign_note | source / confidence note |
|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---|---|
| 0 | 0 | l_hip_pitch_joint | -0.20 | 0.12 | standard | 0.5 | 31.416 | -0.75 | 0.75 | Left pitch default is negative; software sign only, not hardware-confirmed. | SDK order + default pose + URDF limit + standard servo grouping are repo-backed; `motor_id=index` is assumption. |
| 1 | 1 | l_hip_roll_joint | 0.00 | 0.12 | standard | 0.5 | 31.416 | -0.75 | 0.00 | Zero default; roll sign still needs hardware validation. | SDK order + default pose + URDF limit + standard servo grouping are repo-backed; `motor_id=index` is assumption. |
| 2 | 2 | l_hip_yaw_joint | 0.00 | 0.12 | standard | 0.5 | 31.416 | -1.00 | 1.00 | Zero default; yaw sign still needs hardware validation. | SDK order + default pose + URDF limit + standard servo grouping are repo-backed; `motor_id=index` is assumption. |
| 3 | 3 | l_knee_pitch_joint | -0.34 | 0.12 | right_angle | 0.5833 | 26.928 | -0.75 | 0.00 | Left knee pitch default is negative; software sign only, not hardware-confirmed. | SDK order + default pose + URDF limit + right-angle servo grouping are repo-backed; `motor_id=index` is assumption. |
| 4 | 4 | l_ankle_pitch_joint | 0.14 | 0.12 | right_angle | 0.5833 | 26.928 | -0.75 | 0.75 | Left ankle pitch default is positive; software sign only, not hardware-confirmed. | SDK order + default pose + URDF limit + right-angle servo grouping are repo-backed; right-angle ankle assignment remains an assumption in `g0.py`; `motor_id=index` is assumption. |
| 5 | 5 | l_ankle_roll_joint | 0.00 | 0.12 | standard | 0.5 | 31.416 | -0.75 | 0.75 | Zero default; roll sign still needs hardware validation. | SDK order + default pose + URDF limit + standard servo grouping are repo-backed; `motor_id=index` is assumption. |
| 6 | 6 | r_hip_pitch_joint | 0.20 | 0.12 | standard | 0.5 | 31.416 | -0.75 | 0.75 | Right pitch default is positive; software sign only, not hardware-confirmed. | SDK order + default pose + URDF limit + standard servo grouping are repo-backed; `motor_id=index` is assumption. |
| 7 | 7 | r_hip_roll_joint | 0.00 | 0.12 | standard | 0.5 | 31.416 | 0.00 | 0.75 | Zero default; roll sign still needs hardware validation. | SDK order + default pose + URDF limit + standard servo grouping are repo-backed; `motor_id=index` is assumption. |
| 8 | 8 | r_hip_yaw_joint | 0.00 | 0.12 | standard | 0.5 | 31.416 | -1.00 | 1.00 | Zero default; yaw sign still needs hardware validation. | SDK order + default pose + URDF limit + standard servo grouping are repo-backed; `motor_id=index` is assumption. |
| 9 | 9 | r_knee_pitch_joint | 0.34 | 0.12 | right_angle | 0.5833 | 26.928 | 0.00 | 0.75 | Right knee pitch default is positive; software sign only, not hardware-confirmed. | SDK order + default pose + URDF limit + right-angle servo grouping are repo-backed; `motor_id=index` is assumption. |
| 10 | 10 | r_ankle_pitch_joint | -0.14 | 0.12 | right_angle | 0.5833 | 26.928 | -0.75 | 0.75 | Right ankle pitch default is negative; software sign only, not hardware-confirmed. | SDK order + default pose + URDF limit + right-angle servo grouping are repo-backed; right-angle ankle assignment remains an assumption in `g0.py`; `motor_id=index` is assumption. |
| 11 | 11 | r_ankle_roll_joint | 0.00 | 0.12 | standard | 0.5 | 31.416 | -0.75 | 0.75 | Zero default; roll sign still needs hardware validation. | SDK order + default pose + URDF limit + standard servo grouping are repo-backed; `motor_id=index` is assumption. |
| 12 | 12 | waist_yaw_joint | 0.00 | 0.12 | standard | 0.5 | 31.416 | -1.20 | 1.20 | Zero default; yaw sign still needs hardware validation. | SDK order + default pose + URDF limit + standard servo grouping are repo-backed; `motor_id=index` is assumption. |
| 13 | 13 | waist_roll_joint | 0.00 | 0.12 | standard | 0.5 | 31.416 | -0.50 | 0.50 | Zero default; roll sign still needs hardware validation. | SDK order + default pose + URDF limit + standard servo grouping are repo-backed; `motor_id=index` is assumption. |
| 14 | 14 | l_shoulder_pitch_joint | -0.30 | 0.12 | standard | 0.5 | 31.416 | -1.20 | 1.20 | Left shoulder pitch default is negative; software sign only, not hardware-confirmed. | SDK order + default pose + URDF limit + standard servo grouping are repo-backed; `motor_id=index` is assumption. |
| 15 | 15 | l_shoulder_roll_joint | -0.25 | 0.12 | standard | 0.5 | 31.416 | -1.20 | 0.00 | Left shoulder roll default is negative; software sign only, not hardware-confirmed. | SDK order + default pose + URDF limit + standard servo grouping are repo-backed; `motor_id=index` is assumption. |
| 16 | 16 | l_shoulder_yaw_joint | 0.00 | 0.12 | standard | 0.5 | 31.416 | -1.20 | 1.20 | Zero default; yaw sign still needs hardware validation. | SDK order + default pose + URDF limit + standard servo grouping are repo-backed; `motor_id=index` is assumption. |
| 17 | 17 | l_elbow_pitch_joint | 0.97 | 0.12 | right_angle | 0.5833 | 26.928 | 0.00 | 1.20 | Left elbow pitch default is positive; software sign only, not hardware-confirmed. | SDK order + default pose + URDF limit + right-angle servo grouping are repo-backed; `motor_id=index` is assumption. |
| 18 | 18 | r_shoulder_pitch_joint | 0.30 | 0.12 | standard | 0.5 | 31.416 | -1.20 | 1.20 | Right shoulder pitch default is positive; software sign only, not hardware-confirmed. | SDK order + default pose + URDF limit + standard servo grouping are repo-backed; `motor_id=index` is assumption. |
| 19 | 19 | r_shoulder_roll_joint | 0.25 | 0.12 | standard | 0.5 | 31.416 | 0.00 | 1.20 | Right shoulder roll default is positive; software sign only, not hardware-confirmed. | SDK order + default pose + URDF limit + standard servo grouping are repo-backed; `motor_id=index` is assumption. |
| 20 | 20 | r_shoulder_yaw_joint | 0.00 | 0.12 | standard | 0.5 | 31.416 | -1.20 | 1.20 | Zero default; yaw sign still needs hardware validation. | SDK order + default pose + URDF limit + standard servo grouping are repo-backed; `motor_id=index` is assumption. |
| 21 | 21 | r_elbow_pitch_joint | -0.97 | 0.12 | right_angle | 0.5833 | 26.928 | -1.20 | 0.00 | Right elbow pitch default is negative; software sign only, not hardware-confirmed. | SDK order + default pose + URDF limit + right-angle servo grouping are repo-backed; `motor_id=index` is assumption. |

## Safety Filter Contract

The Phase B safety filter is pure Python only.

Required behavior:

- reject `hardware_enabled=True`
- reject NaN or Inf raw action
- clip raw action to `[-1, 1]`
- compute `target = default_joint_pos + action_scale * clipped_action`
- clamp target to position limits
- clamp action changes against `max_action_delta` when previous action exists
- clamp target changes against `max_target_delta` when previous target exists
- reject stale observations when `now - last_obs_time > max_command_age_s`
- on `emergency_stop=True`, return a safe hold command rather than an unsafe target jump
- ensure final `q`, `dq`, and `tau_ff` values are finite

The current implementation is intentionally dry-run-only and does not model a hardware transport.

## LowCmd-Style Dry-Run Command Schema

The Phase B/C/D dry-run mapping core produces a fake command object with:

- `FakeLowCmd.dry_run`
- `FakeLowCmd.control_dt`
- `FakeLowCmd.motors`

Each `FakeMotorCommand` contains:

- `motor_id`
- `joint_name`
- `q`
- `dq`
- `kp`
- `kd`
- `tau_ff`
- `effort_limit`
- `source_action_index`
- `source_raw_action`
- `source_clipped_action`

Current dry-run placeholders:

- `kp = 0`
- `kd = 0`
- `dq = 0`
- `tau_ff = 0`

These placeholders are acceptable for software-side dry-run validation only and are not deployment tuning.

## Hard FAIL vs Diagnostic-Only

| Condition | Status |
|---|---|
| checkpoint missing | hard FAIL |
| checkpoint SHA256 mismatch | hard FAIL |
| env construction failure | hard FAIL |
| policy load failure | hard FAIL |
| NaN or Inf action | hard FAIL |
| FakeLowCmd `dry_run is not True` | hard FAIL |
| motor count != 22 | hard FAIL |
| `motor_id` mismatch | hard FAIL |
| joint order mismatch | hard FAIL |
| non-finite command field | hard FAIL |
| `q` outside safety limits after filtering | hard FAIL |
| `hardware_enabled=True` | hard FAIL |
| stale observation reject missing | hard FAIL |
| emergency stop ignored | hard FAIL |
| raw action used directly without clipping | hard FAIL |
| raw action out-of-range count | diagnostic-only |
| target delta spikes within accepted safety contract | diagnostic-only |
| safety clamp correction magnitude | diagnostic-only |
| effort ratio saturation from separate rollout validation | diagnostic-only |
| joint-limit-margin interpretation at reset from separate rollout validation | diagnostic-only |
| `kp = 0`, `kd = 0` in dry-run path | diagnostic-only |

## Phase A Through Phase D Implementation Summary

### Phase A

- confirmed repository state and base branch ancestry
- inspected robot, actuator, task, helper, and documentation sources
- created the initial 22-joint mapping audit document

### Phase B

- added `scripts/validation/_safety_filter.py`
- added `scripts/validation/_lowcmd_mapping.py`
- kept `tests/helpers/deployment_dryrun.py` backward-compatible
- added deployment contract tests for the safety filter and mapping core

### Phase C

- added `scripts/validation/_joint_contract_tables.py`
- added `scripts/validation/validate_g0_lowcmd_mapping.py --mode offline-contract`
- added optional JSON output for offline validation

### Phase D

- extended `validate_g0_lowcmd_mapping.py` with `--mode isaac-policy-sample`
- reused the existing Isaac rollout loader and fixed validation environment path
- sampled the fixed checkpoint in Isaac Lab and mapped policy actions through the dry-run chain only
- confirmed no hardware path was entered

## Phase C Offline Validation Results

Latest known result:

- result: `PASS`
- joint count: `22`
- command count: `22`
- passed cases: `zero_action`, `all_plus_one`, `all_minus_one`, `overflow_action_ten`, `fast_action_jump`, `emergency_stop`
- rejected cases: `nan_action`, `inf_action`, `stale_observation`, `hardware_enabled_true`
- warning cases: `fast_action_jump`, `emergency_stop`
- motor_id convention: `motor_id = index`, software convention only
- no-hardware confirmation: `True`
- action clipping confirmation: `True`
- target mapping max error: `0`

Meaning:

- expected rejects rejected cleanly
- the dry-run mapping contract remained finite and bounded
- no hardware path was touched

## Phase D Isaac Policy Sampling Results

Latest known result:

- result: `PASS`
- task: `G0-Velocity-v0`
- checkpoint sha256: `1dc0c434a4b991eaaa435a21b9d4265e0267eb781b69b132bd75a0b5883928cd`
- steps: `500`
- num_envs: `1`
- raw action min/max/mean/std: `-2.9595861434936523 / 2.4234588146209717 / -0.028617099631916394 / 0.7163661538912613`
- raw action out_of_range_count: `1814`
- clipped action min/max: `-1.0 / 1.0`
- max target mapping error: `0.0`
- max safety clamp correction: `0.0`
- rejected step count: `0`
- emergency stop count: `0`
- stale observation count: `0`
- motor_id mismatch count: `0`
- non-finite command count: `0`
- worst raw action: `step=5, joint=r_ankle_roll_joint, value=-2.9595861434936523`
- worst target delta: `step=242, joint=l_hip_roll_joint, value=0.0761653697490692`
- worst safety clamp: `none`
- dry_run_only: `True`
- no-hardware confirmation: `True`

Meaning:

- raw policy actions still exceed `[-1, 1]` often, as expected from prior rollout diagnostics
- clipping remained bounded at `[-1, 1]`
- the dry-run mapping chain stayed finite and contract-clean for all 500 sampled steps
- no safety clamp correction was needed for the sampled 500-step run
- no hardware path was touched

## Full Regression Results

The latest Phase E regression run used the following commands:

```bash
python scripts/validation/validate_g0_lowcmd_mapping.py \
  --mode offline-contract \
  --emit-json logs/validation/lowcmd_mapping_offline.json

TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p scripts/validation/validate_g0_lowcmd_mapping.py \
  --mode isaac-policy-sample \
  --task G0-Velocity-v0 \
  --checkpoint logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/model_9999.pt \
  --headless \
  --steps 500 \
  --num-envs 1 \
  --emit-json logs/validation/lowcmd_mapping_isaac_500.json

conda run -n g0_isaaclab python -m pytest -q tests/deployment
conda run -n g0_isaaclab python -m pytest -q tests/unit
TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p -m pytest -q tests/isaaclab -m "not release_gate"
TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p -m pytest -q tests/isaaclab -m "release_gate"
```

Latest regression outcomes:

- offline validator: `PASS`
- Isaac policy sample: `PASS`
- deployment tests: `27 passed in 0.83s`
- unit tests: `16 passed in 0.03s`
- Isaac Lab non-release smoke: exit `0`, output `.`
- Isaac Lab release-gate selection: exit `0`, output `..`

## Forbidden Path Check

A repository scan for `socket`, `DDS`, `LowCmd`, `Unitree`, `sendto`, and `send(` across `scripts/validation`, `tests/helpers`, `tests/deployment`, and `docs` found:

- dry-run command names and documentation references
- deployment guardrail tests that intentionally block real socket sends
- no real hardware transport path added in `scripts/validation`
- no DDS path added in `scripts/validation`
- no real LowCmd SDK path added in `scripts/validation`

## Deployment-Readiness Disclaimer

Passing Phase E does not authorize real robot deployment.

Passing Phase E does not confirm hardware motor IDs.

Passing Phase E does not confirm hardware direction signs.

Passing Phase E does not validate real communication timing.

Passing Phase E does not validate emergency stop on real hardware.

The next stage after Phase E is hardware-free communication rehearsal, not real motor command.

## Phase F Release Gate

Phase F adds an optional Isaac Lab release gate:

`tests/isaaclab/test_release_gate_lowcmd_mapping_chain.py`

This gate runs the fixed 500-step Isaac policy sample through the dry-run mapping chain only.

Hard-fail policy for this gate:

- checkpoint missing
- checkpoint SHA256 mismatch
- Isaac env construction failure
- policy load failure
- raw action contains NaN or Inf
- clipped action contains NaN or Inf
- clipped action leaves `[-1, 1]` beyond tiny tolerance
- `FakeLowCmd.dry_run is not True`
- motor count != 22
- `motor_id != index`
- joint order mismatch
- any `q`, `dq`, `kp`, `kd`, or `tau_ff` is non-finite
- `q` after safety filtering is outside configured position limits
- target mapping error exceeds tolerance
- rejected step count > 0 during normal sampling
- emergency stop count != 0 during normal sampling
- stale observation count != 0 during normal sampling

Diagnostic-only in this gate:

- raw policy action outside `[-1, 1]`
- raw action out-of-range count
- worst raw action value
- target delta observations
- prior rollout effort-ratio diagnostics
- step-0 transient notes from other validation stages

Passing this gate validates the software dry-run mapping chain only. It does not authorize real LowCmd, real hardware, or real-robot deployment.
