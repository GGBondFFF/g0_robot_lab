# LowCmd Mapping Safety Audit

## Purpose

This document records the Phase A audit skeleton for the deployment-facing mapping chain on branch `validation/lowcmd-mapping-safety-audit`.

The immediate goal is to document the current software contract for converting the 22-dimensional G0 locomotion policy action into a LowCmd-style dry-run representation, without entering any real hardware path.

## Scope

Phase A established the joint/motor mapping draft and safety-audit document skeleton.

Phase C adds an offline validation script for the Phase B dry-run safety and mapping core.

Included in scope:

- repository state confirmation
- read-only inspection of the current G0 joint, actuator, dry-run helper, and validation documentation contracts
- an initial 22-joint mapping table draft
- explicit notes about assumptions that still require real hardware confirmation
- offline-only LowCmd mapping contract validation using fake commands

Out of scope in Phase A:

- `_safety_filter.py`
- `_lowcmd_mapping.py`
- `deployment_dryrun.py` refactor
- deployment test additions
- Isaac release gate additions
- any real LowCmd send path
- any real hardware connection or motor command path

## No-Hardware Warning

This audit does not authorize real robot deployment.

This audit does not authorize LowCmd transmission.

This audit does not confirm motor IDs on hardware.

This audit does not confirm joint direction signs on hardware.

Any future real-robot work still requires a separately approved bring-up procedure, hardware-side motor ID confirmation, and low-gain single-joint sign validation.

Phase C offline validation is dry-run only. It must not send real LowCmd, must not connect to hardware, and must not be interpreted as evidence of real-robot readiness.

## Relation To Policy Rollout Safety Validation

The prior Isaac Lab policy rollout safety stage established that the fixed checkpoint can complete the current validation-only rollout checks in simulation, but it also exposed deployment-relevant risks:

- raw policy action can exceed `[-1, 1]`
- any deployment-facing path must clip before producing any LowCmd-compatible representation
- effort ratio, joint-limit margin, and target-delta findings remain diagnostic and do not imply hardware readiness

This audit therefore treats the deployment-facing chain as:

`raw policy action -> clipped action [-1, 1] -> target joint position -> safety filter -> LowCmd-style dry-run command`

The current offline dry-run helper already applies clipping before `default_joint_pos + 0.12 * clipped_action`, which is consistent with the policy rollout diagnostic requirement. Phase A does not extend that helper; it only documents the current state.

## Repository-Backed Contract Snapshot

- Branch under audit: `validation/lowcmd-mapping-safety-audit`
- Base branch ancestor confirmed: `validation/isaac-lowcmd-dryrun`
- Policy action order source: `G0_JOINT_SDK_NAMES`
- Action scale source: `JointPositionActionCfg(scale=0.12, use_default_offset=True, preserve_order=True)`
- Default pose source: `G0_DEFAULT_JOINT_POS`
- Dry-run helper behavior source: clip action to `[-1, 1]`, then compute `default + action_scale * clipped`
- Servo grouping source: `G0_RIGHT_ANGLE_SERVO_JOINT_NAMES` plus the remaining SDK joints as standard servos
- Position-limit source in this Phase A draft: URDF `<limit lower=... upper=...>` values
- Important caveat: `G0ArticulationCfg.soft_joint_pos_limit_factor = 0.9` is configured in `g0.py`, but Phase A does not derive the runtime Isaac soft limits yet, so the table below uses repository-visible URDF limits as the currently inspectable source-backed draft bounds

## Mapping Assumptions Requiring Future Hardware Confirmation

### Motor ID

Phase A documents `motor_id = index` as the current software convention only.

This convention is consistent with the SDK-ordered 22-joint action contract and is a reasonable draft for dry-run mapping tables, but no authoritative hardware-side confirmation was found in the inspected repository sources.

Real robot deployment still requires hardware-side motor ID confirmation.

### Direction Sign

The default standing pose and mirrored URDF axes strongly suggest the current software sign convention for several paired pitch joints, but Phase A does not treat those signs as hardware-confirmed.

Real robot deployment still requires low-gain single-joint sign validation before any motor command path is enabled.

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

## Phase A Findings

### Files Inspected

- `docs/superpowers/plans/2026-05-20-lowcmd-mapping-safety-audit.md`
- `source/g0_robot_lab/g0_robot_lab/assets/robots/g0/g0.py`
- `source/g0_robot_lab/g0_robot_lab/assets/robots/g0/g0_actuators.py`
- `source/g0_robot_lab/g0_robot_lab/tasks/locomotion/robots/g0/velocity_env_cfg.py`
- `tests/unit/test_g0_joint_contract.py`
- `tests/unit/test_velocity_env_static_contract.py`
- `tests/helpers/deployment_dryrun.py`
- `tests/helpers/isaaclab_runtime.py`
- `docs/observation_action_interface.md`
- `docs/pre_deployment_validation.md`
- `docs/run_commands.md`
- `docs/policy_rollout_safety_validation.md`
- `source/g0_robot_lab/g0_robot_lab/assets/robots/g0/inspect_g0_urdf.py`
- `source/g0_robot_lab/g0_robot_lab/assets/robots/g0/urdf/g0.urdf`
- `docs/g0_actuator_parameters.md`

### What Was Confirmed

- The current branch is `validation/lowcmd-mapping-safety-audit`.
- `validation/isaac-lowcmd-dryrun` is an ancestor of the current branch.
- The current worktree includes the untracked plan file `docs/superpowers/plans/2026-05-20-lowcmd-mapping-safety-audit.md`.
- The deployment-facing action order is the 22-joint `G0_JOINT_SDK_NAMES` list.
- The current policy action scale is `0.12`, with `use_default_offset=True` and `preserve_order=True`.
- The default standing pose covers exactly the SDK joint set and uses mirrored left/right pitch defaults for several paired joints.
- The current dry-run helper clips each raw action to `[-1, 1]` before computing `target_position = default_joint_pos + action_scale * clipped`.
- The current servo partition is 16 standard joints plus 6 right-angle joints.
- Repository-visible URDF joint limit values exist for all 22 movable joints.

### What Remains Uncertain

- `motor_id = index` is not hardware-confirmed in the inspected repository.
- Joint direction signs are not hardware-confirmed in the inspected repository.
- The right-angle ankle assignment to `*_ankle_pitch_joint` is explicitly documented as an assumption in `g0.py`.
- Phase A does not yet derive Isaac runtime `soft_joint_pos_limits`; the table currently uses URDF limit values rather than runtime soft-limit outputs.
- No repository source inspected in Phase A establishes hardware-side PD, dq, or torque-feedforward deployment behavior for a real LowCmd path.

### What Must Be Resolved Before Real Hardware

- hardware-side motor ID confirmation for all 22 joints
- low-gain single-joint sign validation for all command directions
- confirmation of which ankle joints use right-angle servos on real hardware
- confirmation of the exact deployment-side position-limit source to enforce, including any soft-limit reduction versus raw URDF bounds
- implementation and validation of an explicit safety filter before any LowCmd-compatible real command path is considered

## Phase C Offline Validation

Phase C adds:

- `scripts/validation/validate_g0_lowcmd_mapping.py`
- optional JSON output at `logs/validation/lowcmd_mapping_offline.json`

Command:

```bash
python scripts/validation/validate_g0_lowcmd_mapping.py \
  --mode offline-contract \
  --emit-json logs/validation/lowcmd_mapping_offline.json
```

This script is offline only and uses the pure-Python safety filter plus the fake LowCmd dry-run mapping core. It does not import Isaac, does not start `AppLauncher`, does not use sockets or DDS, does not use a real LowCmd SDK, and does not connect to hardware.

The offline validator checks:

- 22-joint mapping table shape and order
- `motor_id = index` as the current software convention
- action clipping before target generation
- `target = default_joint_pos + action_scale * clipped_action` for the normal contract cases
- position-limit-safe, finite `q` outputs
- 22 finite dry-run motor commands
- expected rejects for NaN, Inf, stale observations, and `hardware_enabled=True`
- safe hold behavior for `emergency_stop=True`

Passing this offline validator is useful for deployment-path contract confidence only. It is not a release gate, not a hardware check, and not a real-robot readiness signal.

## Phase D Isaac Policy Sampling

Phase D extends the validator with an Isaac policy-sampling mode that reuses the fixed checkpoint and fixed validation environment conditions from the policy rollout workflow, but still maps actions into fake LowCmd dry-run commands only.

Command:

```bash
/home/lz/IsaacLab/isaaclab.sh -p scripts/validation/validate_g0_lowcmd_mapping.py \
  --mode isaac-policy-sample \
  --task G0-Velocity-v0 \
  --checkpoint logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/model_9999.pt \
  --headless \
  --steps 500 \
  --num-envs 1 \
  --emit-json logs/validation/lowcmd_mapping_isaac_500.json
```

This stage is Isaac policy sampling only.

It still:

- uses dry-run fake LowCmd commands only
- sends no real LowCmd
- touches no hardware path
- does not indicate real-robot readiness

The Phase D chain is:

`raw policy action -> clipped action [-1, 1] -> target joint position -> safety filter -> FakeLowCmd dry-run command`
