# G0 Locomotion Training Report

## Project Structure

Current project root:

```text
/home/lz/g0_robot_lab/g0_robot_lab
```

Important paths:

- robot asset: `source/g0_robot_lab/g0_robot_lab/assets/robots/g0/g0.py`
- hardware constants: `source/g0_robot_lab/g0_robot_lab/assets/robots/g0/g0_actuators.py`
- task registration: `source/g0_robot_lab/g0_robot_lab/tasks/locomotion/__init__.py`
- environment config: `source/g0_robot_lab/g0_robot_lab/tasks/locomotion/robots/g0/velocity_env_cfg.py`
- PPO config: `source/g0_robot_lab/g0_robot_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`
- MDP functions: `source/g0_robot_lab/g0_robot_lab/tasks/locomotion/mdp/`
- debug sweep: `scripts/debug/sweep_g0_standing_pose.py`
- train script: `scripts/rsl_rl/train.py`
- play script: `scripts/rsl_rl/play.py`

## Reference Alignment

The current `g0_robot_lab` locomotion layout already follows the useful `unitree_rl_lab` organization:

- `assets/robots/...` defines robot articulation and actuator groups
- `tasks/locomotion/robots/.../velocity_env_cfg.py` defines scene, events, commands, actions, observations, rewards, terminations, curriculum
- `tasks/locomotion/mdp/` holds shared MDP functions
- `tasks/locomotion/agents/rsl_rl_ppo_cfg.py` defines PPO runner config
- `scripts/rsl_rl/train.py` and `play.py` are RSL-RL entry points

What must stay G0-specific:

- Do not use Unitree G1 actuator parameters.
- Do not use Unitree G1 default pose.
- Do not use G1 ankle-roll foot body matching. G0 feet are `l_foot_link` and `r_foot_link`.
- Do not copy G1 base height target. G0 currently uses a much smaller standing height target.

## Hardware Constraints

`g0_actuators.py` contains motor hardware constants such as rated torque, peak torque, max velocity, and armature. Those are treated as physical properties. The current documented baseline keeps the standard servo rated torque at `0.5 N*m`; runtime `--effort-scale` is debug-only and does not permanently change hardware constants.

`stiffness` and `damping` in `g0.py` are PD control parameters, so they can be tuned. `effort_limit_sim` and `velocity_limit_sim` still come from `g0_actuators.py`; no bypass was added.

## Changes Made

`g0.py`:

- changed the default pose to the current best simple two-decimal debug baseline:
  - `root_z = 0.23`
  - `hip_pitch = 0.20`
  - `knee_pitch = 0.34`
  - `ankle_pitch = 0.14`
- kept left/right symmetry with mirrored signs from the G0 URDF axis convention
- applied the requested initial PD values:
  - hip pitch `6.0 / 0.18`
  - hip roll `5.0 / 0.16`
  - hip yaw `3.0 / 0.10`
  - ankle roll `4.5 / 0.15`
  - knee pitch `8.0 / 0.26`
  - ankle pitch `6.5 / 0.22`
  - waist `2.0 / 0.08`
  - shoulder `1.5 / 0.06`
  - elbow pitch `2.0 / 0.08`

`velocity_env_cfg.py`:

- fixed `G0RobotLabPlayEnvCfg` terrain generator access
- set play command ranges to `limit_ranges`

`tasks/locomotion/__init__.py`:

- added `play_env_cfg_entry_point` for `G0RobotLabPlayEnvCfg`

`scripts/debug/sweep_zero_action_standing_pose.py` and `scripts/debug/sweep_g0_standing_pose.py`:

- added fixed and sweep modes
- added configurable two-decimal pose values, including root height
- added PD override options for debug only
- added foot force, foot imbalance, and foot height diagnostics
- added per-step per-joint torque CSV trace
- added `--print-torque-every` for throttled console torque summaries
- added debug-only `--effort-scale` with a hard cap at `2.0`
- kept debug reset/command/randomization disabled

## Zero-Action Results

Commands were run through:

```bash
TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p scripts/debug/sweep_g0_standing_pose.py \
  --task G0-Velocity-v0 \
  --headless \
  --mode fixed \
  --steps 500
```

The older high-precision candidate was tested earlier and failed:

```text
hip=0.1980 knee=0.3470 ankle=0.1504
survive=90/500
reason=bad_orientation
final_root_z=0.128
min_root_z=0.128
final_root_pitch_deg=69.789
pitch_slope_deg_per_step=0.5475
max_ankle_torque_ratio=1.000
```

The newer two-decimal fixed candidate was then tested with real-time torque logging:

```text
root_z=0.23 hip=0.20 knee=0.35 ankle=0.14 effort_scale=1.0
trace=logs/zero_action_torque_trace_20260512_154804.csv
survive=91/500
reason=bad_orientation
final_root_z=0.1262
final_root_pitch_deg=70.137
pitch_slope_deg_per_step=0.54495
max_hip_torque_ratio=0.397
max_knee_torque_ratio=0.597
max_ankle_torque_ratio=1.000
earliest ratio>0.90 joint=l_ankle_pitch_joint at step 57
```

Effort-scale checks on the same candidate:

```text
effort_scale=1.0 survive=91 final_pitch=70.137 max_ankle_ratio=1.000
effort_scale=1.2 survive=91 final_pitch=70.137 max_ankle_ratio=1.000
effort_scale=1.5 survive=91 final_pitch=70.137 max_ankle_ratio=0.959
effort_scale=2.0 survive=91 final_pitch=70.137 max_ankle_ratio=0.720
```

Best quick two-decimal sweep candidate:

```text
root_z=0.23 hip=0.20 knee=0.34 ankle=0.14
effort_scale=1.0 survive=116/500 reason=bad_orientation final_pitch=71.823 max_ankle_ratio=1.000
effort_scale=2.0 survive=116/500 reason=bad_orientation final_pitch=71.823 max_ankle_ratio=0.751
```

No two-decimal candidate has passed 500 zero-action steps yet. Raising effort scale to `2.0` removes long saturation for the best candidate, but the fall remains, so the most likely next target is asset/contact/COM/root-height diagnosis rather than PPO tuning.

## Training Status

Formal PPO training was not started because zero-action standing is not yet acceptable. Starting the 10000-iteration command now would likely train on immediate fall recovery noise rather than locomotion.

Target training command when standing is fixed:

```bash
TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task G0-Velocity-v0 \
  --num_envs 32 \
  --max_iterations 10000 \
  --headless
```

No walking checkpoint was generated in this pass.

Recommended play command after a checkpoint exists:

```bash
TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task G0-Velocity-v0 \
  --num_envs 32 \
  --checkpoint <checkpoint-path>
```

## Next Debug Steps

Recommended next checks before PPO:

- inspect collision geometry and foot contact patch; the foot link height diagnostic shows the foot link origins around `0.018 m`, but the robot still collapses around the sagittal axis
- verify body inertial parameters and COM locations from the converted USD/URDF
- test whether torso/root initial height should be adjusted after measuring actual contact settling
- consider a temporary standing-only controller/debug task with more detailed COM, COP, and contact wrench logging
- only after zero-action reaches close to 500 steps, continue to Stage A training from `docs/g0_locomotion_curriculum_plan.md`
