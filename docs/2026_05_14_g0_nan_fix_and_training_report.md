# 2026-05-14 G0 NaN Fix And Training Report

## Scope

This report covers the 9224-iteration NaN/std failure analysis and the first stabilizing code changes in `g0_robot_lab`.

Current project root:

```text
/home/lz/g0_robot_lab/g0_robot_lab
```

No `humanoid_lab_v0` code, old project `tasks/manager_based` directory, Unitree G1/Go2 joint names, or Unitree actuator numbers were introduced.

## Reading Conclusions

- G0 has 22 movable joints.
- Runtime action order is intended to follow `G0_JOINT_SDK_NAMES`.
- The action is `JointPositionActionCfg`, not torque control.
- Current action semantics are position target offsets around default joint positions:
  - `scale=0.12`
  - `use_default_offset=True`
  - `preserve_order=True`
- Valid foot bodies are only:
  - `l_foot_link`
  - `r_foot_link`
- `g0_actuators.py` keeps the real servo constants:
  - standard rated torque: `0.5 Nm`
  - standard peak torque: `1.0 Nm`
  - standard max velocity: `31.416 rad/s`
  - right-angle rated torque: `0.583333 Nm`
  - right-angle max velocity: `26.928 rad/s`

No motor torque, velocity, gear ratio, armature, or hardware constants were changed.

## Baseline Play

Command run:

```bash
cd /home/lz/g0_robot_lab/g0_robot_lab
source /home/lz/miniconda3/etc/profile.d/conda.sh
conda activate g0_isaaclab
TERM=xterm HYDRA_FULL_ERROR=1 /home/lz/IsaacLab/isaaclab.sh -p scripts/rsl_rl/play.py \
  --task G0-Velocity-v0 \
  --num_envs 1 \
  --checkpoint logs/rsl_rl/g0_velocity/2026-05-13_17-06-23/model_9200.pt \
  --video \
  --video_length 2000
```

Result:

- Isaac Lab started successfully after activating `g0_isaaclab`.
- Video was generated:
  - `logs/rsl_rl/g0_velocity/2026-05-13_17-06-23/videos/play/rl-video-step-0.mp4`
- Exported models were generated:
  - `logs/rsl_rl/g0_velocity/2026-05-13_17-06-23/exported/policy.pt`
  - `logs/rsl_rl/g0_velocity/2026-05-13_17-06-23/exported/policy.onnx`
- Visual sample frames showed the robot upright during the recorded rollout, but this was only a visual baseline, not final validation.

## 9224 Failure Chain

The immediate exception:

```text
RuntimeError: normal expects all elements of std >= 0.0
```

The installed RSL-RL implementation uses `GaussianDistribution(std_type="scalar")` by default. In that mode, `distribution.std_param` is a directly learnable parameter. It is passed directly into `torch.distributions.Normal(mean, std)`. If Adam pushes any element below zero, `Normal.sample()` fails.

The likely causal chain is:

1. Old PPO config used `learning_rate=1e-3`, `init_noise_std=1.0`, no empirical normalization, and scalar learnable std.
2. Raw policy actions could become very large. Analysis of the short resumed checkpoint found raw action max around `80` before action clipping was added.
3. Because `JointPositionActionCfg(scale=0.12)` was unbounded, raw action spikes could request target joint positions many radians away from the default pose.
4. The real motor limits still clipped torque to `0.5/0.583 Nm`, so the robot could not follow those targets. This created saturated, high-rate action behavior.
5. `action_rate_l2` with weight `-0.05` made a single reward term dominate the episode return. Your failed log showed `Episode_Reward/action_rate=-3873.3169`.
6. Large negative returns made critic targets high-variance and unstable. This explains why `Mean value loss` became `nan` before the std exception surfaced.
7. Once value/advantage/loss magnitudes became unstable, optimizer updates could corrupt both network weights and the scalar std parameter.
8. Because scalar std is not constrained positive, the next stochastic action sample could hit `std < 0`, causing the runtime error.

`model_9200.pt` itself was not already broken:

```text
distribution.std_param min=0.251749 max=0.691551 finite=True
```

So the checkpoint appears recoverable, but it should be resumed with safer PPO numerics and optimizer reset.

## Code Changes

Changed files:

- `scripts/rsl_rl/train.py`
- `scripts/rsl_rl/play.py`
- `scripts/debug/check_g0_training_numerics.py`
- `scripts/analysis/analyze_g0_checkpoint.py`
- `source/g0_robot_lab/g0_robot_lab/tasks/locomotion/__init__.py`
- `source/g0_robot_lab/g0_robot_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py`
- `source/g0_robot_lab/g0_robot_lab/tasks/locomotion/mdp/__init__.py`
- `source/g0_robot_lab/g0_robot_lab/tasks/locomotion/mdp/curriculums.py`
- `source/g0_robot_lab/g0_robot_lab/tasks/locomotion/mdp/events.py`
- `source/g0_robot_lab/g0_robot_lab/tasks/locomotion/mdp/rewards.py`
- `source/g0_robot_lab/g0_robot_lab/tasks/locomotion/robots/g0/velocity_env_cfg.py`

Main changes:

- PPO learning rate changed from `1e-3` to `3e-4`.
- Initial std changed from `1.0` to `0.6`.
- RSL-RL distribution changed to official log-std parameterization:
  - `RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.6, std_type="log")`
- Actor and critic empirical observation normalization enabled.
- `clip_actions=1.0` added to prevent raw policy actions from commanding impossible joint targets.
- The exported policy network itself does not bake in this environment-side clip. Deployment code must also clip policy output to `[-1, 1]` before applying `0.12 rad` action scaling.
- `action_rate` reward weight reduced from `-0.05` to `-0.002`.
- Foot clearance replaced by a swing-foot-gated reward.
- `train.py` and `play.py` can migrate old scalar-std checkpoints to log-std in memory.
- The migration resets optimizer state when changing distribution/normalization, avoiding stale Adam moments.
- Added staged event configs:
  - Stage 0: baseline numerics.
  - Stage 1: flat-ground friction/mass/reset randomization.
  - Stage 2: small push perturbations.
  - Stage 3: very light uneven terrain.
- Added task IDs:
  - `G0-Velocity-Stage1-v0`
  - `G0-Velocity-Stage2-v0`
  - `G0-Velocity-Stage3-v0`

## Debug Check

Command:

```bash
TERM=xterm HYDRA_FULL_ERROR=1 /home/lz/IsaacLab/isaaclab.sh -p scripts/debug/check_g0_training_numerics.py \
  --task G0-Velocity-v0 \
  --num_envs 32 \
  --checkpoint logs/rsl_rl/g0_velocity/2026-05-13_17-06-23/model_9200.pt \
  --iterations 1 \
  --steps_per_env 4 \
  --headless
```

Result:

```text
iteration=9200 ok
std_mean=0.555389
loss finite
```

The debug path checks:

- observations
- rewards
- actions
- actor mean/std
- critic value
- loss dict
- reward term statistics
- termination ratios

## Smoke Training

Command:

```bash
TERM=xterm HYDRA_FULL_ERROR=1 /home/lz/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task G0-Velocity-v0 \
  --num_envs 128 \
  --resume \
  --load_run 2026-05-13_17-06-23 \
  --checkpoint model_9200.pt \
  --max_iterations 2 \
  --run_name g0_stage0_nan_smoke_from_9200 \
  --headless \
  --g0_actor_critic_resume_only
```

Generated checkpoint:

```text
logs/rsl_rl/g0_velocity/2026-05-14_09-47-49_g0_stage0_nan_smoke_from_9200/model_9201.pt
```

Important: this is a smoke checkpoint proving the new numeric path can train without immediate NaN/std failure. It is not a final deploy checkpoint.

Smoke metrics:

- `Mean value loss` stayed finite.
- `Mean action std` stayed about `0.56`.
- `Episode_Reward/action_rate` became small compared with the old failed `-3873` scale.
- Early short-run terminations were still high, especially `bad_orientation`; this needs real continued training.

## Exported Smoke Policy

Generated by running `play.py` on `model_9201.pt`:

```text
logs/rsl_rl/g0_velocity/2026-05-14_09-47-49_g0_stage0_nan_smoke_from_9200/exported/policy.pt
logs/rsl_rl/g0_velocity/2026-05-14_09-47-49_g0_stage0_nan_smoke_from_9200/exported/policy.onnx
```

Play video:

```text
logs/rsl_rl/g0_velocity/2026-05-14_09-47-49_g0_stage0_nan_smoke_from_9200/videos/play/rl-video-step-0.mp4
```

## Remaining Work

The requested final deploy-level checkpoint has not been completed in this pass. Required next run:

1. Continue Stage 0 from `model_9200.pt` with the new numeric config until bad orientation/base height terminations drop.
2. Move to `G0-Velocity-Stage1-v0`.
3. Only after stable Stage 1, move to Stage 2 pushes.
4. Only after stable Stage 2, try Stage 3 light terrain.
5. Re-run checkpoint analysis and play video for the final selected checkpoint.

Recommended Stage 0 continuation:

```bash
HYDRA_FULL_ERROR=1 /home/lz/IsaacLab/isaaclab.sh -p scripts/rsl_rl/train.py \
  --task G0-Velocity-v0 \
  --num_envs 4096 \
  --resume \
  --load_run 2026-05-13_17-06-23 \
  --checkpoint model_9200.pt \
  --max_iterations 20000 \
  --run_name g0_stage0_nan_fix_from_9200 \
  --headless \
  --g0_actor_critic_resume_only
```
