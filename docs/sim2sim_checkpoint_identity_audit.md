# Sim2sim Checkpoint Identity Audit

Date: 2026-05-18

## Pinned Policy

- policy_path: `/home/lz/g0_robot_lab/g0_robot_lab/logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/model_9999.pt`
- policy_filename: `model_9999.pt`
- policy_sha256: `1dc0c434a4b991eaaa435a21b9d4265e0267eb781b69b132bd75a0b5883928cd`
- checkpoint_run_folder: `/home/lz/g0_robot_lab/g0_robot_lab/logs/rsl_rl/g0_velocity/2026-05-14_18-29-19`
- task: `G0-Velocity-v0`
- command: `[0.1, 0.0, 0.0]`
- steps: `1000`
- action_dim: `22`
- obs_dim: `385`
- action_scale: `0.12`

## Audit Result

There was a real checkpoint identity risk before this audit. Several scripts and docs used an exported relative path such as `logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/exported/policy.pt`, and one deploy matrix script had that path as a default. That exported artifact does not prove which raw checkpoint produced it, so the previous policy checkpoint identity is `unknown`.

The sim2sim policy loaders now require an explicit absolute `--checkpoint` or `--policy` for live-policy modes. The runners can load either TorchScript or the raw RSL-RL actor checkpoint, and rollout artifacts record policy path, filename, SHA256, run folder, task, command, steps, action dimension, observation dimension, joint names, and action scale.

## Regenerated Artifacts

- Isaac golden I/O: `logs/sim2sim/isaac_model_9999_cmd01.npz`
- custom MuJoCo replay target: `logs/sim2sim/custom_mujoco_replay_target_model_9999_cmd01.npz`
- custom MuJoCo live policy: `logs/sim2sim/custom_mujoco_policy_model_9999_cmd01.npz`
- unitree-style replay target: `logs/sim2sim/unitree_style_replay_target_model_9999_cmd01.npz`
- unitree-style live policy: `logs/sim2sim/unitree_style_policy_model_9999_cmd01.npz`
- custom replay report: `logs/sim2sim/compare_model_9999_custom_replay_target.md`
- custom policy report: `logs/sim2sim/compare_model_9999_custom_policy.md`
- unitree-style report: `logs/sim2sim/compare_model_9999_unitree_style.md`

Additional zero-action spot checks:

- custom zero-action 100 steps: `logs/sim2sim/custom_mujoco_zero_action_100_checkpoint_audit.npz`
- unitree-style zero-action 100 steps: `logs/sim2sim/unitree_style_zero_action_100_checkpoint_audit.npz`

## Minimum Revalidation Results

| rollout | root height min | root height final | action max | action saturation | torque saturation | result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Isaac policy golden | 0.227384 | 0.230000 | 1.000000 | 0.180955 | n/a | stable |
| custom replay target | 0.031360 | 0.040639 | 1.000000 | 0.180955 | n/a | collapsed |
| custom live policy | 0.033228 | 0.035982 | 1.000000 | 0.450955 | n/a | collapsed |
| unitree-style replay target | 0.043864 | 0.045913 | 1.000000 | 0.180955 | 0.000136 | collapsed |
| unitree-style live policy | 0.031130 | 0.034413 | 1.000000 | 0.270182 | 0.002182 | collapsed |

Zero-action spot checks:

| rollout | steps | root height min | root height final | result |
| --- | ---: | ---: | ---: | --- |
| custom zero-action | 100 | 0.027547 | 0.040076 | collapsed |
| unitree-style zero-action | 100 | 0.030151 | 0.040035 | collapsed |

## Answers

- Was policy checkpoint selection ambiguous before this audit? Yes.
- Which policy was previously used? The previous live runs used an exported `policy.pt` path in several commands, but the source raw checkpoint cannot be proven from current artifacts, so the checkpoint identity is `unknown`.
- Is `model_9999.pt` stable inside Isaac Lab? Yes. The deterministic Isaac rollout stayed near 0.23 m root height for 1000 steps.
- Does custom MuJoCo replay target still collapse with `model_9999.pt`? Yes. `target_joint_pos` and action metadata match Isaac, but root height falls to about 0.04 m.
- Does custom MuJoCo live policy still collapse with `model_9999.pt`? Yes.
- Does unitree-style replay/live improve the result? Not in the stability sense. Replay target is slightly higher than custom replay but still collapsed. Live policy has lower action saturation than custom live, but root height still collapses.
- If zero-action still collapses, does it still indicate MuJoCo model/contact/root equilibrium trouble? Yes. Checkpoint identity cannot explain zero-action collapse.
- Does this change the conclusion about retraining? No. Because `model_9999.pt` is stable in Isaac and MuJoCo replay of Isaac target positions still collapses, this audit strengthens the conclusion that MuJoCo model/contact/control dynamics must be fixed before using MuJoCo failure as a reason to retrain.

## Reproduction Commands

Isaac golden:

```bash
TERM=xterm conda run -n g0_isaaclab /home/lz/IsaacLab/isaaclab.sh -p scripts/sim2sim/dump_isaac_golden_io.py \
  --task G0-Velocity-v0 \
  --checkpoint /home/lz/g0_robot_lab/g0_robot_lab/logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/model_9999.pt \
  --steps 1000 \
  --num_envs 1 \
  --output logs/sim2sim/isaac_model_9999_cmd01.npz \
  --deterministic-zero \
  --command 0.1 0 0 \
  --headless
```

Custom MuJoCo replay target:

```bash
conda run -n g0_isaaclab python scripts/sim2sim/play_mujoco_g0.py \
  --model mujoco/g0.xml \
  --steps 1000 \
  --replay logs/sim2sim/isaac_model_9999_cmd01.npz \
  --replay-field target_joint_pos \
  --record-rollout logs/sim2sim/custom_mujoco_replay_target_model_9999_cmd01.npz
```

Custom MuJoCo live policy:

```bash
conda run -n g0_isaaclab python scripts/sim2sim/play_mujoco_g0.py \
  --model mujoco/g0.xml \
  --policy /home/lz/g0_robot_lab/g0_robot_lab/logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/model_9999.pt \
  --steps 1000 \
  --command 0.1 0 0 \
  --record-rollout logs/sim2sim/custom_mujoco_policy_model_9999_cmd01.npz \
  --device cpu
```

Unitree-style replay target:

```bash
conda run -n g0_isaaclab python scripts/unitree_mujoco_g0/run_unitree_mujoco_g0_replay_target.py \
  --steps 1000 \
  --golden logs/sim2sim/isaac_model_9999_cmd01.npz \
  --record-rollout logs/sim2sim/unitree_style_replay_target_model_9999_cmd01.npz
```

Unitree-style live policy:

```bash
conda run -n g0_isaaclab python scripts/unitree_mujoco_g0/run_unitree_mujoco_g0_policy.py \
  --steps 1000 \
  --policy /home/lz/g0_robot_lab/g0_robot_lab/logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/model_9999.pt \
  --command 0.1 0 0 \
  --record-rollout logs/sim2sim/unitree_style_policy_model_9999_cmd01.npz \
  --device cpu
```

Reports:

```bash
conda run -n g0_isaaclab python scripts/sim2sim/compare_isaac_mujoco_rollout.py \
  --isaac logs/sim2sim/isaac_model_9999_cmd01.npz \
  --mujoco logs/sim2sim/custom_mujoco_replay_target_model_9999_cmd01.npz \
  --output logs/sim2sim/compare_model_9999_custom_replay_target.md

conda run -n g0_isaaclab python scripts/sim2sim/compare_isaac_mujoco_rollout.py \
  --isaac logs/sim2sim/isaac_model_9999_cmd01.npz \
  --mujoco logs/sim2sim/custom_mujoco_policy_model_9999_cmd01.npz \
  --output logs/sim2sim/compare_model_9999_custom_policy.md

conda run -n g0_isaaclab python scripts/unitree_mujoco_g0/compare_isaac_unitree_mujoco_rollout.py \
  --isaac logs/sim2sim/isaac_model_9999_cmd01.npz \
  --custom logs/sim2sim/custom_mujoco_policy_model_9999_cmd01.npz \
  --unitree logs/sim2sim/unitree_style_policy_model_9999_cmd01.npz \
  --output logs/sim2sim/compare_model_9999_unitree_style.md
```

## Next Smallest Closed-Loop Experiment

Do not retrain yet. The next smallest sim2sim experiment is to fix MuJoCo root equilibrium/contact/model fidelity until zero-action and replay-target no longer collapse. Only after target replay is physically reasonable should live-policy observation drift and action saturation be used to judge policy transfer quality.
