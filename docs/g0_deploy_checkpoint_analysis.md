# G0 Checkpoint Analysis

## Checkpoint

Analyzed checkpoint:

```text
logs/rsl_rl/g0_velocity/2026-05-14_09-47-49_g0_stage0_nan_smoke_from_9200/model_9201.pt
```

This is a Stage-0 numeric smoke checkpoint, not a final deployment checkpoint.

## Analysis Command

```bash
HYDRA_FULL_ERROR=1 /home/lz/IsaacLab/isaaclab.sh -p scripts/analysis/analyze_g0_checkpoint.py \
  --task G0-Velocity-v0 \
  --checkpoint logs/rsl_rl/g0_velocity/2026-05-14_09-47-49_g0_stage0_nan_smoke_from_9200/model_9201.pt \
  --duration 10 \
  --warmup 2 \
  --headless \
  --output_dir logs/rsl_rl/g0_velocity/2026-05-14_09-47-49_g0_stage0_nan_smoke_from_9200/analysis/2026-05-14_09-51-contactfix
```

The analysis segment is:

```text
t = 2.00s to 12.00s
step index = 100 to 599
```

## Output Files

Raw data:

```text
logs/rsl_rl/g0_velocity/2026-05-14_09-47-49_g0_stage0_nan_smoke_from_9200/analysis/2026-05-14_09-51-contactfix/rollout_data.npz
logs/rsl_rl/g0_velocity/2026-05-14_09-47-49_g0_stage0_nan_smoke_from_9200/analysis/2026-05-14_09-51-contactfix/rollout_summary.csv
```

Video:

```text
logs/rsl_rl/g0_velocity/2026-05-14_09-47-49_g0_stage0_nan_smoke_from_9200/analysis/2026-05-14_09-51-contactfix/videos/rl-video-step-0.mp4
```

Plots:

```text
logs/rsl_rl/g0_velocity/2026-05-14_09-47-49_g0_stage0_nan_smoke_from_9200/analysis/2026-05-14_09-51-contactfix/torque_left_leg.png
logs/rsl_rl/g0_velocity/2026-05-14_09-47-49_g0_stage0_nan_smoke_from_9200/analysis/2026-05-14_09-51-contactfix/torque_right_leg.png
logs/rsl_rl/g0_velocity/2026-05-14_09-47-49_g0_stage0_nan_smoke_from_9200/analysis/2026-05-14_09-51-contactfix/torque_waist.png
logs/rsl_rl/g0_velocity/2026-05-14_09-47-49_g0_stage0_nan_smoke_from_9200/analysis/2026-05-14_09-51-contactfix/torque_left_arm.png
logs/rsl_rl/g0_velocity/2026-05-14_09-47-49_g0_stage0_nan_smoke_from_9200/analysis/2026-05-14_09-51-contactfix/torque_right_arm.png
logs/rsl_rl/g0_velocity/2026-05-14_09-47-49_g0_stage0_nan_smoke_from_9200/analysis/2026-05-14_09-51-contactfix/joint_pos_left_leg.png
logs/rsl_rl/g0_velocity/2026-05-14_09-47-49_g0_stage0_nan_smoke_from_9200/analysis/2026-05-14_09-51-contactfix/joint_pos_right_leg.png
logs/rsl_rl/g0_velocity/2026-05-14_09-47-49_g0_stage0_nan_smoke_from_9200/analysis/2026-05-14_09-51-contactfix/root_position_approximation.png
logs/rsl_rl/g0_velocity/2026-05-14_09-47-49_g0_stage0_nan_smoke_from_9200/analysis/2026-05-14_09-51-contactfix/projected_gravity.png
logs/rsl_rl/g0_velocity/2026-05-14_09-47-49_g0_stage0_nan_smoke_from_9200/analysis/2026-05-14_09-51-contactfix/foot_contact.png
logs/rsl_rl/g0_velocity/2026-05-14_09-47-49_g0_stage0_nan_smoke_from_9200/analysis/2026-05-14_09-51-contactfix/foot_contact_force.png
logs/rsl_rl/g0_velocity/2026-05-14_09-47-49_g0_stage0_nan_smoke_from_9200/analysis/2026-05-14_09-51-contactfix/feet_slide.png
logs/rsl_rl/g0_velocity/2026-05-14_09-47-49_g0_stage0_nan_smoke_from_9200/analysis/2026-05-14_09-51-contactfix/feet_clearance.png
```

## Recorded Signals

The analysis script records:

- all joint positions
- all joint velocities
- raw action
- processed target joint position
- applied torque
- root/base position
- projected gravity
- foot contact state
- foot contact force
- feet slide metric
- feet clearance
- timeout/base_height/bad_orientation termination flags

The script currently uses root/base position as a COM approximation. It labels the plot `root/base position approximation` and does not call it true whole-body COM.

## Key Numeric Results

For `t=2.0s` to `t=12.0s`:

```text
action max abs: 1.0
target joint position max abs: 1.0900 rad
torque max abs: 0.583333 Nm
termination sums [timeout, base_height, bad_orientation]: [0.0, 5.0, 16.0]
root z min/max: 0.1602 / 0.2343 m
left/right foot contact mean: 0.748 / 0.636
left/right max contact force: 21.29 / 21.29 N
left/right mean foot slide: 0.053 / 0.037 m/s while in contact
left/right max foot slide: 0.578 / 0.418 m/s while in contact
```

Torque max matches the configured actuator limits:

- standard servos: `0.5 Nm`
- right-angle servos: `0.583333 Nm`

## Conclusion

The numeric fixes work in the sense that:

- actions entering the environment are bounded;
- target joint positions are no longer absurd multi-radian commands;
- torque limits remain real G0 motor limits;
- exported policy files can be generated;
- rollout data and plots are produced.

The behavior is not yet acceptable for final deployment:

- `bad_orientation` and `base_height` still trigger during the analyzed segment.
- Contact and slide metrics show the gait is not robust enough.
- This checkpoint should be treated as a diagnostic smoke checkpoint only.

Deployment note: `clip_actions=1.0` is applied by the Isaac Lab/RSL-RL environment wrapper. The exported `policy.pt` and `policy.onnx` are just the neural network, so real deployment must explicitly clip policy output to `[-1, 1]` before converting it to G0 joint targets.

The next deploy candidate must be produced after longer Stage 0 stabilization, then Stage 1/2/3 progression.
