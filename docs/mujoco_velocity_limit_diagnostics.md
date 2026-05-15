# G0 MuJoCo Velocity-Limit Diagnostics

## Context

- Branch: `structure/mujoco-sim2sim-layout`
- Test time: `2026-05-15 15:48:50 CST`
- Goal: diagnose observed MuJoCo joint velocities against Isaac `velocity_limit_sim` without changing model parameters.

This pass intentionally does not tune `kp`, damping, friction, solver settings, or contact geometry.

## Command

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/check_mujoco_velocity_limits.py \
  --rollout logs/sim2sim/mujoco_zero_action_rollout.npz \
  --output logs/sim2sim/mujoco_velocity_limit_report.md
```

## Result

```text
exceeded joints: 0/22
worst ratio: 0.0967143 on l_hip_pitch_joint
```

No joint exceeded its Isaac `velocity_limit_sim` during the 100-step zero-action MuJoCo rollout.

Worst observed joints:

```text
l_hip_pitch_joint: max_abs_joint_vel=3.03837 rad/s, limit=31.4159 rad/s, ratio=0.0967143
r_hip_pitch_joint: max_abs_joint_vel=2.97551 rad/s, limit=31.4159 rad/s, ratio=0.0947135
l_shoulder_pitch_joint: max_abs_joint_vel=0.793995 rad/s, limit=31.4159 rad/s, ratio=0.0252736
```

## Interpretation

Velocity limits are not the immediate cause of the remaining zero-action differences in this rollout. The observed velocities are far below the Isaac limits.

However, this does not mean velocity-limit alignment is complete. The current MuJoCo model records Isaac `velocity_limit_sim` in the alignment table, but it does not yet implement a verified MuJoCo enforcement mechanism equivalent to Isaac/PhysX.

## Recommendation

- Do not add velocity-limit tuning based on this zero-action rollout.
- Keep `velocity_limit_sim` as a tracked diagnostic until policy rollout or disturbance tests approach the limits.
- If future policy rollouts exceed these limits, implement and document an explicit MuJoCo-side velocity-limit strategy rather than silently changing actuator gains.
