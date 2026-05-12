# G0 Locomotion Curriculum Plan

## Principle

G0 should not start from the full Unitree locomotion difficulty. The current robot has much lower torque margins than Unitree G1, different joint axes, different feet, and confirmed foot bodies `l_foot_link` and `r_foot_link`. Unitree G1 actuator parameters, default pose, body names, base height, and reward weights must not be copied onto G0.

## Stage A: Standing And Balance Sanity

Goal: stand without obvious collapse.

Settings:

- command fixed to zero or extremely small range
- plane terrain
- no push
- no added base mass randomization
- friction range narrow, for example `0.8-1.0`
- reset yaw small or zero
- joint reset velocity near zero
- loose orientation/height termination while debugging, but keep it strict enough to catch falls

Watch:

- survive steps near 500 in `scripts/debug/sweep_g0_standing_pose.py`
- `abs(final_root_pitch_deg) < 10-15 deg`
- `pitch_slope_deg_per_step` near zero
- `min_root_z > 0.20`
- ankle/knee torque ratio not pinned near 1.0
- left/right foot force not severely imbalanced

Current status: not passed. Best tested candidate survived 200/500 steps and hit ankle torque ratio 1.0.

## Stage B: Slow Walking

Goal: learn low-speed forward stepping after standing is sane.

Settings:

- `lin_vel_x` around `0.0-0.3 m/s`
- `lin_vel_y` near zero
- `ang_vel_z` zero or very small
- action scale kept conservative, current `0.12` is appropriate for first tests
- keep terrain as plane
- keep mass randomization and pushes disabled
- keep gait/contact rewards light until contact behavior is believable

Exit criteria:

- policy can stand at zero command
- forward command produces repeated steps instead of falling immediately
- play rollout survives full episodes more often than not

## Stage C: Normal Velocity Tracking

Goal: widen velocity tracking without jumping straight to the reference difficulty.

Settings:

- gradually expand `lin_vel_x`
- add small yaw command after forward walking is stable
- later add small lateral command
- restore observation noise only after early learning is stable
- reintroduce friction randomization before mass or push randomization

Suggested progression:

```text
lin_vel_x: 0.0-0.3 -> -0.1-0.5 -> -0.2-0.8 -> -0.5-1.0
lin_vel_y: 0.0     -> +/-0.1  -> +/-0.2  -> +/-0.3
ang_vel_z: 0.0     -> +/-0.1  -> +/-0.15 -> +/-0.2
```

## Stage D: Align With unitree_rl_lab Locomotion Difficulty

Goal: approach the reference locomotion setup after G0 has its own stable baseline.

Add back gradually:

- terrain generator curriculum
- broader friction randomization
- base mass randomization
- push perturbations
- stricter orientation/height termination
- wider command range
- more complete gait and foot clearance reward pressure

Keep G0-specific:

- foot bodies: `l_foot_link`, `r_foot_link`
- G0 base height target from measured standing height
- G0 default pose from zero-action sweeps
- G0 actuator effort/velocity limits from `g0_actuators.py`
- G0-specific action scale and reward weights
