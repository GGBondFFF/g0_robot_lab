# G0 Zero-Action Dynamics Compare Report

## Context

- Branch: `structure/mujoco-sim2sim-layout`
- Test time: `2026-05-15 15:48:50 CST`
- Isaac file: `logs/sim2sim/isaac_zero_action_golden_io.npz`
- MuJoCo file: `logs/sim2sim/mujoco_zero_action_rollout.npz`
- Goal: compare zero-action dynamics terms without treating differences as policy failure.

This pass is diagnostic only. No foot collision geometry, actuator gains, damping, friction, or solver parameters were tuned.

## Command

```bash
TERM=xterm conda run -n g0_isaaclab python scripts/sim2sim/compare_zero_action_dynamics.py \
  --isaac logs/sim2sim/isaac_zero_action_golden_io.npz \
  --mujoco logs/sim2sim/mujoco_zero_action_rollout.npz \
  --output logs/sim2sim/zero_action_dynamics_compare_report.md
```

## Main Compare

```text
joint_pos mean/max abs error: 0.0100061 / 0.0607303
joint_vel mean/max abs error: 0.0523146 / 3.18014
root_quat mean/max abs error: 0.237017 / 0.696053
base_ang_vel mean/max abs error: 0.214718 / 4.77339
projected_gravity mean/max abs error: 0.240835 / 0.999848
command mean/max abs error: 0.0566347 / 0.0899589
target_joint_pos mean/max abs error: 0 / 0
root_height mean/max abs error: 0.0723206 / 0.192692
```

The action bridge remains exact through `target_joint_pos`. The remaining differences are dynamics/state terms.

## MuJoCo Diagnostics

The latest MuJoCo rollout records additional diagnostics:

```text
qacc shape: (100, 28)
qacc min/mean/max: -160.078 / -0.168248 / 358.986
joint_acc shape: (100, 22)
joint_acc min/mean/max: -160.078 / -0.219228 / 164.065
contact_count min/mean/max: 0 / 3.37 / 9
max_contact_force_norm min/mean/max: 0 / 10.0759 / 66.815
foot_ground_contact_count min/mean/max: 0 / 2.23 / 5
root_height min/mean/max: 0.0275471 / 0.154425 / 0.23143
qacc finite: True
joint_acc finite: True
```

No QACC warning was observed during this 100-step zero-action rollout, and no NaN/Inf was found in `qacc` or `joint_acc`.

## Missing Cross-Simulator Terms

The current Isaac golden file does not contain:

```text
qacc
joint_acc
contact_count
max_contact_force_norm
foot_ground_contact_count
contact_force
foot_contact_force
```

`dump_isaac_golden_io.py` now attempts to collect `root_height`, `joint_acc`, and contact force data when available, but the existing golden file was produced before those diagnostics were added. If Isaac-side contact/acceleration comparison is needed, regenerate the golden file through Isaac Lab.

## Interpretation

The largest remaining differences are root and frame-sensitive terms:

1. `projected_gravity` and `root_quat` differ substantially, so root orientation and frame conventions still need controlled diagnostics.
2. `base_ang_vel` differs, which may be a frame convention issue and/or dynamics response issue.
3. `root_height` differs, suggesting root initialization/contact settling is still not identical.
4. Contact counts and contact forces are only available on the MuJoCo side in the current files, so contact-force fidelity cannot yet be judged cross-simulator.
5. Velocity limits were not approached in zero-action, so they are not the immediate source of these differences.

## Recommendation

Regenerate Isaac golden I/O with the new optional diagnostics, then compare root height, joint acceleration, and contact force terms again. Keep contact/friction/solver tuning separate from observation-frame debugging so each mismatch has a clear cause.
