# G0 Sim2sim Credibility Criteria

## Purpose

G0 is now in deploy-style MuJoCo sim2sim validation. This document defines how to label the current state without overstating it.

## A. Sim2sim Started

This level means the deploy-style chain exists and can run.

Current G0 status: satisfied.

Evidence:

- `deploy.yaml` can be exported from `G0-Velocity-v0`.
- The deploy runner can load `deploy.yaml`.
- The deploy runner can run zero-action rollout.
- The deploy runner can run TorchScript `policy.pt` rollout.
- Both `position` and `pd_torque` modes run.
- The validation matrix runner runs.
- The rollout checker runs.
- The matrix has produced `16/16` runner OK and `16/16` checker OK at 200 and 500 steps.

This level does not mean the sim2sim result is physically credible.

## B. Sim2sim Smoke Pass

This level means the deploy-style chain is stable enough for short active-policy debugging.

Minimum criteria:

- Policy `position` and policy `pd_torque` both run at least 500 steps.
- Root height stays above the danger threshold, currently `0.16 m`, for policy cases.
- `velocity_limit_sim` exceeded joints remain `0/22`.
- Raw torque saturation is not sustained, with matrix-level torque saturation ratio at or below `5%`.
- Policy action saturation is not sustained, with matrix-level action saturation ratio at or below `30%`.
- Rollouts contain no NaN/Inf.
- No MuJoCo QACC warning is observed in the tested matrix.
- Foot contact force does not show obvious explosion.
- `projected_gravity` and `base_ang_vel` controlled diagnostics do not show a frame convention error.

Current G0 status: not yet satisfied.

Reason:

- 500-step policy cases run, but several policy cases fall below the `0.16 m` root-height threshold.
- 500-step policy action saturation exceeds `30%` in some cases.
- Velocity limits and torque saturation are acceptable, and controlled frame diagnostics now pass.

## C. Sim2sim Credible Pass

This level means the deploy-style result is explainable and reproducible enough to guide sim2real or deeper physics matching work.

Higher criteria:

- Policy `position` and `pd_torque` both run at least 1000 steps.
- Multiple commands keep stable root height.
- Controlled `projected_gravity` diagnostic passes.
- Controlled `base_ang_vel` diagnostic passes.
- Position vs `pd_torque` differences are explainable.
- Zero-action collapse is explained and does not get confused with policy failure.
- Contact force, root height, joint velocity, action saturation, and torque saturation all have reports.
- Failures can be categorized as interface mismatch, observation frame mismatch, actuator implementation mismatch, contact/root/dynamics mismatch, or policy limitation.

Current G0 status: not yet satisfied.

Reason:

- The 1000-step matrix has not been run.
- Several 500-step policy cases still fail root-height smoke criteria.
- Action saturation and position/pd_torque command-dependent differences still need explanation.

## Failure Category Vocabulary

Use these categories when interpreting results:

- Interface mismatch: shape/order/scale/action-processing/logging disagreement.
- Observation frame mismatch: `root_quat`, `projected_gravity`, `base_ang_vel`, or history layout disagreement.
- Actuator implementation mismatch: position actuator vs LowCmd-style PD torque behavior, clipping, limits, or timing.
- Contact/root/dynamics mismatch: default pose settling, contact force, root height, inertia, or collision response mismatch.
- Policy limitation: only considered after interface, frame, actuator, and contact/dynamics evidence is controlled.
