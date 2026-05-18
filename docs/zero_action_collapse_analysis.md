# Zero-Action Collapse Analysis

## Context

This note explains the current zero-action behavior in the deploy-style MuJoCo validation matrix. It does not tune the model or policy.

Zero-action means:

```text
policy_action = 0
target_joint_pos = default_joint_pos
q_des = target_joint_pos
dq_des = 0
tau_ff = 0
```

## Observed Result

In the 500-step validation matrix:

- All zero-action cases ran and passed the checker.
- All zero-action cases triggered the early-fall heuristic.
- `target_joint_pos` exactly equals `default_joint_pos`.
- `max_target_default_abs_err = 0`.
- `max_abs_policy_action = 0`.
- `velocity-limit exceeded joints = 0/22`.
- No NaN/Inf was reported.

Representative zero-action position case:

```text
case: zero_action_position_c0_c0_c0
root_height min/max/final: 0.027547 / 0.23143 / 0.0400742
pd_tau_cmd max abs: 0.578934
torque saturation ratio: 0.000181818
joint_vel max abs: 3.03837
foot_contact_force_norm min/mean/max: 0 / 1.63322 / 13.6471
```

Representative zero-action pd_torque cases show the same qualitative behavior:

```text
root_height min/final: about 0.03015 / 0.04007
pd_tau_cmd max abs: about 0.7618
velocity-limit exceeded joints: 0/22
```

## Comparison With Policy Rollout

The policy rollouts are not simply "zero-action plus noise." The policy actively changes target joint positions and can compensate for some of the passive/default-pose dynamics.

In the 500-step matrix:

- Some policy cases remain above the `0.16 m` danger threshold.
- Some policy cases still fall below the threshold after the corrected `base_ang_vel` frame convention.
- Policy action saturation is present, especially in low/zero forward command cases.

This means zero-action collapse is a useful dynamics/contact/default-pose signal, but it is not proof of policy failure.

## What Zero-Action Collapse Does And Does Not Mean

It does mean:

- The current MuJoCo default target pose is not a static equilibrium under the current model dynamics/contact response.
- Root/contact settling is significant.
- Default-pose behavior needs to be explained before treating zero-action as a standing baseline.

It does not mean:

- The policy is necessarily bad.
- The deploy runner is broken.
- The action bridge is wrong. The checker confirms `target_joint_pos = default_joint_pos` for zero action.
- The fix should be kp/damping/friction/solver/root-height tuning.

## Most Likely Causes

1. The default pose is not a static equilibrium in the current MuJoCo dynamics.
2. Root/contact settling pulls the robot down even when joint targets equal default pose.
3. Actuator/contact/inertia differences remain between Isaac/PhysX and MuJoCo.
4. The policy can actively compensate for some root-height loss, while zero-action cannot.

## Current Debug Boundary

Do not tune parameters from this analysis alone. The next useful work is controlled comparison:

- Compare default-pose joint torques and root acceleration in Isaac and MuJoCo.
- Inspect contact force timing during the first 100 zero-action steps.
- Compare position mode and pd_torque mode under identical default-pose targets.
- Keep foot collision geometry unchanged while diagnosing contact behavior.
