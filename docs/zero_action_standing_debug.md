# Zero-Action Standing Debug

This document summarizes the current zero-action standing debug conclusions for `G0-Velocity-v0`.

## Current Conclusion

Do not start large-scale training directly from the current state.

Zero-action standing is still a key baseline check. If the robot cannot hold the default pose with zero policy action, PPO training is likely to learn around asset/contact artifacts or immediate fall recovery instead of clean locomotion.

## Observed Failure Mode

Previous zero-action tests showed that the robot tips forward and eventually fails posture/height checks.

The current best simple two-decimal debug pose is:

```text
root_z = 0.23
hip_pitch = 0.20
knee_pitch = 0.34
ankle_pitch = 0.14
```

This pose is useful as a debug baseline, but it is not considered stable.

## Effort Scale Debug

The debug script can temporarily apply `--effort-scale` to actuator effort limits. This is diagnostic only.

Important conclusion:

```text
Even when temporary effort scaling reduces saturation, the robot can still tip and fail.
```

So the issue is not necessarily only insufficient torque. Torque limits may be one bottleneck at nominal settings, especially around ankle pitch, but the remaining fall behavior points to geometry, contact, COM, height, or inertia as important next checks.

Do not convert debug-only effort scaling into permanent hardware parameters.

## Why Termination Removal Is Not A Fix

Disabling `base_height` or `bad_orientation` termination can help inspect the natural fall path, but it does not make the standing pose valid.

If the robot continues to rotate and collapse after termination is disabled, the termination condition is reporting the failure rather than causing it.

Do not solve the problem by permanently deleting termination conditions.

## Next Checks

Before large PPO training, inspect:

- COM location and COM motion during early fall
- support polygon relative to the feet
- foot collision geometry
- foot contact patch size and shape
- foot link frame height versus actual sole/contact height
- root initial height
- whether feet start suspended or penetrated
- contact forces and contact point count at reset
- body mass and inertial parameters
- left/right pitch-axis sign convention and default pose symmetry

## Guardrails

Do not hide the problem by:

- deleting fall-related terminations permanently
- permanently enlarging real hardware torque limits
- weakening rewards until falling becomes acceptable
- starting a long training run before zero-action standing is understood

The next useful work should be physical consistency and contact debugging, then small train/play checks once the standing baseline is healthier.
