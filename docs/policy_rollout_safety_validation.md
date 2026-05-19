# Policy Rollout Safety Validation

This document records the Isaac Lab policy rollout safety diagnostic for `g0_robot_lab`.

This stage is for Isaac Lab policy-behavior and numerical-safety diagnostics only. It does not prove real-robot deployment readiness, does not authorize hardware bring-up, and must not be interpreted as permission to send LowCmd or real motor commands.

## Scope

- Environment: Isaac Lab only
- Task: `G0-Velocity-v0`
- Policy source: raw RSL-RL checkpoint loaded through the same runner path as `scripts/rsl_rl/play.py`
- Hardware: forbidden
- LowCmd: forbidden
- MuJoCo sim2sim: out of scope

## Fixed Checkpoint

Checkpoint path:

```text
logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/model_9999.pt
```

Expected SHA256:

```text
1dc0c434a4b991eaaa435a21b9d4265e0267eb781b69b132bd75a0b5883928cd
```

The diagnostic must verify both checkpoint existence and SHA256 before rollout.

## Fixed Environment Conditions

The rollout uses the same fixed standing-style environment conditions as the zero-action release gate:

- `scene.num_envs = 1` by default
- root pose initialized at `z = 0.233`
- identity root rotation
- zero root linear and angular velocity
- zero initial joint velocity
- randomization disabled
- external disturbance events disabled
- curriculum disabled
- standing command fixed with zero commanded linear and angular velocity
- observation corruption disabled
- reward-only `undesired_contacts` disabled

These settings are intended to make the diagnostic deterministic and comparable to the existing release-gate standing baseline.

## Phase 1 Contract Rules

Phase 1 is a diagnostic contract, not a physical-success gate.

Hard fail only on:

- checkpoint missing
- checkpoint SHA256 mismatch
- environment construction failure
- runner or policy load failure
- NaN or Inf in observations
- NaN or Inf in actions
- action shape mismatch

Do not hard-fail in Phase 1 on:

- falls
- resets
- `time_out`
- root height variation
- effort saturation
- raw policy action outside `[-1, 1]`
- slightly negative joint limit margin

Those remain diagnostics and warnings only in this phase.

## Results

Phase 1 diagnostic runs completed successfully for 500, 1000, and 2000 steps.

| Steps | Contract Result | Raw Policy Action Range | Raw Out of Range Count | Clipped Action Range | Root Z Min | Root Z Max | Root Z Mean | base_height | bad_orientation | time_out | Resets | Effort Ratio Max | Effort Steps > 0.9 | Joint Limit Margin Min |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | PASS | `[-2.9596, 2.4235]` | 1814 | `[-1.0, 1.0]` | 0.22861 | 0.23564 | 0.23253 | 0 | 0 | 0 | 0 | 1.0 | 4 | -0.05134 |
| 1000 | PASS | `[-2.9596, 2.4360]` | 3646 | `[-1.0, 1.0]` | 0.22855 | 0.23564 | 0.23248 | 0 | 0 | 1 | 1 | 1.0 | 4 | not separately captured in summary table |
| 2000 | PASS | `[-2.9596, 2.4360]` | 7308 | `[-1.0, 1.0]` | 0.22853 | 0.23564 | 0.23245 | 0 | 0 | 2 | 2 | 1.0 | 5 | not separately captured in summary table |

## Raw Policy Action vs Clipped Action

The diagnostic now reports two action views:

- `raw_policy_action`: direct policy output before environment clipping
- `clipped_action`: the equivalent `[-1, 1]` clipped action used for interpretation and target reconstruction

This distinction matters because the current policy routinely emits raw values outside `[-1, 1]`, while the Isaac Lab environment wrapper clips actions internally before they become effective joint-position commands.

Current implication:

- raw policy output is not directly safe to pass into any deployment path
- deployment must apply equivalent clipping before converting actions into joint targets or any future LowCmd-compatible representation

This clipping requirement must remain explicit in any deployment-facing implementation.

## Time-Out Interpretation

The `time_out` resets observed at 1000 and 2000 steps are not physical failure signals by themselves.

In this diagnostic context, `time_out` means the environment episode horizon was reached and the rollout reset normally. The 1000-step run saw one such reset, and the 2000-step run saw two. These do not indicate a fall, base-height violation, or bad-orientation event.

## Phase 2.5 Worst-Case Diagnostic Findings

Phase 2.5 added worst-case step and joint reporting to the diagnostic output. The 500-, 1000-, and 2000-step runs all stayed `PASS (contract)`, and the worst-case findings were:

- raw policy action minimum: step `5`, `r_ankle_roll_joint`, value `-2.9596`
- raw policy action maximum: step `892`, `r_knee_pitch_joint`, value `2.4360`
- effort ratio worst: step `0`, `l_knee_pitch_joint`, value `1.0`
- joint limit margin worst: step `0`, `r_shoulder_roll_joint`, value `-0.05134`
- target delta worst: step `0`, `r_elbow_pitch_joint`, value `0.86026`

Interpretation:

- The raw action worst cases confirm that any deployment-facing path must apply clipping before converting policy output into any LowCmd-compatible representation.
- The effort-ratio worst case appearing at step `0` suggests likely initial transient or reset-alignment behavior. Phase 3 should separate that startup behavior from sustained saturation before enabling any hard gate.
- The joint-limit-margin worst case also appearing at step `0` suggests that Phase 3 should not blindly hard-fail on this value until reset-time joint and limit interpretation is confirmed.
- The target-delta worst case at step `0` suggests that the initial target jump should be analyzed before it becomes a hard gate.
- Phase 3 thresholds remain inactive.

## Phase 3 Release Gate

A conservative 500-step pytest release gate has now been added for Isaac Lab policy rollout safety:

- it runs the fixed `G0-Velocity-v0` checkpoint rollout for 500 steps in Isaac Lab only
- it checks the rollout contract plus no physical-failure signals for this 500-step gate
- it hard-fails on checkpoint identity problems, environment or policy-load failures, non-finite observations or actions, action shape mismatch, base-height termination, bad-orientation termination, reset count above zero, or clipped-action output outside `[-1, 1]` except for tiny numerical tolerance

The following remain diagnostic-only and are not hard failures in this gate:

- raw policy action outside `[-1, 1]`
- effort ratio reaching `1.0`
- effort ratio steps above `0.9`
- slightly negative joint-limit margin
- target-delta worst-case values

Passing this release gate still does not mean real-robot deployment readiness, does not authorize LowCmd transmission, and does not permit hardware bring-up.

## Remaining Risks

The Phase 1 contract passed, but the following risks remain open:

1. Raw action out of range

The policy emits many raw actions outside `[-1, 1]`. Isaac Lab clipping prevents this from becoming an immediate contract failure here, but any deployment path must enforce equivalent clipping before producing joint targets.

2. Effort ratio saturation

The effort ratio briefly reaches `1.0`. This is reported as a warning only in Phase 1, but it is a real signal that some joints may be touching simulated effort limits.

3. Joint limit margin slightly negative

After correcting the margin formula to:

```text
lower_margin = joint_pos - lower_limit
upper_margin = upper_limit - joint_pos
margin = min(lower_margin, upper_margin)
```

the minimum observed margin remained slightly negative. This is no longer a sign-convention bug. Before Phase 3 turns this into any kind of hard threshold, the diagnostic should identify the worst joint and step explicitly so the result can be interpreted correctly.

## Proposed Phase 3 Candidate Thresholds

These are proposed calibration candidates only. They are not active yet, are not part of default CI, and must not be treated as release gates yet.

Candidate thresholds:

- NaN or Inf in observations: `0`
- NaN or Inf in actions: `0`
- action shape mismatch: `0`
- raw policy action outside `[-1, 1]`: report-only for now; not active
- effort ratio max: candidate ceiling `<= 1.05`
- effort ratio steps above `0.9`:
  - 500 steps: candidate `<= 50`
  - 1000 steps: candidate `<= 100`
  - 2000 steps: candidate `<= 200`
- `time_out`: report-only while horizon-reset interpretation remains the same
- `base_height`: candidate hard threshold deferred
- `bad_orientation`: candidate hard threshold deferred
- `joint_limit_margin_min`: still inactive until reset-time interpretation is confirmed
- `target_delta_worst_value`: still inactive until initial-step target-jump behavior is understood

## Safety Statement

Passing this diagnostic does not indicate readiness for real-robot deployment, hardware bring-up, or LowCmd transmission.
