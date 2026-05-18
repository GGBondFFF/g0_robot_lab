# G0 Deploy Sim2sim Validation Matrix Report

## Scope

- model: `mujoco/g0.xml`
- deploy_cfg: `logs/sim2sim/g0_deploy/params/deploy.yaml`
- policy: `logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/exported/policy.pt`
- steps per case: `500`
- output_dir: `logs/sim2sim/g0_deploy/validation_matrix_500`

This report validates the deploy-style MuJoCo runtime. It does not tune policy quality or physics parameters.

## Case Matrix

| case | run | check | action max | action sat | torque max | torque sat | vel exceeded | root h min/final | contacts mean/max | foot force mean/max | early fall |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `zero_action_position_c0_c0_c0` | `True` | `True` | 0 | 0 | 0.5789 | 0.0001818 | 0 | 0.02755/0.04007 | 4.674/9 | 1.633/13.65 | `True` |
| `zero_action_position_c0p05_c0_c0` | `True` | `True` | 0 | 0 | 0.5789 | 0.0001818 | 0 | 0.02755/0.04007 | 4.674/9 | 1.633/13.65 | `True` |
| `zero_action_position_c0p1_c0_c0` | `True` | `True` | 0 | 0 | 0.5789 | 0.0001818 | 0 | 0.02755/0.04007 | 4.674/9 | 1.633/13.65 | `True` |
| `zero_action_position_c0_c0_c0p1` | `True` | `True` | 0 | 0 | 0.5789 | 0.0001818 | 0 | 0.02755/0.04007 | 4.674/9 | 1.633/13.65 | `True` |
| `zero_action_pd_torque_c0_c0_c0` | `True` | `True` | 0 | 0 | 0.7618 | 0.0001818 | 0 | 0.03015/0.04007 | 4.632/7 | 1.801/13.56 | `True` |
| `zero_action_pd_torque_c0p05_c0_c0` | `True` | `True` | 0 | 0 | 0.7618 | 0.0001818 | 0 | 0.03015/0.04007 | 4.632/7 | 1.801/13.56 | `True` |
| `zero_action_pd_torque_c0p1_c0_c0` | `True` | `True` | 0 | 0 | 0.7618 | 0.0001818 | 0 | 0.03015/0.04007 | 4.632/7 | 1.801/13.56 | `True` |
| `zero_action_pd_torque_c0_c0_c0p1` | `True` | `True` | 0 | 0 | 0.7618 | 0.0001818 | 0 | 0.03015/0.04007 | 4.632/7 | 1.801/13.56 | `True` |
| `policy_position_c0_c0_c0` | `True` | `True` | 17.64 | 0.4053 | 1.201 | 0.01291 | 0 | 0.02918/0.03409 | 2.306/7 | 10.19/21.41 | `True` |
| `policy_position_c0p05_c0_c0` | `True` | `True` | 23.39 | 0.373 | 1.421 | 0.01655 | 0 | 0.0327/0.03666 | 2.302/5 | 11.13/19.46 | `True` |
| `policy_position_c0p1_c0_c0` | `True` | `True` | 3.512 | 0.1735 | 1.874 | 0.01091 | 0 | 0.2165/0.2165 | 2.75/6 | 15.16/22.09 | `False` |
| `policy_position_c0_c0_c0p1` | `True` | `True` | 2.147 | 0.1386 | 0.6735 | 0.001455 | 0 | 0.2292/0.2298 | 2.786/5 | 15.69/19.75 | `False` |
| `policy_pd_torque_c0_c0_c0` | `True` | `True` | 16.63 | 0.2718 | 1.283 | 0.008182 | 0 | 0.03247/0.03628 | 2.514/8 | 13.23/20.89 | `True` |
| `policy_pd_torque_c0p05_c0_c0` | `True` | `True` | 9.039 | 0.2113 | 1.439 | 0.006636 | 0 | 0.03638/0.0364 | 2.888/7 | 13.96/20.68 | `True` |
| `policy_pd_torque_c0p1_c0_c0` | `True` | `True` | 2.127 | 0.1546 | 0.54 | 9.091e-05 | 0 | 0.2292/0.2292 | 3.088/5 | 15.76/19.8 | `False` |
| `policy_pd_torque_c0_c0_c0p1` | `True` | `True` | 20.5 | 0.2759 | 0.9196 | 0.007091 | 0 | 0.03102/0.03469 | 2.522/6 | 12.98/20.12 | `True` |

## Summary

- cases run: `16`
- cases with runner success: `16/16`
- cases with checker success: `16/16`
- most stable case by root height / velocity / torque / contact score: `policy_position_c0_c0_c0p1`
- least stable case by root height / velocity / torque / contact score: `zero_action_position_c0_c0_c0`
- highest action saturation case: `policy_position_c0_c0_c0`
- highest torque saturation case: `policy_position_c0p05_c0_c0`
- early fall cases: `['zero_action_position_c0_c0_c0', 'zero_action_position_c0p05_c0_c0', 'zero_action_position_c0p1_c0_c0', 'zero_action_position_c0_c0_c0p1', 'zero_action_pd_torque_c0_c0_c0', 'zero_action_pd_torque_c0p05_c0_c0', 'zero_action_pd_torque_c0p1_c0_c0', 'zero_action_pd_torque_c0_c0_c0p1', 'policy_position_c0_c0_c0', 'policy_position_c0p05_c0_c0', 'policy_pd_torque_c0_c0_c0', 'policy_pd_torque_c0p05_c0_c0', 'policy_pd_torque_c0_c0_c0p1']`
- velocity-limit exceeded cases: `[]`

## Credibility Assessment

- sim2sim started: `True`
- sim2sim smoke pass by this matrix: `False`
- sim2sim credible pass: `False`

Smoke-pass criteria used here:

- policy position and policy pd_torque both run at least 500 steps for all matrix commands
- policy root height stays at or above `0.16 m`
- velocity-limit exceeded joints are `0/22`
- torque saturation ratio stays at or below `5%`
- action saturation ratio stays at or below `30%`
- no NaN/Inf is present

- policy root-height criterion: `False`
- policy torque saturation criterion: `True`
- policy action saturation criterion: `False`
- policy velocity-limit criterion: `True`
- finite-data criterion: `True`
- zero-action early-fall cases: `['zero_action_position_c0_c0_c0', 'zero_action_position_c0p05_c0_c0', 'zero_action_position_c0p1_c0_c0', 'zero_action_position_c0_c0_c0p1', 'zero_action_pd_torque_c0_c0_c0', 'zero_action_pd_torque_c0p05_c0_c0', 'zero_action_pd_torque_c0p1_c0_c0', 'zero_action_pd_torque_c0_c0_c0p1']`
- controlled root-frame diagnostics: `pass`
- projected_gravity formula corrected this round: `False`
- base_ang_vel formula corrected this round: `True`

Current interpretation:

- Passing policy cases support active-control matrix execution, but smoke pass remains false because some policy cases fail the root-height/action-saturation criteria.
- Zero-action collapse does not by itself imply policy failure; it says the default target pose is not a static MuJoCo equilibrium under the current dynamics/contact model.
- Credible pass is still false because it requires longer 1000-step runs, explainable position/pd_torque differences, and deeper actuator/contact behavior reports.
- Cases supporting smoke direction: `policy_position_c0p1_c0_c0`, `policy_position_c0_c0_c0p1`, `policy_pd_torque_c0p1_c0_c0`.
- Cases blocking smoke pass: `policy_position_c0_c0_c0`, `policy_position_c0p05_c0_c0`, `policy_pd_torque_c0_c0_c0`, `policy_pd_torque_c0p05_c0_c0`, `policy_pd_torque_c0_c0_c0p1`.

## Position Vs PD Torque

- `zero_action` command `[0.0, 0.0, 0.0]`: root_min position/pd_torque `0.02755` / `0.03015`, torque_sat `0.0001818` / `0.0001818`, foot_force_max `13.65` / `13.56`
- `zero_action` command `[0.05, 0.0, 0.0]`: root_min position/pd_torque `0.02755` / `0.03015`, torque_sat `0.0001818` / `0.0001818`, foot_force_max `13.65` / `13.56`
- `zero_action` command `[0.1, 0.0, 0.0]`: root_min position/pd_torque `0.02755` / `0.03015`, torque_sat `0.0001818` / `0.0001818`, foot_force_max `13.65` / `13.56`
- `zero_action` command `[0.0, 0.0, 0.1]`: root_min position/pd_torque `0.02755` / `0.03015`, torque_sat `0.0001818` / `0.0001818`, foot_force_max `13.65` / `13.56`
- `policy` command `[0.0, 0.0, 0.0]`: root_min position/pd_torque `0.02918` / `0.03247`, torque_sat `0.01291` / `0.008182`, foot_force_max `21.41` / `20.89`
- `policy` command `[0.05, 0.0, 0.0]`: root_min position/pd_torque `0.0327` / `0.03638`, torque_sat `0.01655` / `0.006636`, foot_force_max `19.46` / `20.68`
- `policy` command `[0.1, 0.0, 0.0]`: root_min position/pd_torque `0.2165` / `0.2292`, torque_sat `0.01091` / `9.091e-05`, foot_force_max `22.09` / `19.8`
- `policy` command `[0.0, 0.0, 0.1]`: root_min position/pd_torque `0.2292` / `0.03102`, torque_sat `0.001455` / `0.007091`, foot_force_max `19.75` / `20.12`

## Zero-Action Vs Policy

- `position` command `[0.0, 0.0, 0.0]`: action_sat zero/policy `0` / `0.4053`, root_min `0.02755` / `0.02918`, torque_sat `0.0001818` / `0.01291`
- `position` command `[0.05, 0.0, 0.0]`: action_sat zero/policy `0` / `0.373`, root_min `0.02755` / `0.0327`, torque_sat `0.0001818` / `0.01655`
- `position` command `[0.1, 0.0, 0.0]`: action_sat zero/policy `0` / `0.1735`, root_min `0.02755` / `0.2165`, torque_sat `0.0001818` / `0.01091`
- `position` command `[0.0, 0.0, 0.1]`: action_sat zero/policy `0` / `0.1386`, root_min `0.02755` / `0.2292`, torque_sat `0.0001818` / `0.001455`
- `pd_torque` command `[0.0, 0.0, 0.0]`: action_sat zero/policy `0` / `0.2718`, root_min `0.03015` / `0.03247`, torque_sat `0.0001818` / `0.008182`
- `pd_torque` command `[0.05, 0.0, 0.0]`: action_sat zero/policy `0` / `0.2113`, root_min `0.03015` / `0.03638`, torque_sat `0.0001818` / `0.006636`
- `pd_torque` command `[0.1, 0.0, 0.0]`: action_sat zero/policy `0` / `0.1546`, root_min `0.03015` / `0.2292`, torque_sat `0.0001818` / `9.091e-05`
- `pd_torque` command `[0.0, 0.0, 0.1]`: action_sat zero/policy `0` / `0.2759`, root_min `0.03015` / `0.03102`, torque_sat `0.0001818` / `0.007091`

## Likely Failure Causes Ranked

1. Root height drops below the fall heuristic in multiple zero-action cases, so passive/default-pose settling and contact dynamics need inspection before interpreting policy quality.
2. Policy action saturation is present; check policy observation conventions, action scale, and exported action processing.
3. Raw PD torque exceeds effort limits in some samples; inspect actuator limits and PD implementation before changing gains.
4. No joint exceeded `velocity_limit_sim` in this matrix.
5. Contact-force variation remains a physics fidelity signal; inspect contact/friction/solver only with controlled comparisons, not policy tuning.

## Failure-Window Analysis

Follow-up policy-only window analysis was run on the 500-step matrix:

```text
logs/sim2sim/g0_deploy/failure_window_analysis.md
docs/g0_policy_failure_window_analysis.md
```

Summary:

- Failed policy cases: `policy_pd_torque_c0_c0_c0`, `policy_pd_torque_c0_c0_c0p1`, `policy_pd_torque_c0p05_c0_c0`, `policy_position_c0_c0_c0`, `policy_position_c0p05_c0_c0`.
- Stable policy cases: `policy_pd_torque_c0p1_c0_c0`, `policy_position_c0_c0_c0p1`, `policy_position_c0p1_c0_c0`.
- Failure steps: `421`, `414`, `449`, `324`, `355` respectively.
- The most common pre-failure signal is `action saturation first`.
- Contact loss appears after root-height failure in the failed cases, so it is not the earliest detected precursor.
- Torque saturation is low and not the primary signal.
- Velocity limits remain non-suspect in this matrix.
- Aggregate suspicious joints: `r_knee_pitch_joint`, `l_hip_pitch_joint`, `r_hip_pitch_joint`, `l_ankle_pitch_joint`, `l_knee_pitch_joint`, `r_ankle_roll_joint`.

## Next Steps

- If action or torque saturation grows, inspect policy/action scale/exported actuator limits before touching gains.
- If root-frame signals look inconsistent, run controlled frame diagnostics for `base_ang_vel` and `projected_gravity`.
- If contact force is abnormal, compare contact geometry, friction, and solver behavior without changing formal foot mesh.
- If `pd_torque` and `position` diverge strongly, isolate actuator implementation and timing differences.
- Regenerate this matrix after each narrow sim2sim fix and compare only one variable at a time.
