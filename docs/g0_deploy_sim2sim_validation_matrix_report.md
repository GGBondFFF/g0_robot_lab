# G0 Deploy Sim2sim Validation Matrix Report

## Scope

- model: `mujoco/g0.xml`
- deploy_cfg: `logs/sim2sim/g0_deploy/params/deploy.yaml`
- policy: `logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/exported/policy.pt`
- steps per case: `200`
- output_dir: `logs/sim2sim/g0_deploy/validation_matrix`

This report validates the deploy-style MuJoCo runtime. It does not tune policy quality or physics parameters.

## Case Matrix

| case | run | check | action max | action sat | torque max | torque sat | vel exceeded | root h min/final | contacts mean/max | foot force mean/max | early fall |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `zero_action_position_c0_c0_c0` | `True` | `True` | 0 | 0 | 0.5789 | 0.0004545 | 0 | 0.02755/0.04007 | 4.185/9 | 3.688/13.65 | `True` |
| `zero_action_position_c0p05_c0_c0` | `True` | `True` | 0 | 0 | 0.5789 | 0.0004545 | 0 | 0.02755/0.04007 | 4.185/9 | 3.688/13.65 | `True` |
| `zero_action_position_c0p1_c0_c0` | `True` | `True` | 0 | 0 | 0.5789 | 0.0004545 | 0 | 0.02755/0.04007 | 4.185/9 | 3.688/13.65 | `True` |
| `zero_action_position_c0_c0_c0p1` | `True` | `True` | 0 | 0 | 0.5789 | 0.0004545 | 0 | 0.02755/0.04007 | 4.185/9 | 3.688/13.65 | `True` |
| `zero_action_pd_torque_c0_c0_c0` | `True` | `True` | 0 | 0 | 0.7618 | 0.0004545 | 0 | 0.03015/0.04007 | 4.08/7 | 4.106/13.56 | `True` |
| `zero_action_pd_torque_c0p05_c0_c0` | `True` | `True` | 0 | 0 | 0.7618 | 0.0004545 | 0 | 0.03015/0.04007 | 4.08/7 | 4.106/13.56 | `True` |
| `zero_action_pd_torque_c0p1_c0_c0` | `True` | `True` | 0 | 0 | 0.7618 | 0.0004545 | 0 | 0.03015/0.04007 | 4.08/7 | 4.106/13.56 | `True` |
| `zero_action_pd_torque_c0_c0_c0p1` | `True` | `True` | 0 | 0 | 0.7618 | 0.0004545 | 0 | 0.03015/0.04007 | 4.08/7 | 4.106/13.56 | `True` |
| `policy_position_c0_c0_c0` | `True` | `True` | 2.149 | 0.1368 | 0.6803 | 0.001136 | 0 | 0.2294/0.2318 | 2.795/5 | 15.69/19.43 | `False` |
| `policy_position_c0p05_c0_c0` | `True` | `True` | 2.12 | 0.1445 | 0.5927 | 0.0004545 | 0 | 0.2289/0.2309 | 2.78/6 | 15.54/19.92 | `False` |
| `policy_position_c0p1_c0_c0` | `True` | `True` | 2.091 | 0.1705 | 0.9588 | 0.0125 | 0 | 0.2289/0.23 | 2.78/5 | 15.15/21.1 | `False` |
| `policy_position_c0_c0_c0p1` | `True` | `True` | 2.124 | 0.1373 | 0.6662 | 0.001136 | 0 | 0.2294/0.2312 | 2.78/4 | 15.7/21.3 | `False` |
| `policy_pd_torque_c0_c0_c0` | `True` | `True` | 2.086 | 0.1345 | 0.54 | 0.0002273 | 0 | 0.2299/0.2314 | 2.85/5 | 15.94/19.78 | `False` |
| `policy_pd_torque_c0p05_c0_c0` | `True` | `True` | 2.073 | 0.1473 | 0.54 | 0.0002273 | 0 | 0.2297/0.2313 | 2.955/5 | 15.84/19.3 | `False` |
| `policy_pd_torque_c0p1_c0_c0` | `True` | `True` | 2.065 | 0.1568 | 0.54 | 0.0004545 | 0 | 0.2295/0.231 | 3.04/5 | 15.8/19.66 | `False` |
| `policy_pd_torque_c0_c0_c0p1` | `True` | `True` | 2.083 | 0.138 | 0.54 | 0.0002273 | 0 | 0.2299/0.2314 | 2.735/5 | 15.82/19.49 | `False` |

## Summary

- cases run: `16`
- cases with runner success: `16/16`
- cases with checker success: `16/16`
- most stable case by root height / velocity / torque / contact score: `policy_pd_torque_c0_c0_c0`
- least stable case by root height / velocity / torque / contact score: `zero_action_position_c0_c0_c0`
- highest action saturation case: `policy_position_c0p1_c0_c0`
- highest torque saturation case: `policy_position_c0p1_c0_c0`
- early fall cases: `['zero_action_position_c0_c0_c0', 'zero_action_position_c0p05_c0_c0', 'zero_action_position_c0p1_c0_c0', 'zero_action_position_c0_c0_c0p1', 'zero_action_pd_torque_c0_c0_c0', 'zero_action_pd_torque_c0p05_c0_c0', 'zero_action_pd_torque_c0p1_c0_c0', 'zero_action_pd_torque_c0_c0_c0p1']`
- velocity-limit exceeded cases: `[]`

## Position Vs PD Torque

- `zero_action` command `[0.0, 0.0, 0.0]`: root_min position/pd_torque `0.02755` / `0.03015`, torque_sat `0.0004545` / `0.0004545`, foot_force_max `13.65` / `13.56`
- `zero_action` command `[0.05, 0.0, 0.0]`: root_min position/pd_torque `0.02755` / `0.03015`, torque_sat `0.0004545` / `0.0004545`, foot_force_max `13.65` / `13.56`
- `zero_action` command `[0.1, 0.0, 0.0]`: root_min position/pd_torque `0.02755` / `0.03015`, torque_sat `0.0004545` / `0.0004545`, foot_force_max `13.65` / `13.56`
- `zero_action` command `[0.0, 0.0, 0.1]`: root_min position/pd_torque `0.02755` / `0.03015`, torque_sat `0.0004545` / `0.0004545`, foot_force_max `13.65` / `13.56`
- `policy` command `[0.0, 0.0, 0.0]`: root_min position/pd_torque `0.2294` / `0.2299`, torque_sat `0.001136` / `0.0002273`, foot_force_max `19.43` / `19.78`
- `policy` command `[0.05, 0.0, 0.0]`: root_min position/pd_torque `0.2289` / `0.2297`, torque_sat `0.0004545` / `0.0002273`, foot_force_max `19.92` / `19.3`
- `policy` command `[0.1, 0.0, 0.0]`: root_min position/pd_torque `0.2289` / `0.2295`, torque_sat `0.0125` / `0.0004545`, foot_force_max `21.1` / `19.66`
- `policy` command `[0.0, 0.0, 0.1]`: root_min position/pd_torque `0.2294` / `0.2299`, torque_sat `0.001136` / `0.0002273`, foot_force_max `21.3` / `19.49`

## Zero-Action Vs Policy

- `position` command `[0.0, 0.0, 0.0]`: action_sat zero/policy `0` / `0.1368`, root_min `0.02755` / `0.2294`, torque_sat `0.0004545` / `0.001136`
- `position` command `[0.05, 0.0, 0.0]`: action_sat zero/policy `0` / `0.1445`, root_min `0.02755` / `0.2289`, torque_sat `0.0004545` / `0.0004545`
- `position` command `[0.1, 0.0, 0.0]`: action_sat zero/policy `0` / `0.1705`, root_min `0.02755` / `0.2289`, torque_sat `0.0004545` / `0.0125`
- `position` command `[0.0, 0.0, 0.1]`: action_sat zero/policy `0` / `0.1373`, root_min `0.02755` / `0.2294`, torque_sat `0.0004545` / `0.001136`
- `pd_torque` command `[0.0, 0.0, 0.0]`: action_sat zero/policy `0` / `0.1345`, root_min `0.03015` / `0.2299`, torque_sat `0.0004545` / `0.0002273`
- `pd_torque` command `[0.05, 0.0, 0.0]`: action_sat zero/policy `0` / `0.1473`, root_min `0.03015` / `0.2297`, torque_sat `0.0004545` / `0.0002273`
- `pd_torque` command `[0.1, 0.0, 0.0]`: action_sat zero/policy `0` / `0.1568`, root_min `0.03015` / `0.2295`, torque_sat `0.0004545` / `0.0004545`
- `pd_torque` command `[0.0, 0.0, 0.1]`: action_sat zero/policy `0` / `0.138`, root_min `0.03015` / `0.2299`, torque_sat `0.0004545` / `0.0002273`

## Likely Failure Causes Ranked

1. Root height drops below the fall heuristic in multiple zero-action cases, so passive/default-pose settling and contact dynamics need inspection before interpreting policy quality.
2. Policy action saturation is present; check policy observation conventions, action scale, and exported action processing.
3. Raw PD torque exceeds effort limits in some samples; inspect actuator limits and PD implementation before changing gains.
4. No joint exceeded `velocity_limit_sim` in this matrix.
5. Contact-force variation remains a physics fidelity signal; inspect contact/friction/solver only with controlled comparisons, not policy tuning.

## Next Steps

- If action or torque saturation grows, inspect policy/action scale/exported actuator limits before touching gains.
- If root-frame signals look inconsistent, run controlled frame diagnostics for `base_ang_vel` and `projected_gravity`.
- If contact force is abnormal, compare contact geometry, friction, and solver behavior without changing formal foot mesh.
- If `pd_torque` and `position` diverge strongly, isolate actuator implementation and timing differences.
- Regenerate this matrix after each narrow sim2sim fix and compare only one variable at a time.
