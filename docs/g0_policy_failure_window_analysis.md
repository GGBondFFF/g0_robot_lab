# G0 Policy Failure Window Analysis

## Scope

- policy rollout cases analyzed: `8`
- root danger threshold: `0.16` m
- root warning threshold: `0.2` m
- action saturation step threshold: `0.3`
- torque saturation step threshold: `0.05`
- pre-failure window: `20` steps

## Case Summary

| case | mode | command | stable | failure step | <0.20 step | likely precursor | root min/final | action sat mean/max | torque sat mean/max | foot force mean/max |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| `policy_pd_torque_c0_c0_c0` | `pd_torque` | `[0.0, 0.0, 0.0]` | `False` | 421 | 417 | `action saturation first` | 0.03247/0.03628 | 0.2718/1 | 0.008182/0.1364 | 13.23/20.89 |
| `policy_pd_torque_c0_c0_c0p1` | `pd_torque` | `[0.0, 0.0, 0.1]` | `False` | 414 | 410 | `action saturation first` | 0.03102/0.03469 | 0.2759/0.9545 | 0.007091/0.1818 | 12.98/20.12 |
| `policy_pd_torque_c0p05_c0_c0` | `pd_torque` | `[0.05, 0.0, 0.0]` | `False` | 449 | 445 | `action saturation first` | 0.03638/0.0364 | 0.2113/0.8182 | 0.006636/0.09091 | 13.96/20.68 |
| `policy_pd_torque_c0p1_c0_c0` | `pd_torque` | `[0.1, 0.0, 0.0]` | `True` | n/a | n/a | `stable` | 0.2292/0.2292 | 0.1546/0.2727 | 9.091e-05/0.04545 | 15.76/19.8 |
| `policy_position_c0_c0_c0` | `position` | `[0.0, 0.0, 0.0]` | `False` | 324 | 320 | `action saturation first` | 0.02918/0.03409 | 0.4053/1 | 0.01291/0.2273 | 10.19/21.41 |
| `policy_position_c0_c0_c0p1` | `position` | `[0.0, 0.0, 0.1]` | `True` | n/a | n/a | `stable` | 0.2292/0.2298 | 0.1386/0.3182 | 0.001455/0.04545 | 15.69/19.75 |
| `policy_position_c0p05_c0_c0` | `position` | `[0.05, 0.0, 0.0]` | `False` | 355 | 351 | `action saturation first` | 0.0327/0.03666 | 0.373/1 | 0.01655/0.2727 | 11.13/19.46 |
| `policy_position_c0p1_c0_c0` | `position` | `[0.1, 0.0, 0.0]` | `True` | n/a | n/a | `stable` | 0.2165/0.2165 | 0.1735/0.5 | 0.01091/0.09091 | 15.16/22.09 |

## Direct Answers

1. Failed policy cases: `['policy_pd_torque_c0_c0_c0', 'policy_pd_torque_c0_c0_c0p1', 'policy_pd_torque_c0p05_c0_c0', 'policy_position_c0_c0_c0', 'policy_position_c0p05_c0_c0']`
2. Stable policy cases: `['policy_pd_torque_c0p1_c0_c0', 'policy_position_c0_c0_c0p1', 'policy_position_c0p1_c0_c0']`
3. Most common pre-failure signal: `action saturation first`
4. Action saturation is a major suspect when it appears before root-height failure and is high in failing low/zero-command cases.
5. Torque saturation is not the primary suspect in this matrix: it is low and usually not the earliest event.
6. Velocity limit is not the primary suspect: no policy case crosses the configured velocity limits in the matrix checker.
7. Foot contact changes are present, but this script treats contact as a dynamics/contact precursor only when contact loss appears before root-height failure.
8. `pd_torque` delays or avoids instability for the forward command `[0.1, 0.0, 0.0]`, but it does not dominate every command.
9. Current likely cause ranking: policy action saturation / action scale, actuator implementation difference, contact/root settling difference, dynamics/inertia difference, policy robustness limitation.

## Suspicious Joint Ranking

- Aggregate suspicious joints: `[('r_knee_pitch_joint', 34.0), ('l_hip_pitch_joint', 32.0), ('r_hip_pitch_joint', 16.0), ('l_ankle_pitch_joint', 14.0), ('l_knee_pitch_joint', 9.0), ('r_ankle_roll_joint', 9.0), ('l_hip_roll_joint', 5.0), ('r_ankle_pitch_joint', 1.0)]`

## Per-Case Details

### policy_pd_torque_c0_c0_c0

- stable / unstable: `unstable`
- failure step: `421`
- first root_height < 0.20 step: `417`
- likely precursor: `action saturation first`
- event steps: action_sat `227`, root_tilt `412`, contact_loss `426`, torque_sat `299`, velocity_spike `n/a`
- most suspicious joints: `['r_knee_pitch_joint', 'r_hip_pitch_joint', 'l_hip_pitch_joint', 'l_ankle_pitch_joint', 'l_knee_pitch_joint']`
- max_abs_action per joint top: `waist_yaw_joint=16.63, l_shoulder_pitch_joint=13.97, r_shoulder_pitch_joint=13.94, r_hip_yaw_joint=13.8, l_ankle_pitch_joint=13.53`
- action saturation ratio per joint top: `r_knee_pitch_joint=0.64, l_ankle_roll_joint=0.458, l_ankle_pitch_joint=0.448, l_knee_pitch_joint=0.446, r_ankle_roll_joint=0.442`
- pd_tau_cmd max per joint top: `r_hip_pitch_joint=1.283, r_knee_pitch_joint=1.136, l_hip_pitch_joint=1.127, l_ankle_pitch_joint=0.736, l_knee_pitch_joint=0.6789`
- joint_vel max per joint top: `l_shoulder_pitch_joint=10.02, waist_roll_joint=4.852, l_hip_pitch_joint=3.773, r_shoulder_pitch_joint=3.649, l_hip_yaw_joint=2.975`
- root roll/pitch max abs deg: `168.3` / `87.63`
- base_ang_vel max abs: `5.48`
- projected_gravity xy max abs: `0.9991`
- contact_count min/mean/max: `1` / `2.514` / `8`
- foot_ground_contact_count min/mean/max: `0` / `2.314` / `5`
- 20-step window before failure/end: `401` to `421`
- window root_height min/mean/max: `0.1682` / `0.2157` / `0.2303`
- window action saturation min/mean/max: `0.09091` / `0.4023` / `0.5909`
- window torque saturation min/mean/max: `0` / `0.02045` / `0.04545`
- window joint_vel_abs_max min/mean/max: `0.5245` / `0.961` / `1.49`
- window foot_contact_force_norm min/mean/max: `6.637` / `10.92` / `15.21`
- window foot_ground_contact_count min/mean/max: `1` / `1.85` / `2`

### policy_pd_torque_c0_c0_c0p1

- stable / unstable: `unstable`
- failure step: `414`
- first root_height < 0.20 step: `410`
- likely precursor: `action saturation first`
- event steps: action_sat `187`, root_tilt `405`, contact_loss `419`, torque_sat `342`, velocity_spike `n/a`
- most suspicious joints: `['r_knee_pitch_joint', 'l_hip_pitch_joint', 'r_ankle_roll_joint', 'r_hip_pitch_joint', 'l_knee_pitch_joint']`
- max_abs_action per joint top: `l_ankle_pitch_joint=20.5, waist_yaw_joint=15.44, r_shoulder_pitch_joint=14.51, l_shoulder_pitch_joint=13.23, r_hip_yaw_joint=12.54`
- action saturation ratio per joint top: `r_knee_pitch_joint=0.626, l_ankle_roll_joint=0.45, r_ankle_roll_joint=0.44, l_knee_pitch_joint=0.434, l_ankle_pitch_joint=0.426`
- pd_tau_cmd max per joint top: `r_knee_pitch_joint=0.9196, r_hip_pitch_joint=0.8204, l_shoulder_pitch_joint=0.7537, l_hip_pitch_joint=0.7383, l_knee_pitch_joint=0.7075`
- joint_vel max per joint top: `l_shoulder_pitch_joint=9.955, waist_roll_joint=4.06, l_elbow_pitch_joint=3.716, l_hip_yaw_joint=3.582, r_shoulder_pitch_joint=3.57`
- root roll/pitch max abs deg: `151.6` / `85.54`
- base_ang_vel max abs: `7.876`
- projected_gravity xy max abs: `0.997`
- contact_count min/mean/max: `1` / `2.522` / `6`
- foot_ground_contact_count min/mean/max: `0` / `2.3` / `6`
- 20-step window before failure/end: `394` to `414`
- window root_height min/mean/max: `0.1626` / `0.2136` / `0.2304`
- window action saturation min/mean/max: `0.1364` / `0.3432` / `0.5`
- window torque saturation min/mean/max: `0` / `0.03636` / `0.09091`
- window joint_vel_abs_max min/mean/max: `0.6598` / `1.047` / `1.373`
- window foot_contact_force_norm min/mean/max: `6.033` / `11.44` / `16.27`
- window foot_ground_contact_count min/mean/max: `1` / `1.85` / `2`

### policy_pd_torque_c0p05_c0_c0

- stable / unstable: `unstable`
- failure step: `449`
- first root_height < 0.20 step: `445`
- likely precursor: `action saturation first`
- event steps: action_sat `351`, root_tilt `438`, contact_loss `456`, torque_sat `422`, velocity_spike `n/a`
- most suspicious joints: `['l_ankle_pitch_joint', 'r_knee_pitch_joint', 'l_hip_pitch_joint', 'r_hip_pitch_joint', 'l_hip_roll_joint']`
- max_abs_action per joint top: `r_ankle_roll_joint=9.039, l_hip_yaw_joint=8.971, r_hip_pitch_joint=7.991, r_ankle_pitch_joint=7.55, r_shoulder_roll_joint=6.735`
- action saturation ratio per joint top: `r_knee_pitch_joint=0.506, l_knee_pitch_joint=0.47, l_hip_pitch_joint=0.418, l_ankle_roll_joint=0.402, r_ankle_roll_joint=0.398`
- pd_tau_cmd max per joint top: `l_ankle_pitch_joint=1.439, r_knee_pitch_joint=0.7798, l_hip_pitch_joint=0.7319, l_shoulder_pitch_joint=0.7183, l_hip_roll_joint=0.7179`
- joint_vel max per joint top: `l_shoulder_pitch_joint=7.369, l_shoulder_yaw_joint=3.437, r_shoulder_pitch_joint=3.336, l_elbow_pitch_joint=3.099, l_shoulder_roll_joint=2.576`
- root roll/pitch max abs deg: `103.7` / `89.4`
- base_ang_vel max abs: `5.604`
- projected_gravity xy max abs: `0.9999`
- contact_count min/mean/max: `1` / `2.888` / `7`
- foot_ground_contact_count min/mean/max: `0` / `2.538` / `5`
- 20-step window before failure/end: `429` to `449`
- window root_height min/mean/max: `0.1724` / `0.2167` / `0.2318`
- window action saturation min/mean/max: `0.1818` / `0.2455` / `0.3636`
- window torque saturation min/mean/max: `0` / `0.02273` / `0.04545`
- window joint_vel_abs_max min/mean/max: `0.4951` / `1.048` / `2.069`
- window foot_contact_force_norm min/mean/max: `7.695` / `11.59` / `13.83`
- window foot_ground_contact_count min/mean/max: `1` / `1.9` / `2`

### policy_pd_torque_c0p1_c0_c0

- stable / unstable: `stable`
- failure step: `n/a`
- first root_height < 0.20 step: `n/a`
- likely precursor: `stable`
- event steps: action_sat `n/a`, root_tilt `n/a`, contact_loss `n/a`, torque_sat `n/a`, velocity_spike `n/a`
- most suspicious joints: `['l_hip_pitch_joint', 'r_knee_pitch_joint', 'l_hip_roll_joint', 'l_knee_pitch_joint', 'r_ankle_roll_joint']`
- max_abs_action per joint top: `r_ankle_roll_joint=2.127, l_ankle_roll_joint=1.944, l_knee_pitch_joint=1.825, l_ankle_pitch_joint=1.559, r_knee_pitch_joint=1.504`
- action saturation ratio per joint top: `l_hip_pitch_joint=0.546, r_knee_pitch_joint=0.524, l_knee_pitch_joint=0.496, l_hip_roll_joint=0.338, r_hip_roll_joint=0.312`
- pd_tau_cmd max per joint top: `r_ankle_roll_joint=0.54, l_ankle_pitch_joint=0.5391, l_hip_pitch_joint=0.4996, r_hip_pitch_joint=0.4886, l_hip_roll_joint=0.4823`
- joint_vel max per joint top: `r_ankle_roll_joint=1.9, l_hip_pitch_joint=1.554, r_hip_pitch_joint=1.484, r_knee_pitch_joint=1.443, l_ankle_roll_joint=1.348`
- root roll/pitch max abs deg: `1.555` / `4.602`
- base_ang_vel max abs: `0.5801`
- projected_gravity xy max abs: `0.08023`
- contact_count min/mean/max: `2` / `3.088` / `5`
- foot_ground_contact_count min/mean/max: `2` / `3.088` / `5`
- 20-step window before failure/end: `480` to `500`
- window root_height min/mean/max: `0.2292` / `0.2306` / `0.2315`
- window action saturation min/mean/max: `0.04545` / `0.1341` / `0.2273`
- window torque saturation min/mean/max: `0` / `0` / `0`
- window joint_vel_abs_max min/mean/max: `0.5711` / `1.028` / `1.448`
- window foot_contact_force_norm min/mean/max: `12.2` / `15.49` / `17.12`
- window foot_ground_contact_count min/mean/max: `2` / `2.9` / `4`

### policy_position_c0_c0_c0

- stable / unstable: `unstable`
- failure step: `324`
- first root_height < 0.20 step: `320`
- likely precursor: `action saturation first`
- event steps: action_sat `148`, root_tilt `314`, contact_loss `328`, torque_sat `295`, velocity_spike `n/a`
- most suspicious joints: `['l_hip_pitch_joint', 'r_knee_pitch_joint', 'r_hip_pitch_joint', 'r_ankle_roll_joint', 'r_ankle_pitch_joint']`
- max_abs_action per joint top: `waist_yaw_joint=17.64, l_ankle_pitch_joint=16.93, l_shoulder_pitch_joint=14.62, r_hip_yaw_joint=14.24, r_shoulder_pitch_joint=14.11`
- action saturation ratio per joint top: `r_knee_pitch_joint=0.702, l_hip_pitch_joint=0.574, r_hip_roll_joint=0.514, l_ankle_pitch_joint=0.508, r_ankle_roll_joint=0.504`
- pd_tau_cmd max per joint top: `l_hip_pitch_joint=1.201, r_knee_pitch_joint=0.9788, r_hip_pitch_joint=0.9364, l_knee_pitch_joint=0.9188, r_ankle_pitch_joint=0.9063`
- joint_vel max per joint top: `l_hip_pitch_joint=5.17, r_hip_pitch_joint=4.987, l_elbow_pitch_joint=4.568, l_hip_yaw_joint=4.317, waist_yaw_joint=4.193`
- root roll/pitch max abs deg: `175.8` / `88.4`
- base_ang_vel max abs: `7.568`
- projected_gravity xy max abs: `0.9996`
- contact_count min/mean/max: `0` / `2.306` / `7`
- foot_ground_contact_count min/mean/max: `0` / `1.862` / `5`
- 20-step window before failure/end: `304` to `324`
- window root_height min/mean/max: `0.164` / `0.2147` / `0.2312`
- window action saturation min/mean/max: `0.3182` / `0.4364` / `0.5909`
- window torque saturation min/mean/max: `0` / `0.05682` / `0.09091`
- window joint_vel_abs_max min/mean/max: `1.018` / `1.631` / `2.321`
- window foot_contact_force_norm min/mean/max: `5.537` / `11.77` / `15.92`
- window foot_ground_contact_count min/mean/max: `1` / `1.75` / `2`

### policy_position_c0_c0_c0p1

- stable / unstable: `stable`
- failure step: `n/a`
- first root_height < 0.20 step: `n/a`
- likely precursor: `stable`
- event steps: action_sat `67`, root_tilt `n/a`, contact_loss `n/a`, torque_sat `n/a`, velocity_spike `n/a`
- most suspicious joints: `['r_knee_pitch_joint', 'l_knee_pitch_joint', 'l_hip_pitch_joint', 'r_ankle_roll_joint', 'l_hip_roll_joint']`
- max_abs_action per joint top: `l_ankle_roll_joint=2.147, r_ankle_roll_joint=2.057, l_knee_pitch_joint=1.802, l_hip_roll_joint=1.76, r_knee_pitch_joint=1.709`
- action saturation ratio per joint top: `r_knee_pitch_joint=0.56, l_knee_pitch_joint=0.452, l_hip_pitch_joint=0.372, l_ankle_roll_joint=0.326, r_hip_roll_joint=0.3`
- pd_tau_cmd max per joint top: `l_ankle_pitch_joint=0.6735, r_ankle_roll_joint=0.54, l_knee_pitch_joint=0.5323, l_hip_roll_joint=0.5102, l_hip_pitch_joint=0.4892`
- joint_vel max per joint top: `r_ankle_roll_joint=2.348, r_knee_pitch_joint=1.924, waist_roll_joint=1.817, l_hip_pitch_joint=1.766, l_ankle_roll_joint=1.679`
- root roll/pitch max abs deg: `3.196` / `5.764`
- base_ang_vel max abs: `1.205`
- projected_gravity xy max abs: `0.1004`
- contact_count min/mean/max: `1` / `2.786` / `5`
- foot_ground_contact_count min/mean/max: `1` / `2.786` / `5`
- 20-step window before failure/end: `480` to `500`
- window root_height min/mean/max: `0.2297` / `0.2312` / `0.2327`
- window action saturation min/mean/max: `0.04545` / `0.1023` / `0.1818`
- window torque saturation min/mean/max: `0` / `0` / `0`
- window joint_vel_abs_max min/mean/max: `0.6654` / `1.178` / `1.596`
- window foot_contact_force_norm min/mean/max: `12.45` / `15.81` / `18.46`
- window foot_ground_contact_count min/mean/max: `2` / `2.75` / `5`

### policy_position_c0p05_c0_c0

- stable / unstable: `unstable`
- failure step: `355`
- first root_height < 0.20 step: `351`
- likely precursor: `action saturation first`
- event steps: action_sat `239`, root_tilt `347`, contact_loss `359`, torque_sat `240`, velocity_spike `n/a`
- most suspicious joints: `['l_hip_pitch_joint', 'r_knee_pitch_joint', 'r_hip_pitch_joint', 'l_ankle_pitch_joint', 'r_ankle_roll_joint']`
- max_abs_action per joint top: `l_ankle_pitch_joint=23.39, l_hip_yaw_joint=18.25, waist_yaw_joint=14.74, r_shoulder_pitch_joint=14.05, r_hip_yaw_joint=13.64`
- action saturation ratio per joint top: `r_knee_pitch_joint=0.654, l_hip_pitch_joint=0.596, l_ankle_pitch_joint=0.496, r_hip_pitch_joint=0.49, r_ankle_roll_joint=0.486`
- pd_tau_cmd max per joint top: `l_hip_pitch_joint=1.421, r_knee_pitch_joint=1.19, l_ankle_pitch_joint=1.107, r_hip_pitch_joint=0.9932, l_knee_pitch_joint=0.8972`
- joint_vel max per joint top: `l_shoulder_roll_joint=9.746, l_shoulder_pitch_joint=9.372, waist_roll_joint=6.368, r_shoulder_pitch_joint=5.425, l_hip_yaw_joint=4.001`
- root roll/pitch max abs deg: `174.9` / `88.37`
- base_ang_vel max abs: `7.368`
- projected_gravity xy max abs: `0.9996`
- contact_count min/mean/max: `0` / `2.302` / `5`
- foot_ground_contact_count min/mean/max: `0` / `1.938` / `5`
- 20-step window before failure/end: `335` to `355`
- window root_height min/mean/max: `0.1712` / `0.219` / `0.2377`
- window action saturation min/mean/max: `0.3636` / `0.5159` / `0.8636`
- window torque saturation min/mean/max: `0` / `0.05` / `0.1364`
- window joint_vel_abs_max min/mean/max: `1.449` / `2.054` / `2.718`
- window foot_contact_force_norm min/mean/max: `6.662` / `13.99` / `17.19`
- window foot_ground_contact_count min/mean/max: `1` / `1.4` / `2`

### policy_position_c0p1_c0_c0

- stable / unstable: `stable`
- failure step: `n/a`
- first root_height < 0.20 step: `n/a`
- likely precursor: `stable`
- event steps: action_sat `56`, root_tilt `496`, contact_loss `n/a`, torque_sat `44`, velocity_spike `n/a`
- most suspicious joints: `['l_ankle_pitch_joint', 'l_hip_pitch_joint', 'r_knee_pitch_joint', 'r_hip_pitch_joint', 'l_knee_pitch_joint']`
- max_abs_action per joint top: `l_hip_pitch_joint=3.512, r_hip_pitch_joint=3.261, l_ankle_pitch_joint=2.84, r_ankle_roll_joint=2.642, l_knee_pitch_joint=2.216`
- action saturation ratio per joint top: `l_hip_pitch_joint=0.576, l_knee_pitch_joint=0.52, r_knee_pitch_joint=0.484, r_hip_pitch_joint=0.382, l_ankle_roll_joint=0.314`
- pd_tau_cmd max per joint top: `l_ankle_pitch_joint=1.874, r_ankle_pitch_joint=0.7084, r_knee_pitch_joint=0.6409, r_hip_pitch_joint=0.6286, l_hip_pitch_joint=0.5887`
- joint_vel max per joint top: `r_knee_pitch_joint=2.754, r_shoulder_pitch_joint=2.688, l_ankle_roll_joint=2.335, r_ankle_roll_joint=2.315, l_hip_pitch_joint=2.142`
- root roll/pitch max abs deg: `12.97` / `26.55`
- base_ang_vel max abs: `1.717`
- projected_gravity xy max abs: `0.447`
- contact_count min/mean/max: `1` / `2.75` / `6`
- foot_ground_contact_count min/mean/max: `1` / `2.75` / `6`
- 20-step window before failure/end: `480` to `500`
- window root_height min/mean/max: `0.2165` / `0.2267` / `0.2326`
- window action saturation min/mean/max: `0.1364` / `0.3909` / `0.5`
- window torque saturation min/mean/max: `0` / `0.05` / `0.09091`
- window joint_vel_abs_max min/mean/max: `0.7358` / `1.493` / `2.754`
- window foot_contact_force_norm min/mean/max: `11.35` / `14.16` / `16.98`
- window foot_ground_contact_count min/mean/max: `1` / `1.7` / `2`

## Next Validation Steps

- Compare saturated joints against action scale and default pose offsets before changing any actuator or solver parameter.
- Run a narrow actuator timing comparison for matching command pairs where position and pd_torque diverge.
- Inspect contact/root settling windows for failed cases, especially the first 100 steps.
- Keep model, collision geometry, gains, friction, solver, root height, and policy unchanged until the cause category is isolated.
