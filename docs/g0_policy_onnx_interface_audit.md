# G0 Policy / ONNX Interface Audit

Read-only audit of `G0-Velocity-v0` training env vs exported `policy.onnx` deployment interface.

**Audit date:** 2026-05-26  
**Env config:** `source/g0_robot_lab/g0_robot_lab/tasks/locomotion/robots/g0/velocity_env_cfg.py`  
**Robot / gains source:** `source/g0_robot_lab/g0_robot_lab/assets/robots/g0/g0.py`  
**Reference ONNX (verified on disk):** `logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/exported/policy.onnx`  
All recent exports under `logs/rsl_rl/g0_velocity/*/exported/policy.onnx` share the same I/O signature.

---

## 1. `policy.onnx` tensor interface

Verified via `onnxruntime.InferenceSession` (`g0_isaaclab` env):

| Field | Value |
|-------|-------|
| **input_name** | `obs` |
| **input_shape** | `[1, 385]` (batch, flat obs vector) |
| **input_dim** | **385** |
| **output_name** | `actions` |
| **output_shape** | `[1, 22]` |
| **output_dim** | **22** |
| **dtype** | `float32` (both I/O) |
| **empirical_normalization** | `False` (`rsl_rl_ppo_cfg.py`) — no obs normalizer in export |

Export path: `scripts/rsl_rl/play.py` → `export_policy_to_onnx(..., filename="policy.onnx")`.

---

## 2. Isaac Lab env — policy observation group

Task id: `G0-Velocity-v0`  
Group: `ObservationsCfg.PolicyCfg` (`velocity_env_cfg.py`)

### 2.1 Group settings

```text
history_length      = 5
concatenate_terms   = True
enable_corruption   = True   (training only; disable for deployment parity tests)
```

Policy control rate:

```text
sim.dt       = 0.005 s
decimation   = 4
env.step_dt  = 0.02 s  → 50 Hz
```

### 2.2 Policy obs terms (declaration order = concat order)

| # | Term | Per-step dim | Scale | Noise (train) | Notes |
|---|------|-------------|-------|---------------|-------|
| 1 | `base_ang_vel` | 3 | **0.2** | Uniform ±0.2 | Body-frame angular velocity (rad/s), scaled |
| 2 | `projected_gravity` | 3 | 1.0 | Uniform ±0.05 | `R_world→body @ [0,0,-1]` |
| 3 | `velocity_commands` | 3 | 1.0 | — | From `commands.base_velocity`: `[lin_vel_x, lin_vel_y, ang_vel_z]` |
| 4 | `joint_pos_rel` | 22 | 1.0 | Uniform ±0.01 | `q - default_joint_pos`, **articulation order** (see §5.2) |
| 5 | `joint_vel_rel` | 22 | **0.05** | Uniform ±1.5 | Joint velocities (rad/s), **articulation order** |
| 6 | `last_action` | 22 | 1.0 | — | Previous raw policy action, **SDK order** (see §5.1) |
| 7 | `gait_phase` | 2 | 1.0 | — | `[sin(φ), cos(φ)]`, `φ = 2π · (episode_step · step_dt mod period) / period`, **period = 0.8 s** |

**Per-step total:** 3 + 3 + 3 + 22 + 22 + 22 + 2 = **77**

### 2.3 History + concatenation layout → 385-dim `obs`

Isaac Lab with `concatenate_terms=True` and `history_length=5`:

1. For **each term**, maintain a buffer of shape `(5, term_dim)`.
2. Flatten each buffer **row-major** (history slowest, term components fastest): `(5 × term_dim,)`.
3. **Concatenate** flattened term vectors in **declaration order** above.

| Term | History × dim | Flattened size |
|------|---------------|----------------|
| `base_ang_vel` | 5 × 3 | 15 |
| `projected_gravity` | 5 × 3 | 15 |
| `velocity_commands` | 5 × 3 | 15 |
| `joint_pos_rel` | 5 × 22 | 110 |
| `joint_vel_rel` | 5 × 22 | 110 |
| `last_action` | 5 × 22 | 110 |
| `gait_phase` | 5 × 2 | 10 |
| **Total** | | **385** |

**Within-term history order:** oldest → newest (index 0 = oldest, index 4 = newest). Matches Isaac Lab default and `policy_inference.toml` `[obs.history].order = "back"`.

**Flat index formula** (term `t` with per-step dim `d_t`, history index `h ∈ {0..4}`):

```text
offset(term) = sum of (5 × d) for all preceding terms
index = offset(term) + h * d_t + component
```

Reference implementation: `test/policy_test/python/policy_inference.py` (`ObsBuilder`).

---

## 3. Action interface

| Field | Value |
|-------|-------|
| **Action term** | `JointPositionActionCfg` (`actions.joint_pos`) |
| **action_dim** | **22** |
| **action_scale** | **0.12** |
| **use_default_offset** | **True** |
| **preserve_order** | **True** |
| **joint_names** | `G0_JOINT_SDK_NAMES` (§5.1) |

**Target joint position (rad):**

```text
target_joint_pos = default_joint_pos + action_scale × policy_action
                 = default_joint_pos + 0.12 × actions
```

`policy_action` / ONNX `actions` are **raw network outputs** in **SDK joint order**. No extra post-scale inside ONNX.

Training wrapper: `RslRlVecEnvWrapper(..., clip_actions=agent_cfg.clip_actions)` — check runner cfg at export time if action clipping matters for deployment.

---

## 4. `default_joint_pos` (rad)

Source: `G0_DEFAULT_JOINT_POS` in `g0.py` → `G0_CFG.init_state.joint_pos`.

Values below are listed in **SDK / action order** (same as ONNX `actions` index).

| Idx | Joint | default (rad) |
|-----|-------|---------------|
| 0 | `l_hip_pitch_joint` | -0.20 |
| 1 | `l_hip_roll_joint` | 0.00 |
| 2 | `l_hip_yaw_joint` | 0.00 |
| 3 | `l_knee_pitch_joint` | 0.34 |
| 4 | `l_ankle_pitch_joint` | -0.14 |
| 5 | `l_ankle_roll_joint` | 0.00 |
| 6 | `r_hip_pitch_joint` | -0.20 |
| 7 | `r_hip_roll_joint` | 0.00 |
| 8 | `r_hip_yaw_joint` | 0.00 |
| 9 | `r_knee_pitch_joint` | 0.34 |
| 10 | `r_ankle_pitch_joint` | -0.14 |
| 11 | `r_ankle_roll_joint` | 0.00 |
| 12 | `waist_yaw_joint` | 0.00 |
| 13 | `waist_roll_joint` | 0.00 |
| 14 | `l_shoulder_pitch_joint` | 0.30 |
| 15 | `l_shoulder_roll_joint` | 0.25 |
| 16 | `l_shoulder_yaw_joint` | 0.00 |
| 17 | `l_elbow_pitch_joint` | -0.97 |
| 18 | `r_shoulder_pitch_joint` | 0.30 |
| 19 | `r_shoulder_roll_joint` | -0.25 |
| 20 | `r_shoulder_yaw_joint` | 0.00 |
| 21 | `r_elbow_pitch_joint` | -0.97 |

Initial root height: `G0_CFG.init_state.pos = (0, 0, 0.23)`.

---

## 5. Joint order (three orderings)

> **Critical:** SDK order (actions) ≠ articulation order (`joint_pos_rel` / `joint_vel_rel` obs) ≠ wire motor-id order (hardware DDS).

### 5.1 SDK order — `G0_JOINT_SDK_NAMES`

Used by:

- `JointPositionActionCfg.joint_names`
- ONNX output `actions[i]`
- obs term `last_action[i]`

```text
 0  l_hip_pitch_joint
 1  l_hip_roll_joint
 2  l_hip_yaw_joint
 3  l_knee_pitch_joint
 4  l_ankle_pitch_joint
 5  l_ankle_roll_joint
 6  r_hip_pitch_joint
 7  r_hip_roll_joint
 8  r_hip_yaw_joint
 9  r_knee_pitch_joint
10  r_ankle_pitch_joint
11  r_ankle_roll_joint
12  waist_yaw_joint
13  waist_roll_joint
14  l_shoulder_pitch_joint
15  l_shoulder_roll_joint
16  l_shoulder_yaw_joint
17  l_elbow_pitch_joint
18  r_shoulder_pitch_joint
19  r_shoulder_roll_joint
20  r_shoulder_yaw_joint
21  r_elbow_pitch_joint
```

### 5.2 Articulation order — Isaac Lab runtime (USD / PhysX tree)

Used by:

- `mdp.joint_pos_rel` (all 22 DOF, no `joint_names` filter)
- `mdp.joint_vel_rel`

Captured from `isaac_gui_plant.py` articulation print (2026-05-22); stored in `test/policy_test/config/policy_inference.toml` as `urdf_order_explicit`.

**Not the same as `G0_JOINT_NAMES`** (URDF file traversal in `g0.py` comments).

```text
 0  l_hip_pitch_joint
 1  r_hip_pitch_joint
 2  waist_yaw_joint
 3  l_hip_roll_joint
 4  r_hip_roll_joint
 5  waist_roll_joint
 6  l_hip_yaw_joint
 7  r_hip_yaw_joint
 8  l_shoulder_pitch_joint
 9  r_shoulder_pitch_joint
10  l_knee_pitch_joint
11  r_knee_pitch_joint
12  l_shoulder_roll_joint
13  r_shoulder_roll_joint
14  l_ankle_pitch_joint
15  r_ankle_pitch_joint
16  l_shoulder_yaw_joint
17  r_shoulder_yaw_joint
18  l_ankle_roll_joint
19  r_ankle_roll_joint
20  l_elbow_pitch_joint
21  r_elbow_pitch_joint
```

Re-verify after URDF/USD changes: `scripts/debug/debug_runtime_joint_order.py`.

### 5.3 Wire order — motor_id 1..22 (deployment / DDS)

From `test/policy_test/python/joint_mapping.py` / `cpp/include/joint_mapping.hpp`:

```text
 1  waist_yaw_joint
 2  waist_roll_joint
 3  l_shoulder_pitch_joint
 4  l_shoulder_roll_joint
 5  l_shoulder_yaw_joint
 6  l_elbow_pitch_joint
 7  r_shoulder_pitch_joint
 8  r_shoulder_roll_joint
 9  r_shoulder_yaw_joint
10  r_elbow_pitch_joint
11  l_hip_pitch_joint
12  l_hip_roll_joint
13  l_hip_yaw_joint
14  l_knee_pitch_joint
15  l_ankle_pitch_joint
16  l_ankle_roll_joint
17  r_hip_pitch_joint
18  r_hip_roll_joint
19  r_hip_yaw_joint
20  r_knee_pitch_joint
21  r_ankle_pitch_joint
22  r_ankle_roll_joint
```

Permutation helpers: `test/policy_test/python/joint_mapping.py` (`build_ordering`).

---

## 6. Stiffness / damping @ `policy.onnx`

**ONNX does not embed PD gains.** The network maps `obs → actions` only.

Deployment PD gains come from **`G0_CFG.actuators`** in `g0.py` (Isaac implicit actuator tuning). `policy_inference.py` optionally emits them on the DDS wire (`emit_pd_gains_on_wire = true`) using wire units:

```text
kp_wire (N·m/deg) = stiffness (N·m/rad) × (π/180)
kd_wire (N·m/(deg/s)) = damping (N·m/(rad/s)) × (π/180)
```

### 6.1 Isaac Lab source (`g0.py`, rad-domain)

**standard_servos** (16 joints):

| Pattern / joint | stiffness | damping |
|-----------------|-----------|---------|
| `.*_hip_pitch_joint` | 4.0 | 0.18 |
| `.*_hip_roll_joint` | 4.0 | 0.16 |
| `.*_hip_yaw_joint` | 3.0 | 0.10 |
| `.*_ankle_roll_joint` | 4.5 | 0.15 |
| `waist_yaw_joint` | 2.0 | 0.08 |
| `waist_roll_joint` | 2.0 | 0.08 |
| `.*_shoulder_pitch_joint` | 1.5 | 0.06 |
| `.*_shoulder_roll_joint` | 1.5 | 0.06 |
| `.*_shoulder_yaw_joint` | 1.5 | 0.06 |

**right_angle_servos** (6 joints: elbows, knees, ankle_pitch):

| Pattern / joint | stiffness | damping |
|-----------------|-----------|---------|
| `.*_knee_pitch_joint` | 4.0 | 0.26 |
| `.*_ankle_pitch_joint` | 4.0 | 0.22 |
| `.*_elbow_pitch_joint` | 2.0 | 0.08 |

### 6.2 Resolved per joint (wire / motor_id order)

| id | Joint | kp (N·m/rad) | kd (N·m/(rad/s)) | kp_wire | kd_wire |
|----|-------|-------------|------------------|---------|---------|
| 1 | waist_yaw_joint | 2.00 | 0.08 | 0.03491 | 0.00140 |
| 2 | waist_roll_joint | 2.00 | 0.08 | 0.03491 | 0.00140 |
| 3 | l_shoulder_pitch_joint | 1.50 | 0.06 | 0.02618 | 0.00105 |
| 4 | l_shoulder_roll_joint | 1.50 | 0.06 | 0.02618 | 0.00105 |
| 5 | l_shoulder_yaw_joint | 1.50 | 0.06 | 0.02618 | 0.00105 |
| 6 | l_elbow_pitch_joint | 2.00 | 0.08 | 0.03491 | 0.00140 |
| 7 | r_shoulder_pitch_joint | 1.50 | 0.06 | 0.02618 | 0.00105 |
| 8 | r_shoulder_roll_joint | 1.50 | 0.06 | 0.02618 | 0.00105 |
| 9 | r_shoulder_yaw_joint | 1.50 | 0.06 | 0.02618 | 0.00105 |
| 10 | r_elbow_pitch_joint | 2.00 | 0.08 | 0.03491 | 0.00140 |
| 11 | l_hip_pitch_joint | 4.00 | 0.18 | 0.06981 | 0.00314 |
| 12 | l_hip_roll_joint | 4.00 | 0.16 | 0.06981 | 0.00279 |
| 13 | l_hip_yaw_joint | 3.00 | 0.10 | 0.05236 | 0.00175 |
| 14 | l_knee_pitch_joint | 4.00 | 0.26 | 0.06981 | 0.00454 |
| 15 | l_ankle_pitch_joint | 4.00 | 0.22 | 0.06981 | 0.00384 |
| 16 | l_ankle_roll_joint | 4.50 | 0.15 | 0.07854 | 0.00262 |
| 17 | r_hip_pitch_joint | 4.00 | 0.18 | 0.06981 | 0.00314 |
| 18 | r_hip_roll_joint | 4.00 | 0.16 | 0.06981 | 0.00279 |
| 19 | r_hip_yaw_joint | 3.00 | 0.10 | 0.05236 | 0.00175 |
| 20 | r_knee_pitch_joint | 4.00 | 0.26 | 0.06981 | 0.00454 |
| 21 | r_ankle_pitch_joint | 4.00 | 0.22 | 0.06981 | 0.00384 |
| 22 | r_ankle_roll_joint | 4.50 | 0.15 | 0.07854 | 0.00262 |

Codegen mirror: `test/single_joint_test/cpp/include/pd_gains_generated.hpp`.

In Isaac sim (Option A), plant PD is authoritative; wire kp/kd are for real-hardware parity.

---

## 7. Deployment checklist (sim2sim / sim2real)

Must match exactly:

- [ ] `obs` layout: term order, per-term dims, history length 5, oldest→newest
- [ ] Obs scales: `base_ang_vel×0.2`, `joint_vel×0.05`, others ×1
- [ ] `joint_pos_rel` / `joint_vel_rel` in **articulation order** (§5.2)
- [ ] `last_action` / ONNX `actions` in **SDK order** (§5.1)
- [ ] `default_joint_pos` + `action_scale=0.12`
- [ ] Control dt 50 Hz (`step_dt=0.02`)
- [ ] `gait_phase` period 0.8 s, sin/cos convention
- [ ] Command ranges if mapping joystick → velocity command
- [ ] PD gains if firmware uses wire kp/kd (§6)

Related docs:

- `docs/observation_action_interface.md`
- `test/policy_test/config/policy_inference.toml`
- `test/policy_test/python/policy_inference.py`
