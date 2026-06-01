# G0 Sim2Sim Staging SOP (Unitree-Aligned)

Standard operating procedure for bringing the G0 policy up in MuJoCo using the
same **operational flow** as Unitree's G1 sim2sim (elastic band → FixStand →
lower to ground → RLBase → disable band → stable closed loop **without** band).

This document covers the **single-process Python stack** in `deploy/`. It does
**not** use `unitree_mujoco` or `unitree_sdk2` — that two-process / DDS path is a
separate plan (see `.cursor/plans/final-virtual-dds-gate_*.plan.md`, Phase 6).

> Canonical entry point: `python -m deploy.robots.g0.main` + `deploy_staging.yaml`.
> The `scripts/sim2sim/g0_mujoco_onnx_gui_runner.py` runner remains a diagnostic
> tool, not the SOP entry point.

---

## 1. Official Unitree G1 reference flow (two terminals)

Unitree's `unitree_rl_lab` sim2sim runs **two** processes that talk over DDS:

| Terminal | Process | Role |
|----------|---------|------|
| A | `unitree_mujoco` | MuJoCo sim + DDS bridge; owns the **elastic band** (`simulate/src/main.cc`) and the MuJoCo window |
| B | `g1_ctrl` | Controller; FixStand ramp via `LowCmd`, then the RL policy |

Operator sequence (gamepad + MuJoCo window):

1. Start `unitree_mujoco` with `enable_elastic_band: 1` (band **on**).
2. Start `g1_ctrl`.
3. Gamepad **L2 + Up** → FixStand (controller ramps joints to the stand pose).
4. In the MuJoCo window, press **`8`** repeatedly to **loosen / descend** the band so the
   feet approach the ground; **`7`** **tightens / lifts**.
5. Gamepad **R1 + X** → RL policy (RLBase).
6. In the MuJoCo window, press **`9`** to **toggle the band off**.
7. Robot should keep standing / walking with the band disabled.

> **Key-map note.** `unitree_mujoco/simulate/src/main.cc` uses **8 = loosen/descend,
> 7 = tighten/lift, 9 = toggle band**. This matches the English deploy steps; the
> Chinese mujoco readme labels 7/8 the other way around — we follow the English
> deploy convention.

---

## 2. G0 equivalent (single process)

G0 collapses Unitree's two terminals into **one** viewer process. The elastic
band and the FixStand ramp and the policy all live in
`deploy/robots/g0/main.py`. The band keys `7`/`8`/`9` are handled in the **same**
viewer window — there is no second simulator binary.

Run (conda env `g0_mujoco`):

```bash
cd /home/lz/g0_robot_lab/g0_robot_lab
python -m deploy.robots.g0.main \
  --config deploy/robots/g0/config/policy/velocity/v0/deploy_staging.yaml \
  --duration 120 --realtime
```

`deploy_staging.yaml` differs from `deploy.yaml` only in that it turns the band
**on** by default and adds the staging key bindings + `staging:` ground-confirm
block. Production `deploy.yaml` behavior is unchanged (band off by default).

### G0 operator sequence

1. Start deploy with the **staging** config (band = on).
2. Press **`f`** → FixStand (joint cosine ramp; band calibrates slack on enter).
3. Press **`8`** repeatedly → `L_rest += 0.1 m` per press; feet approach the
   ground (watch `base_z`). Press **`7`** to lift / tighten.
4. Press **`g`** to confirm the feet are on the ground (or enable
   `auto_ground_detect`) → `feet_on_ground = True`.
5. Press **`r`** → RLBase. This transition is **gated** until `feet_on_ground`.
6. With `zero_cmd` stable, press **`9`** → band disabled.
7. Acceptance: RLBase **without** band ≥ 30 s, `root_z > 0.15`, no NaN/inf.

---

## 3. Keyboard map

| Key | Action | Notes |
|-----|--------|-------|
| `p` | Passive (tau = 0) | always allowed |
| `f` | FixStand | ramp to default pose; band calibrates slack on enter |
| `r` | RLBase | gated until `feet_on_ground` when staging enabled |
| `0` | zero velocity command | vx = vy = wz = 0 |
| `7` | band **tighten / lift** | `L_rest -= length_step_m` (default 0.1 m) |
| `8` | band **loosen / descend** | `L_rest += length_step_m` |
| `9` | band **toggle** on/off | disable the band before acceptance |
| `g` | confirm ground | sets `feet_on_ground = True` (manual gate) |
| `w` / `s` | vx +/- 0.1 | forward / back |
| `a` / `d` | vy +/- 0.1 | strafe |
| `q` / `e` | wz +/- 0.2 | yaw |

> The `7`/`8`/`9`/`g` keys only do something when the config provides those
> bindings (staging profile). Production `deploy.yaml` omits them, so its loop is
> unaffected.

### `default_stand` vs `suspended_stand`

- `keyframe: default_stand` (`root_z ≈ 0.23`) — feet start near the ground; press
  `8` only a few times. Default for the staging profile.
- `keyframe: suspended_stand` (optional, Task 6) — robot starts suspended higher
  (`root_z` in `[0.45, 0.55]`); use when you want to demonstrate the full
  band-descend sequence. Requires the band to be strong enough to hold the robot;
  raise `band-anchor` toward Unitree's `z = 3.0` if the suspend is weak.

---

## 4. Acceptance levels

G0 ships these incrementally (L1 → L2 → L3). Unitree's public sim2sim docs
effectively target **L4 (visual demo)** only; G0's CI is **stricter**.

| Level | Criterion | Automation |
|-------|-----------|------------|
| **L1 Process** | FSM + 7/8/9/g + band-disable logic correct | `tests/deployment/test_g0_staging_sop.py`, `tests/deployment/test_g0_staging_fsm_gate.py` |
| **L2 With band** | RLBase + band ≥ 30 s, no NaN, `max_band_force > 0` during RLBase | `tests/deployment/test_g0_deploy_soak.py` (extended soak) |
| **L3 Without band** | After key `9`, RLBase ≥ 30 s, `root_z > 0.15` | `tests/deployment/test_g0_deploy_rlbase_30s_no_band.py` |
| **L4 Visual** | Locomotion comparable to Unitree demo gifs | manual GUI + cmd modes |

### Per-level verification commands

Run from the repo root in the `g0_mujoco` conda env.

```bash
# L1 — process (MUST pass, no xfail): band keys, ground gate, transitions
pytest tests/unit/test_elastic_band_unitree_keys.py \
       tests/unit/test_remote_controller_band_keys.py \
       tests/unit/test_main_band_wiring.py \
       tests/unit/test_gui_runner_band_adapter.py \
       tests/deployment/test_g0_staging_fsm_gate.py \
       tests/deployment/test_g0_staging_sop.py::test_staging_sop_process -v

# L2 — with band (RLBase + band >=30s, no NaN, band force > 0)
pytest tests/deployment/test_g0_deploy_soak.py -v

# L3 — without band (RLBase >=30s, root_z > 0.15). xfail on the v0 policy
#      until Phase 5 / Task 10 fixes standing.
pytest tests/deployment/test_g0_staging_sop.py::test_staging_sop_rlbase_30s_no_band -v
pytest tests/deployment/test_g0_deploy_rlbase_30s_no_band.py -v   # Task 10
```

### L4 manual GUI checklist

Requires a local display (X/Wayland). Launch with the staging profile:

```bash
python -m deploy.robots.g0.main \
  --config deploy/robots/g0/config/policy/velocity/v0/deploy_staging.yaml \
  --duration 120 --realtime
```

- [ ] FixStand: press `f`; robot ramps to the stand pose, band visibly holds it.
- [ ] Lower: press `8` several times; base descends toward the ground (watch `base_z`).
- [ ] Lift: press `7`; base rises (band tightens) — confirms 7/8 direction.
- [ ] Confirm ground: press `g`; console prints `feet_on_ground=True`.
- [ ] Gate: press `r` BEFORE `g` → rejected; after `g` → enters RLBase.
- [ ] Disable band: press `9`; console prints `enabled=False`.
- [ ] Commands: `w/s/a/d/q/e` produce reasonable locomotion response (vs Unitree demo).

### Known v0 policy limitation

The current v0 checkpoint (`logs/rsl_rl/g0_velocity/2026-05-26_18-36-57`) does
**not** yet pass L3. As documented in
`docs/sim2sim/g0_onnx_closed_loop_gui_test_zh.md` **§8**, under `zero_cmd` the
robot's `root_z` drops below 0.10 at **~1.4 s** in MuJoCo and aborts. The
closed-loop control chain itself is verified (obs / mapping / ONNX equivalence
all green); the standing failure is a **policy / MJCF-stance** issue, tracked
separately in Phase 5 (Task 10). Until then, L3 assertions are marked
`xfail` and L1/L2 are the merge gate (Milestone M1).

### Phase 5 / Task 10 root-cause isolation

Systematic isolation (MuJoCo side) pins the failure on the **policy**, not the
physics or the deploy pipeline:

| Probe | Command | Result |
|-------|---------|--------|
| PD-hold (zero action) at `default_stand`, 10 s | `g0_mujoco_zero_action.py` | `root_z=0.231` steady, `max\|dq\|→0` → **physics/pose/contact/PD stable** |
| Closed-loop v0 policy, zero_cmd, no band, 10 s | `g0_mujoco_onnx_gui_runner.py --no-abort` | `root_z` 0.231 → 0.089 (t=1.0s) → ~0.053 fallen; `\|act\|max` 1.3 → 15 |
| Obs / ONNX / mapping | contract scripts | all green (see `g0_onnx_closed_loop_gui_test_zh.md` §2) |

Because PD-hold stands but the closed-loop policy drives itself over with
growing actions, the fix is **policy-side** (retrain / domain-randomize / new
checkpoint, plan Step 4 — needs an Isaac training run), not MJCF/PD tuning.
The canonical L3 gate is `tests/deployment/test_g0_deploy_rlbase_30s_no_band.py`
(`xfail` until a checkpoint passes; the next concrete step is the Isaac
replay-rollout, `g0_action_diagnose.py --phase isaac`, to compare the trained
policy's own rollout against the MuJoCo deploy).
