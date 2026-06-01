# G0 Unitree Sim2Sim Staging Alignment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align G0 sim2sim **operational flow and acceptance gates** with the Unitree G1 pattern (elastic band → FixStand → lower to ground → RLBase → disable band → stable closed loop without band), **without** introducing `unitree_mujoco` or `unitree_sdk2` in this plan.

**Architecture:** Keep the existing single-process Python stack (`deploy/robots/g0/main.py` + `MujocoBackend`). Extend `ElasticBand` and `RemoteController` for Unitree 7/8/9 semantics; add a `deploy_staging.yaml` profile and a documented SOP; automate the SOP headlessly in tests. Policy/MJCF tuning is a separate Phase 5 track and does not block Phase 1–3 “process alignment.”

**Tech Stack:** Python 3, `mujoco`, `onnxruntime`, `pytest`, existing `deploy/fsm/*` and `scripts/sim2sim/*` contract toolchain.

**Explicitly out of scope for this plan:**
- Two-process DDS + [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
- C++ `g1_ctrl` and production `LowCmd` wire format (G0 uses a custom motor frame; see `final-virtual-dds-gate` plan)

---

## References

| Resource | Role |
|----------|------|
| [unitree_rl_lab — Deploy / Sim2Sim](https://github.com/unitreerobotics/unitree_rl_lab) | Official G1 sim2sim SOP: `unitree_mujoco` + `g1_ctrl`, band on, L2+Up, key 8, R1+X, key 9 |
| [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) | MuJoCo sim + DDS bridge; elastic band in `simulate/src/main.cc` |
| `g0_robot_lab/deploy/` | Python mirror of `unitree_rl_lab/deploy/` layout |
| `g0_robot_lab/scripts/sim2sim/` | Contract/diagnostic layer (Unitree has no equivalent) |
| `docs/sim2sim/g0_onnx_closed_loop_gui_test_zh.md` | Current closed-loop pass/fail criteria |
| `.cursor/plans/final-virtual-dds-gate_*.plan.md` | Future DDS alignment (Phase 6) |

---

## Current Baseline (Plan Starting Point)

| Item | G0 today | Unitree G1 |
|------|----------|------------|
| Elastic band implementation | Yes — `deploy/common/elastic_band.py` (ported from `unitree_mujoco`) | Yes — `simulate/src/main.cc` |
| Band enabled by default for sim2sim | No — `deploy.yaml` `elastic_band.enabled: false` | Yes — `enable_elastic_band: 1` |
| Keys 7 / 8 / 9 (lower / tighten / toggle) | No | Yes — MuJoCo window (`main.cc`) |
| FixStand joint ramp | Yes — `deploy/fsm/state_fix_stand.py` | Yes — `g1_ctrl` via `LowCmd` |
| SOP: feet on ground → policy → disable band | Not documented | Yes — press 8, R1+X, press 9 |
| Acceptance: RLBase ≥30s **without** band | Not met (~1.4s fall, `g0_onnx_closed_loop_gui_test_zh.md` §8) | Demo-level; no numeric CI gate |
| Contract layer (obs/history/ONNX) | Green | N/A |

**Key code anchors (read before implementing):**

- `deploy/common/elastic_band.py` — `rope` / `world_up` modes, `calibrate()`, `update()`
- `deploy/robots/g0/main.py` — `tick()`, FSM loop, optional `--elastic-band on|off`
- `deploy/robots/g0/config/policy/velocity/v0/deploy.yaml` — `elastic_band.enabled: false`
- `deploy/fsm/state_fix_stand.py` — calibrates band on enter; recalibrates after ramp
- `tests/deployment/test_g0_deploy_soak.py` — forces band on; does not assert standing quality in RLBase
- `source/.../mujoco/model_patched.xml` — `default_stand` keyframe at `root_z ≈ 0.23`

---

## Target G0 SOP (Unitree-Aligned Keyboard Map)

Derived from [unitree_rl_lab README](https://github.com/unitreerobotics/unitree_rl_lab) and `unitree_mujoco/simulate/src/main.cc` (**8 = loosen / descend, 7 = tighten / lift, 9 = toggle band** — matches English deploy steps, not the swapped 7/8 labels in the Chinese mujoco readme):

```text
1. Start deploy with staging config (band=on)
2. Press f → FixStand (joint cosine ramp; band calibrates slack on enter)
3. Press 8 repeatedly → L_rest += 0.1 m per press; feet approach ground (watch base_z)
4. Press g (confirm ground) OR auto-detect (optional) → feet_on_ground=True
5. Press r → RLBase (gated until feet_on_ground)
6. zero_cmd stable; press 9 → band.enabled=False
7. Acceptance: RLBase without band ≥30s, root_z > 0.15, no NaN/inf
```

Unitree uses a **gamepad + MuJoCo window** for steps 3 and 6; G0 maps those to keyboard `8` and `9` in the **same** viewer process (no second simulator binary).

---

## File Map (Create / Modify)

| File | Responsibility |
|------|----------------|
| `deploy/common/elastic_band.py` | Add `adjust_rest_length(delta)`, `toggle_enabled()` |
| `deploy/common/staging_context.py` | Small mutable `feet_on_ground: bool` (+ optional auto-detect config) |
| `deploy/common/remote_controller.py` | Band key actions + `confirm_ground` |
| `deploy/robots/g0/main.py` | Consume band actions; print SOP banner |
| `deploy/robots/g0/config/policy/velocity/v0/deploy_staging.yaml` | Sim2sim profile: band on, key bindings, `length_step_m: 0.1` |
| `deploy/fsm/state_fix_stand.py` | Optional staging gate for `rl_base` |
| `docs/sim2sim/g0_unitree_staging_sop_en.md` | English SOP + Unitree comparison table |
| `docs/sim2sim/g0_unitree_staging_sop_zh.md` | Chinese SOP (optional mirror) |
| `tests/unit/test_elastic_band_unitree_keys.py` | Band API unit tests |
| `tests/unit/test_remote_controller_band_keys.py` | Keyboard band action tests |
| `tests/deployment/test_g0_staging_sop.py` | Headless automated SOP |
| `tests/deployment/test_g0_deploy_rlbase_30s_no_band.py` | Final L3 acceptance (Phase 5; may start as xfail) |
| `scripts/sim2sim/g0_mujoco_onnx_gui_runner.py` | Reuse `ElasticBand` API (DRY); `--staging-sop` preset |
| `deploy/README.md` | Link staging profile and acceptance levels |

---

## Acceptance Levels (Stricter Than Unitree Public Docs)

| Level | Criterion | Automation |
|-------|-----------|------------|
| **L1 Process** | FSM + 7/8/9/g + band disable logic correct | `test_g0_staging_sop` |
| **L2 With band** | RLBase + band ≥30s, no NaN, `max_band_force > 0` during RLBase | Extended soak |
| **L3 Without band** | After key 9, RLBase ≥30s, `root_z > 0.15` | `test_g0_deploy_rlbase_30s_no_band` |
| **L4 Visual** | Locomotion comparable to Unitree demo gifs | Manual GUI + cmd modes |

Unitree official sim2sim effectively targets **L4 (visual)**. G0 should ship **L1 → L2 → L3** incrementally.

---

## Phase 0: Documentation and Config Split

### Task 0: Write SOP documentation

**Files:**
- Create: `docs/sim2sim/g0_unitree_staging_sop_en.md`
- Optional: `docs/sim2sim/g0_unitree_staging_sop_zh.md`

- [ ] **Step 1:** Document official Unitree G1 steps (two terminals: `unitree_mujoco`, `g1_ctrl`)
- [ ] **Step 2:** Document G0 equivalent single-process command:

```bash
cd /home/lz/g0_robot_lab/g0_robot_lab
python -m deploy.robots.g0.main \
  --config deploy/robots/g0/config/policy/velocity/v0/deploy_staging.yaml \
  --duration 120 --realtime
```

- [ ] **Step 3:** Keyboard table: `f` / `r` / `p`, `7` / `8` / `9`, `g`, `0`, `w/s/a/d/q/e`
- [ ] **Step 4:** Link acceptance levels L1–L4 and known v0 policy limitation (§8 of `g0_onnx_closed_loop_gui_test_zh.md`)

- [ ] **Step 5: Commit**

```bash
git add docs/sim2sim/g0_unitree_staging_sop_en.md
git commit -m "docs: add Unitree-aligned G0 sim2sim staging SOP"
```

---

### Task 1: Add staging deploy config

**Files:**
- Create: `deploy/robots/g0/config/policy/velocity/v0/deploy_staging.yaml`
- Modify: `deploy/README.md` (staging section only)

- [ ] **Step 1:** Copy `deploy.yaml`; apply staging overrides:

```yaml
# deploy_staging.yaml — key differences from deploy.yaml
keyframe: default_stand   # Task 6 may switch to suspended_stand

elastic_band:
  enabled: true
  anchor: [0.0, 0.0, 2.0]
  stiffness: 50.0
  damping: 10.0
  rest_length: 0.0
  one_sided: true
  mode: rope
  length_step_m: 0.1

staging:
  require_ground_confirm: true
  auto_ground_detect: false   # optional: true after Task 5
  ground_base_z_max: 0.28
  ground_rp_max_rad: 0.35
  ground_hold_s: 0.5

keys:
  passive: "p"
  fix_stand: "f"
  rl_base: "r"
  zero_cmd: "0"
  vx_up: "w"
  vx_down: "s"
  vy_left: "a"
  vy_right: "d"
  wz_left: "q"
  wz_right: "e"
  band_toggle: "9"
  band_tighten: "7"
  band_loosen: "8"
  confirm_ground: "g"
```

- [ ] **Step 2:** README: add “Sim2sim staging (Unitree-aligned)” pointing to `deploy_staging.yaml`

- [ ] **Step 3: Commit**

```bash
git add deploy/robots/g0/config/policy/velocity/v0/deploy_staging.yaml deploy/README.md
git commit -m "feat(deploy): add Unitree-aligned staging deploy profile"
```

---

## Phase 1: Elastic Band API + Keyboard (TDD)

### Task 2: ElasticBand length adjust unit tests

**Files:**
- Create: `tests/unit/test_elastic_band_unitree_keys.py`
- Modify: `deploy/common/elastic_band.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_elastic_band_unitree_keys.py
import numpy as np
import pytest
from deploy.common.elastic_band import ElasticBand


class _FakeBackend:
    def __init__(self, z=1.0):
        self._z = z
        self.last_f = None

    def read_state(self):
        return {
            "base_pos_w": np.array([0.0, 0.0, self._z]),
            "base_lin_vel_w": np.zeros(3),
        }

    def clear_external_force(self):
        pass

    def apply_external_force(self, f):
        self.last_f = np.asarray(f, dtype=np.float64)


def test_adjust_rest_length_matches_unitree_step():
    band = ElasticBand(
        anchor=(0.0, 0.0, 2.0),
        stiffness=50.0,
        damping=10.0,
        rest_length=0.0,
        enabled=True,
        one_sided=True,
    )
    band.adjust_rest_length(+0.1)
    assert band.L == pytest.approx(0.1)
    band.adjust_rest_length(-0.1)
    assert band.L == pytest.approx(0.0)


def test_toggle_enabled():
    band = ElasticBand(enabled=True)
    band.toggle_enabled()
    assert band.enabled is False
    band.toggle_enabled()
    assert band.enabled is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/lz/g0_robot_lab/g0_robot_lab && pytest tests/unit/test_elastic_band_unitree_keys.py -v`  
Expected: FAIL (`AttributeError: adjust_rest_length`)

- [ ] **Step 3: Minimal implementation**

```python
# deploy/common/elastic_band.py — add methods to ElasticBand

def adjust_rest_length(self, delta: float) -> None:
    self.L = float(self.L) + float(delta)

def toggle_enabled(self) -> None:
    self.enabled = not self.enabled
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/unit/test_elastic_band_unitree_keys.py -v`  
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_elastic_band_unitree_keys.py deploy/common/elastic_band.py
git commit -m "feat(deploy): Unitree-style elastic band length adjust API"
```

---

### Task 3: RemoteController band keys

**Files:**
- Modify: `deploy/common/remote_controller.py`
- Create: `tests/unit/test_remote_controller_band_keys.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_remote_controller_band_keys.py
from deploy.common.remote_controller import RemoteController


def test_band_loosen_key():
    rc = RemoteController({
        "passive": "p", "fix_stand": "f", "rl_base": "r", "zero_cmd": "0",
        "vx_up": "w", "vx_down": "s", "vy_left": "a", "vy_right": "d",
        "wz_left": "q", "wz_right": "e",
        "band_toggle": "9", "band_tighten": "7", "band_loosen": "8",
        "confirm_ground": "g",
    })
    rc.on_key("8")
    assert rc.consume_band_action() == "loosen"
    assert rc.consume_band_action() is None


def test_confirm_ground_key():
    rc = RemoteController({
        "passive": "p", "fix_stand": "f", "rl_base": "r", "zero_cmd": "0",
        "vx_up": "w", "vx_down": "s", "vy_left": "a", "vy_right": "d",
        "wz_left": "q", "wz_right": "e",
        "band_toggle": "9", "band_tighten": "7", "band_loosen": "8",
        "confirm_ground": "g",
    })
    rc.on_key("g")
    assert rc.consume_band_action() == "confirm_ground"
```

- [ ] **Step 2: Run test — expect FAIL**

- [ ] **Step 3: Implement `pending_band_action` + `consume_band_action()`**

Use `.get("band_loosen")` on `key_bindings` so missing keys do not break production `deploy.yaml`.

- [ ] **Step 4: Run test — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add deploy/common/remote_controller.py tests/unit/test_remote_controller_band_keys.py
git commit -m "feat(deploy): keyboard band actions for Unitree staging SOP"
```

---

### Task 4: Wire band actions in main.py

**Files:**
- Modify: `deploy/robots/g0/main.py`

- [ ] **Step 1: Import / construct `StagingContext` from yaml `staging:` block (Task 5 may add file; inline dict is OK for first pass)**

- [ ] **Step 2: In `tick()`, before `band.update(backend)`:**

```python
action = rc.consume_band_action()
step = eb_cfg.get("length_step_m", 0.1)
if action == "loosen":
    band.adjust_rest_length(+step)
elif action == "tighten":
    band.adjust_rest_length(-step)
elif action == "toggle":
    band.toggle_enabled()
    print(f"[ElasticBand] enabled={band.enabled}")
elif action == "confirm_ground" and staging is not None:
    staging.feet_on_ground = True
    print("[Staging] feet_on_ground=True (manual confirm)")
```

- [ ] **Step 3: Print SOP banner at startup when `deploy_staging.yaml` or `staging` section present**

- [ ] **Step 4: Manual GUI check**

```bash
python -m deploy.robots.g0.main \
  --config deploy/robots/g0/config/policy/velocity/v0/deploy_staging.yaml \
  --realtime --duration 60
```

- [ ] **Step 5: Commit**

```bash
git add deploy/robots/g0/main.py
git commit -m "feat(deploy): apply Unitree 7/8/9 band keys in main loop"
```

---

## Phase 2: FSM Gates and Optional Suspended Keyframe

### Task 5: `feet_on_ground` gate for RLBase

**Files:**
- Create: `deploy/common/staging_context.py`
- Modify: `deploy/fsm/state_fix_stand.py`
- Modify: `deploy/robots/g0/main.py` (pass `staging` into FSM states)
- Create: `tests/deployment/test_g0_staging_fsm_gate.py`

- [ ] **Step 1: Write failing test**

```python
# tests/deployment/test_g0_staging_fsm_gate.py
# Reuse _build_pipeline from test_g0_deploy_soak.py but load deploy_staging.yaml
# After fix_stand ramp completes: press r -> still fix_stand
# press g, press r -> rl_base
```

- [ ] **Step 2: Implement `StagingContext`**

```python
# deploy/common/staging_context.py
class StagingContext:
    def __init__(self, cfg: dict | None):
        self.cfg = cfg or {}
        self.feet_on_ground = False
        self._ground_timer = 0.0

    def enabled(self) -> bool:
        return bool(self.cfg)

    def maybe_auto_detect(self, backend, step_dt: float) -> None:
        if not self.cfg.get("auto_ground_detect", False):
            return
        z = float(backend.data.qpos[2])
        # roll/pitch from read_state or qpos — use same convention as deploy
        ...
```

- [ ] **Step 3: Update `StateFixStand.check_transition`**

```python
if requested == "rl_base":
    if self.staging is not None and self.staging.enabled() and not self.staging.feet_on_ground:
        print("[FSM] reject rl_base: confirm ground (press 'g') after lowering with '8'")
        return None
    if self._t >= self.ramp_time_s:
        return "rl_base"
```

- [ ] **Step 4: Optional auto-detect in `tick()` when staging enabled**

- [ ] **Step 5: pytest PASS + commit**

---

### Task 6: Optional `suspended_stand` keyframe

**Files:**
- Modify: `source/g0_robot_lab/g0_robot_lab/assets/robots/g0/mujoco/model_patched.xml` (or `scripts/sim2sim/patch_g0_mjcf_for_sim2sim.py`)
- Modify: `deploy_staging.yaml` → `keyframe: suspended_stand`

- [ ] **Step 1:** Add keyframe with `root_z` in `[0.45, 0.55]`, joint qpos matching `default_stand` leg/arm layout

- [ ] **Step 2:** Verify band can hold robot: run `scripts/sim2sim/g0_mujoco_zero_action.py` with band equivalent or deploy staging

- [ ] **Step 3:** Document when to use `suspended_stand` vs `default_stand` in SOP doc

- [ ] **Step 4: Commit**

---

## Phase 3: Headless SOP Automation

### Task 7: `test_g0_staging_sop.py`

**Files:**
- Create: `tests/deployment/test_g0_staging_sop.py`

- [ ] **Step 1:** Copy pipeline builder from `test_g0_deploy_soak.py`; point to `deploy_staging.yaml`

- [ ] **Step 2:** Scripted sequence:

```python
def test_staging_sop_headless(staging_cfg):
    backend, fsm, band, cmd_mgr, rc, step_dt = _build_pipeline(staging_cfg, elastic_band_on=True)

    rc.on_key("f")
    _drive_until_fix_stand_complete(...)

    for _ in range(8):
        rc.on_key("8")
        _tick_one_step(...)

    assert float(backend.data.qpos[2]) < staging_cfg["staging"]["ground_base_z_max"]

    rc.on_key("g")
    rc.on_key("r")
    _drive_rl_base(3.0)

    assert fsm.current.name == "rl_base"

    rc.on_key("9")
    assert band.enabled is False

    _drive_rl_base(30.0)
    # L3 may fail on v0 policy — see Task 10
```

- [ ] **Step 3:** Mark policy-dependent assertion as `@pytest.mark.xfail(strict=False, reason="v0 policy MuJoCo standing")` until Phase 5 passes

- [ ] **Step 4: Run**

```bash
pytest tests/deployment/test_g0_staging_sop.py tests/unit/test_elastic_band_unitree_keys.py -v
```

- [ ] **Step 5: Commit**

---

### Task 8: DRY band logic in `g0_mujoco_onnx_gui_runner.py`

**Files:**
- Modify: `scripts/sim2sim/g0_mujoco_onnx_gui_runner.py`
- Modify: `docs/sim2sim/g0_onnx_closed_loop_gui_test_zh.md` (staging command section)

- [ ] **Step 1:** Replace inline `xfrc_applied` band math with `ElasticBand.update()` via a thin adapter implementing `read_state` / `apply_external_force` / `clear_external_force`

- [ ] **Step 2:** Add `--staging-sop` flag: enables band + prints 7/8/9 help

- [ ] **Step 3:** Commit + update doc

---

## Phase 4: Acceptance Documentation

### Task 9: Publish acceptance matrix

**Files:**
- Modify: `docs/sim2sim/g0_unitree_staging_sop_en.md`
- Modify: `deploy/README.md`

- [ ] **Step 1:** Table L1–L4 (see “Acceptance Levels” above)
- [ ] **Step 2:** Map each level to pytest files and manual GUI checklist
- [ ] **Step 3:** Commit

---

## Phase 5: Root Cause for No-Band Standing (Parallel Debug Track)

> Use superpowers:systematic-debugging. Does not block merging Phase 1–3.

### Task 10: Physics vs policy isolation

**Files:**
- Use existing: `scripts/sim2sim/g0_action_diagnose.py`, `g0_obs_contract_check.py`, `patch_g0_mjcf_for_sim2sim.py`
- Create: `tests/deployment/test_g0_deploy_rlbase_30s_no_band.py`

- [ ] **Step 1:** Run replay-rollout:

```bash
# Phase isaac (g0_isaaclab env) — record trajectory
python scripts/sim2sim/g0_action_diagnose.py --phase isaac --rollout-steps 200 ...

# Phase replay-rollout (g0_mujoco env)
python scripts/sim2sim/g0_action_diagnose.py --phase replay-rollout ...
```

- [ ] **Step 2:** Compare Isaac play vs MuJoCo deploy zero_cmd (10s) — log `root_z`, joint trajectories

- [ ] **Step 3:** Tune MJCF contact/friction/solref and deploy PD gains vs Isaac `JointPositionActionCfg`

- [ ] **Step 4:** If replay is tight but closed-loop fails → retrain or new checkpoint

- [ ] **Step 5:** Implement `test_g0_deploy_rlbase_30s_no_band.py`; remove xfail when L3 green

---

## Phase 6 (Separate Plan): DDS / Two-Process Alignment

Do **not** mix into the same PR as Phase 1–3.

| Option | Description |
|--------|-------------|
| **A. Integrate unitree_mujoco** | Port G0 MJCF into `unitree_robots/`; adapt `unitree_hg` idl; high effort |
| **B. Virtual DDS gate** | Follow `.cursor/plans/final-virtual-dds-gate_*.plan.md`: deploy publishes sandbox motor frames; MuJoCo receiver writes `data.ctrl` |

**Recommended path:** Complete Phase 1–5 (single-process SOP) → Phase 6B (virtual DDS) → real hardware SDK last.

---

## Milestones and Schedule

```text
Week 1: Tasks 0–4  → deploy_staging.yaml + 7/8/9 keys + unit tests
Week 2: Tasks 5–8  → FSM ground gate + headless SOP test + runner DRY
Week 3: Tasks 9–10 → L3 no-band 30s (policy/physics dependent)
Later:  Phase 6     → DDS (separate plan / branch)
```

| Milestone | Definition |
|-----------|------------|
| **M1 — Process aligned with Unitree** | L1 tests green + SOP doc + one successful GUI walkthrough |
| **M2 — Sim2sim sign-off** | L3 test green (RLBase ≥30s without band) |

---

## Decisions for Implementer (No TBD)

1. **Manual `g` vs auto ground detect:** Implement both; default `auto_ground_detect: false` in `deploy_staging.yaml`.
2. **Band anchor height:** Start `z=2.0`; document tuning toward Unitree `z=3.0` in SOP if suspend is weak.
3. **`g0_mujoco_onnx_gui_runner` vs `deploy/main`:** Staging SOP canonical entry is **`deploy/main` + deploy_staging.yaml**; runner remains diagnostic.
4. **Do not delete** `scripts/sim2sim/*` contract tools — they remain the pre-DDS safety net Unitree does not ship.

---

## Self-Review (Plan Quality)

| Check | Status |
|-------|--------|
| Spec: band SOP | Tasks 0–4, 7 |
| Spec: FixStand → ground → RL → disable band | Tasks 5, 7, 9 |
| Spec: acceptance L1–L3 | Tasks 9, 10 |
| Spec: DDS deferred | Phase 6 |
| No placeholder steps | Concrete paths, tests, commands |
| Matches existing codebase | Anchors listed in baseline section |

---

## Execution Handoff

**Plan location:** `docs/superpowers/plans/2026-06-01-unitree-sim2sim-staging-alignment.md`

**Two execution options:**

1. **Subagent-Driven (recommended)** — Fresh subagent per task; review between tasks. REQUIRED: superpowers:subagent-driven-development.

2. **Inline Execution** — Same session, superpowers:executing-plans with checkpoints after Tasks 4, 7, and 10.

**Branch recommendation:** Use superpowers:using-git-worktrees → branch `feat/unitree-staging-sop` before Task 2.

**Which approach do you want?**
