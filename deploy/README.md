# G0 Deploy (Python, MuJoCo-backed)

Mirrors the `unitree_rl_lab/deploy/` layout but in Python, using
`mujoco-python` and `onnxruntime` directly — no Unitree SDK, no DDS.

## Layout

```
deploy/
  backends/           # MujocoBackend — single MuJoCo contact point (swap for real robot)
  common/             # rotation utils, keyboard "remote controller"
  isaaclab/
    envs/             # ManagerBasedRLEnv
    managers/         # ObservationManager / ActionManager / VelocityCommandManager
    algorithms/       # OrtRunner (onnxruntime)
  fsm/                # CtrlFSM + StatePassive / StateFixStand / StateRLBase
  robots/g0/
    main.py
    config/policy/velocity/v0/deploy.yaml
    config/policy/velocity/v0/policy.onnx -> logs/.../exported/policy.onnx
```

## Run

```bash
cd /home/lz/g0_robot_lab/g0_robot_lab
python -m deploy.robots.g0.main \
    --config deploy/robots/g0/config/policy/velocity/v0/deploy.yaml \
    --duration 60 --realtime
```

Keys in the viewer:

| Key | Action |
|-----|--------|
| `p` | request Passive (tau=0) |
| `f` | request FixStand (ramp to default pose) |
| `r` | request RLBase (run policy.onnx) |
| `0` | zero velocity command |
| `w/s` | vx +/- 0.1 |
| `a/d` | vy +/- 0.1 |
| `q/e` | wz +/- 0.2 |

Transitions are gated by FSM rules (`fsm/state_*.py`):
- `passive -> fix_stand` always allowed
- `fix_stand -> rl_base` allowed after ramp completes
- `rl_base -> passive` allowed (emergency stop)
- `fix_stand -> passive` allowed
- All other transitions rejected.

## Acceptance (sim2sim, before real robot)

1. Passive — robot tau=0, body falls / rests; loop stable, no NaN.
2. FixStand — from any initial pose, joints reach default within ramp_time_s.
3. RLBase — policy.onnx closed loop runs >=30s without divergence (with elastic
   band on for now; goal: without band, matching Unitree G1's baseline).

Headless soak test (runs the full pipeline + drives the FSM through all three
states without a display; also asserts the elastic-band crutch is wired up):

```bash
cd /home/lz/g0_robot_lab/g0_robot_lab
pytest tests/deployment/test_g0_deploy_soak.py -v
```

## Sim2sim staging (Unitree-aligned)

For the Unitree-style staging flow (elastic band on → FixStand → lower feet with
key `8` → confirm ground with `g` → RLBase → disable band with `9`), use the
**staging** profile instead of `deploy.yaml`:

```bash
cd /home/lz/g0_robot_lab/g0_robot_lab
python -m deploy.robots.g0.main \
    --config deploy/robots/g0/config/policy/velocity/v0/deploy_staging.yaml \
    --duration 120 --realtime
```

`deploy_staging.yaml` turns the elastic band **on** by default, adds the
`7`/`8`/`9`/`g` band+ground keys, and a `staging:` ground-confirm gate.
Production `deploy.yaml` is unchanged (band off, no staging keys).

Full operator SOP and the L1–L4 acceptance levels:
[`docs/sim2sim/g0_unitree_staging_sop_en.md`](../docs/sim2sim/g0_unitree_staging_sop_en.md).

## What this does NOT do

- Train. Use `scripts/sim2sim/g0_train_dr_dryrun.py` and IsaacLab for that.
- Real robot. `MujocoBackend` will be swapped for a `RealBackend` later.
- DDS. We're going Python/MuJoCo direct.
