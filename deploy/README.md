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

### Acceptance levels (sim2sim)

| Level | Criterion | Test |
|-------|-----------|------|
| L1 Process | band keys + ground gate + transitions | `test_g0_staging_sop.py::test_staging_sop_process`, `test_g0_staging_fsm_gate.py` (+ unit tests) |
| L2 With band | RLBase + band ≥30s, no NaN, band force > 0 | `test_g0_deploy_soak.py` |
| L3 Without band | after key `9`, RLBase ≥30s, `root_z > 0.15` | `test_g0_staging_sop.py::test_staging_sop_rlbase_30s_no_band`, `test_g0_deploy_rlbase_30s_no_band.py` (xfail on v0 policy) |
| L4 Visual | locomotion vs Unitree demo | manual GUI (see SOP doc checklist) |

L3 is the M2 sign-off gate and is currently `xfail` (v0 policy fall, Phase 5 / Task 10).

## Real robot (kingkong G0 over CycloneDDS / mbus)

The real G0 is a **custom kingkong robot — NOT Unitree**. It speaks **CycloneDDS**
via the `mbus` library (`MotorControl::Control` / `MotorControl::State` topics),
and its motors run **MIT-mode onboard PD**. `RealBackend` swaps in for
`MujocoBackend`; the policy / obs / action / FSM layers are unchanged.

> First-bringup scope: only the two motor topics (`mc/motor_control`,
> `mc/motor_state`). Orientation is derived from the raw accelerometer
> (valid quasi-static — fine for standing; switch to the fused
> `Imu::ImuOutput.q` topic for walking).

### Topology (who runs what)

```
   YOUR LAPTOP                              ROBOT (onboard SoC, RK3576)
 ┌─────────────────────┐   CycloneDDS    ┌────────────────────────────┐
 │ RL policy + RealBackend│  domain 0     │ mc_forwarder.service        │
 │  send mc/motor_control ───────────────▶│   reads cmd → drives motors │
 │  recv mc/motor_state  ◀───────────────│   reads encoders+IMU → state│
 │ real_state_probe (read-only)          │                            │
 └─────────────────────┘  WiFi / eth     └────────────────────────────┘
                                          motion-planner.service: MUST be stopped
```

**Everything in this repo runs on your laptop.** DDS is location-transparent —
the probe/backend just subscribe to what the robot publishes over the network.
**No script here runs on the robot.**

### Robot-side prerequisites (over ssh)

The robot's low-level DDS↔motor bridge is `mc_forwarder.service` (kingkong
firmware — you don't write or run it manually, just ensure it's up). The
high-level gait brain `motion-planner.service` **also publishes
`mc/motor_control`**, so it will fight your laptop — stop it.

```bash
ssh <robot> "sudo systemctl stop motion-planner"
ssh <robot> "systemctl is-active mc_forwarder"   # must print: active
```

`mc_forwarder` publishes `mc/motor_state` on its own (just reading sensors), so
the read-only probe works with zero commands sent — motors never move.

### Laptop-side prerequisites

The deploy/probe code is **all Python** — DDS is language-neutral on the wire, so
Python (cyclonedds) interoperates with the robot's C++ `mc_forwarder`. **No C++,
no `cpp/` folder.** Install into the **same env that runs the policy** (the one
with `onnxruntime`, e.g. `g0_mujoco`):

```bash
export CYCLONEDDS_HOME=/home/lz/ws/dds-install     # the prebuilt CycloneDDS C lib
pip install cyclonedds                             # eclipse python binding (one pkg)
pip install /home/lz/ws/mbus/python                # kingkong mbus python package
```

At **runtime** the python binding must find the C lib, and CycloneDDS reads its
network config from `CYCLONEDDS_URI`:

```bash
export LD_LIBRARY_PATH=/home/lz/ws/dds-install/lib:$LD_LIBRARY_PATH
export CYCLONEDDS_URI=file:///home/lz/ws/mbus/install/etc/mbus/config/mbus_config.xml
```

Two XMLs, **distinct roles** — you do NOT need to `sudo cp` to `/etc`:
- **`mbus_config.xml`** → network/discovery (interfaces, multicast, peers).
  Consumed via the `CYCLONEDDS_URI` env var above.
- **`mbus_qos.xml`** → QoS profiles. Passed explicitly: probe `--qos-file ...`,
  or yaml `real.qos_file: ...`. **Required** — the motor topics are
  `BEST_EFFORT`; with the default (RELIABLE) QoS the reader will NOT match
  `mc_forwarder` and you get **zero frames**.

> Verified on the laptop (no robot needed): loopback pub/sub of a full
> `MotorControl::State` round-trips with all fields intact.

### Network (read before WiFi)

`mbus_config.xml` uses **multicast discovery** (`AllowMulticast=true`, no static
peers; interfaces `lo`/`wlan0`/`eth0`), domain **0**.

- **Many WiFi APs drop multicast** → the laptop never discovers the robot and
  the probe sees no frames. Either use a router with working IGMP/multicast, or
  add the robot as a unicast peer in your `mbus_config.xml` (then point
  `CYCLONEDDS_URI` at it):
  ```xml
  <Discovery><Peers><Peer address="ROBOT_IP"/></Peers></Discovery>
  ```
- A 50 Hz balance policy closing the loop over WiFi suffers latency + jitter.
  **For first standing bringup use a wired Ethernet link (eth0).** WiFi later.

### Bringup checklist (in order)

1. **Connectivity (read-only, safe).** Robot powered, `mc_forwarder` up,
   `motion-planner` stopped, laptop wired to robot. On the laptop (env vars from
   "Laptop-side prerequisites" already exported):
   ```bash
   python -m deploy.tools.real_state_probe \
       --qos-file /home/lz/ws/mbus/install/etc/mbus/config/mbus_qos.xml
   ```
   Expect: frames arriving, `motor_count=22`, IMU valid. The probe prints a
   verdict and resolves the three unknowns below. If no frames → see Network.

2. **Resolve the 3 hardware unknowns** (TODOs in `backends/real_backend.py`):
   | Unknown | How the probe tells you |
   |---|---|
   | `BUS_MOTOR_ORDER` | `--watch`, then hand-wiggle ONE joint; the moving index is its bus slot |
   | `IMU_GYRO_IN_DEG` | rotate base ~90° in ~1 s: peak `~1.5` → rad/s, `~90` → deg/s |
   | `ACC_SIGN` | standing upright, `proj_grav` must be `~[0,0,-1]`; flip `--acc-sign` if not |
   Also confirm `kp/kd` units (Nm/rad vs Nm/deg) on a stand before trusting gains.

3. **Run against the robot.** `main.py` takes `--backend real`: it swaps in
   `RealBackend` + `RealEnv` (one MIT command per policy step, paced to
   `step_dt`), forces the elastic band **off**, starts in **Passive** (damping),
   and reads `p`/`f`/`r` from stdin (no viewer). `p` is the software E-stop
   (damping, not tau=0).
   ```bash
   python -m deploy.robots.g0.main \
       --config deploy/robots/g0/config/policy/velocity/v0/deploy.yaml \
       --backend real --duration 120
   ```
   Optional `real:` block in the yaml: `domain_id`, `bus_motor_order`,
   `imu_gyro_in_deg`, `acc_sign` (the values resolved in step 2).

4. **Suspended/stand test first.** Hang the robot or keep it on a stand. From
   Passive press `f` (FixStand: PD ramp to default pose); only then `r` (RLBase).
   Keep a hand on the E-stop / `p` (damping) at all times.

## What this does NOT do

- Train. Use `scripts/sim2sim/g0_train_dr_dryrun.py` and IsaacLab for that.
- Real-robot **walking** yet. `RealBackend` (skeleton) covers standing via the
  motor topics + acc-derived gravity; walking needs the fused IMU quaternion
  and the hardware unknowns resolved (see checklist), plus `main.py` wiring.
