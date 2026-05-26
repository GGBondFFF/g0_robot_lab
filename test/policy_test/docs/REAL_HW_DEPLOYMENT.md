# G0 Real-Hardware Deployment Guide

Step-by-step deployment of the velocity locomotion policy
(`logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/exported/policy.onnx`) onto the
real G0 humanoid. Validated stack-mate of `policy_test/` (sim-only) — same
bridges, same wire format, same TOML.

Every step has a **prerequisite**, **command**, **acceptance**, and **abort**
section. **Never skip an acceptance check. Never proceed past an abort
condition without root-causing it.**

Notation:
- `>>>` = command to type on the listed host
- `[PASS]` = acceptance condition (must observe all)
- `[ABORT]` = if you see this, hit e-stop / power-off and stop here

---

## 0. Background

Sim and real hardware share one binary protocol:
| Signal           | Sim (g0_sim/*)                 | Real HW (mc/*)            |
|------------------|--------------------------------|---------------------------|
| Operator command | `g0_sim/robot_control`         | `robot_control` (no prefix) |
| Motor cmd        | `g0_sim/motor_control`         | `mc/motor_control`        |
| Motor state      | `g0_sim/motor_state`           | `mc/motor_state`          |

On real HW the robot daemon (motion-planner robot-side) is what publishes
`mc/motor_state` and subscribes `mc/motor_control`. **Our policy replaces
the motion-planner brain.** We keep:
- `policy_control_bridge` — DDS<->UDS bridge for the Python policy.
- `policy_inference.py` — runs `policy.onnx`.
- `fake_control_agent` — operator command source (or motion-planner's own
  agent, same `robot_control` schema).

We **do not** run:
- `isaac_gui_plant.py` — real robot replaces it.
- `plant_state_bridge` — real robot daemon replaces it.

All the binaries that touch DDS now have a hard guard on `mc/` topics. To
talk to real hardware you must pass `--allow-real-hw` explicitly. The
binaries print a loud warning when the flag is set.

---

## 1. Pre-flight checklist (before powering the robot)

### Prerequisites
- Robot on a **test stand** with feet **off the ground** (or harness-suspended).
  Locomotion policies output gait-driven leg motion the instant they start.
- E-stop physically within reach.
- Computer on the same DDS multicast domain (LAN / loopback) as the robot.
  Confirm `cyclonedds` is configured to use the same network interface the
  robot daemon does — check `CYCLONEDDS_URI` env var.
- All policy_test binaries built and up to date:

  >>> `cd ~/g0_robot_lab/g0_robot_lab/test/policy_test/cpp && cmake --build build -j`

- The exported policy is current (sim acceptance passed):

  >>> `ls -lh ~/g0_robot_lab/g0_robot_lab/logs/rsl_rl/g0_velocity/2026-05-14_18-29-19/exported/policy.onnx`

### [PASS]
- All four binaries present and recently built:
  - `cpp/build/policy_control_bridge`
  - `cpp/build/sniff_motor_control`
  - `cpp/build/fake_control_agent`
  - `~/g0_robot_lab/g0_robot_lab/test/single_joint_test/cpp/build/single_joint_sender`
- `policy.onnx` mtime matches the training run you intend to deploy.
- Test stand verified, e-stop verified.

### [ABORT]
- Robot is on the floor, free-standing, or not isolated. Move to test stand.
- E-stop not within arm's reach.
- Network unclear → stop, fix DDS reachability before powering robot.

---

## 2. Phase A — Passive comms verification (no commands sent)

Power the robot. Let the robot daemon come up so `mc/motor_state` is being
published. **Do nothing else yet.**

### A.1 Sniff one frame of `mc/motor_state` shape

This proves DDS reachability and that the byte layout matches our IDL.

>>> Terminal A:
```
cd ~/g0_robot_lab/g0_robot_lab/test/policy_test
./cpp/build/sniff_motor_control --topic mc/motor_state --count 1 --skip 0 \
    --out docs/realhw_motor_state_baseline.log --allow-real-hw
```

*Note:* `sniff_motor_control` was written for `MotorControl::Control`. For
`MotorControl::State` we need its sister. If `realhw_motor_state_baseline.log`
shows reasonable joint positions (degrees, magnitudes plausible for the
robot's stance), this acceptance is met. If not, build the state sniffer:

>>> (one-time) ask Claude to add `sniff_motor_state.cpp` mirroring the
control sniffer.

### A.2 Sniff one frame of `mc/motor_control` from motion-planner

If motion-planner's own brain is currently publishing, you can observe what
it sends:

>>> `./cpp/build/sniff_motor_control --topic mc/motor_control --count 1 \
    --out docs/realhw_mp_motor_control_baseline.log --allow-real-hw`

### [PASS]
- `realhw_motor_state_baseline.log` contains one frame with 22 joints, each
  with `pos` within physical range (e.g., knee pitch around the standing
  pose written in `g0.py`).
- `sequence_id` field present, `timestamp_ns` recent (within minutes).

### [ABORT]
- Sniffer hangs forever → DDS comms not working. Check
  `CYCLONEDDS_URI`, firewall, IP routing.
- Frame `motor_count != 22` or `pos` values are NaN / wildly out of range
  → IDL version mismatch or a corrupted publisher. Stop, do not proceed.

---

## 3. Phase B — Single-joint real-HW test (low-risk joint, tiny amplitude)

Goal: prove sender → DDS → robot → joint actually moves the correct joint
in the correct direction by the correct amount.

We use `single_joint_sender` from `single_joint_test`. It already has the
sim_sign-aware single-slot fill that puts the commanded delta in **one**
slot and zero in the other 21.

### B.1 Choose a safe joint

Recommended first joint: **`l_elbow_pitch_joint` (motor_id=6)**. Reasons:
- Low torque, small inertia.
- Free-air motion (arm), no ground contact.
- Visually obvious (you can watch the elbow flex).
- Symmetric with right arm — easy second test.

Avoid until comfortable: knees, hips, ankles (load-bearing).

### B.2 Dry-run first (no DDS write, just verify the frame the sender would send)

>>> Terminal B:
```
cd ~/g0_robot_lab/g0_robot_lab/test/single_joint_test
./cpp/build/single_joint_sender \
    --motor-id 6 --pos 3.0 \
    --baseline default \
    --rate-hz 50 --duration-sec 2 \
    --dry-run true
```

This builds a 22-slot frame with slot 6 = (default + 3°) × sim_sign, all
other 21 slots = `kDefaultPosByMotorIdDeg`, and prints the would-be frame
without publishing.

### [PASS — B.2]
- 22-row table prints, slot 6 `pos` ≈ `default_l_elbow_pitch_deg + 3.0`.
- Other 21 slots `pos` = their default-pose values from
  `pd_gains_generated.hpp`.
- `kp/kd` per slot match `g0.py` (deg-domain, e.g. `l_knee_pitch_joint`
  kp=0.069813).

### B.3 LIVE single-joint command (small amplitude, ramp first)

You must:
- Have the arm clear of obstacles.
- Have one hand on the e-stop.
- **Confirm the joint zero-position visually before sending.**

First try a **3° step from default**:

>>> Terminal B:
```
./cpp/build/single_joint_sender \
    --motor-id 6 --pos 3.0 \
    --baseline default \
    --rate-hz 50 --duration-sec 5 \
    --topic mc/motor_control \
    --allow-real-hw
```

`--duration-sec 5` = 5 seconds of commands at 50 Hz, then the sender exits and
the robot daemon should hold the last commanded pose (or revert to its
safe default depending on firmware behaviour).

### [PASS — B.3]
- The l_elbow_pitch joint moves approximately 3° from its initial pose
  within ~0.5 s.
- No other joint moves visibly.
- Sender prints its summary and exits cleanly after ~5 s.
- Sniff a frame mid-run (separate terminal) to confirm what's on the wire:

  >>> `./cpp/build/sniff_motor_control --topic mc/motor_control \
      --count 1 --out docs/realhw_sj_b3.log --allow-real-hw`

  → slot 6 (`l_elbow_pitch_joint`) has the commanded value.

### [ABORT — B.3]
- Wrong joint moves → motor_id-to-name mapping is wrong on the real robot.
  Stop, diff `joint_mapping.hpp` against the robot's firmware mapping table.
- Joint moves in the wrong direction by ≈3° → `sim_sign_observed` for that
  joint disagrees with the real robot. The sim test stand mirrored real
  behaviour for this exact reason; if it now diverges, check whether the
  robot has been re-fitted with mirrored mechanics or whether the firmware
  applies its own sign.
- Joint overshoots / oscillates → kp/kd on the wire don't match what
  firmware expects. Confirm firmware reads deg-domain gains (N·m/deg), not
  rad-domain.
- Multiple joints move → frame layout mismatch. Diff
  `g0_pt::MotorControlFrame` byte order against firmware expectation.

### B.4 Repeat B.3 on the right arm (motor_id=10, l_elbow → r_elbow)

Confirms left/right symmetry holds.

### B.5 Repeat with one leg joint (when comfortable)

Start with `l_hip_yaw_joint` (motor_id=13) at ±2° — yaw is the safest leg
DOF because it doesn't cause the robot to lift its own weight.

---

## 4. Phase C — Hold-default smoke test (policy in dry-run, no motion intent)

Goal: prove the full policy → DDS pipeline reaches the robot, but with
zero policy output so the robot just receives "stay at default pose"
commands.

### C.1 Bridge

>>> Terminal 1:
```
cd ~/g0_robot_lab/g0_robot_lab/test/policy_test
./cpp/build/policy_control_bridge \
    --topic-rc robot_control \
    --topic-ms mc/motor_state \
    --topic-mc mc/motor_control \
    --allow-real-hw
```

UDS paths default to `/tmp/g0_sim/*.sock` — bridge owns them; the policy
process connects as client.

### C.2 Fake operator (or use motion-planner's own publisher)

>>> Terminal 2:
```
./cpp/build/fake_control_agent \
    --config config/fake_control_agent.toml \
    --topic robot_control
```

Make sure `config/fake_control_agent.toml` has `vx=vy=vz=yaw_rate=0`.

### C.3 Policy in dry-run

>>> Terminal 3 (in g0_isaaclab conda env):
```
conda activate g0_isaaclab
cd ~/g0_robot_lab/g0_robot_lab/test/policy_test/python
python policy_inference.py \
    --config ../config/policy_inference_real_hw.toml \
    --dry-run
```

Dry-run forces all 22 raw actions to zero, so the wire `pos` = default pose
(in deg). Joints should hold their default pose.

### [PASS — C]
- `policy_control_bridge` prints incrementing `rc_in`, `ms_in`, `mc_out`.
- `policy_inference.py` prints `cmd=[0,0,0]` and `|action|=0.000`.
- Sniff one frame:

  >>> `./cpp/build/sniff_motor_control --topic mc/motor_control --count 1 \
      --out docs/realhw_phase_c.log --allow-real-hw`

  Every slot's `pos` = its g0.py default pose ± rounding; `kp/kd` populated.
- Robot **does not move** beyond holding default pose. Slight settling
  motion is OK.

### [ABORT — C]
- Robot drifts away from default pose → joint reorder is wrong on real HW
  (the wire/SDK/URDF lists in `joint_mapping.py` may not match the robot
  firmware's slot order). Stop, hit e-stop, diff orders.
- Robot oscillates → kp/kd magnitudes wrong; or firmware applies an
  additional sign flip. Stop, lower kp via custom TOML, re-test.
- `cmd=[0,0,0]` but `|action|` nonzero in non-dry-run → you forgot
  `--dry-run`.

---

## 5. Phase D — Policy live, zero command, robot held / suspended

Goal: enable the trained policy and verify gait-phase oscillation at
**zero command** stays bounded.

### D.1 Confirm test stand isolation

- Feet off the ground, OR
- Body suspended in harness with foot free to oscillate.

**Do not** allow the feet to touch the ground in this phase yet — the
policy will produce small stepping motions even at zero command (gait
phase keeps advancing) and we don't want unintended ground reaction
forces in the first live run.

### D.2 Re-run the same stack as Phase C, but **without** `--dry-run`

>>> Terminal 1: same `policy_control_bridge` (already running, leave it)

>>> Terminal 2: same `fake_control_agent` (zero command, already running)

>>> Terminal 3 (replace dry-run command):
```
python policy_inference.py \
    --config ../config/policy_inference_real_hw.toml
```

Note `policy_inference_real_hw.toml` has `action_clip = 3.0` (tighter than
sim's 5.0) as a real-HW safety net.

### [PASS — D]
- `|action|` printed every second is bounded in roughly `[0.5, 4.5]`.
- All four legs / arms perform a small cyclic motion at ~0.8 s period
  (matches `[obs.gait].period_s` in the TOML).
- No joint reaches its mechanical limit.
- After 30 seconds the robot is still alive, comms healthy, no temperature
  flag in `mc/motor_state` (sniff one frame to check `fpc_temper` /
  `pcb_temper`).

### [ABORT — D]
- `|action|` grows monotonically or exceeds the clip ceiling repeatedly
  → obs alignment is wrong on real HW even though sim worked. Re-check:
  - Is `mc/motor_state` joint slot order the same as our wire order?
    Sniff one frame and compare.
  - Is `imu` populated by the robot in the same convention the sim plant
    used (acc = `-projected_gravity * g`)?
  - If unsure, set `[runtime].dry_run = true` in TOML, retest C, fix the
    diff, then come back to D.
- Robot trips its own thermal / fault flag → drop `action_scale` (in TOML)
  by half and re-test.

---

## 6. Phase E — Velocity commands, robot still suspended

Goal: prove the command pipeline reaches the policy's gait generator,
incrementally.

### E.1 vx = +0.1 (very slow forward intent)

Edit `config/fake_control_agent.toml`:
```toml
[body]
vx = 0.1
```

Restart only Terminal 2. The policy will pick up the new command on the
very next robot_control packet.

### [PASS — E.1]
- The leg gait amplitude increases visibly compared to zero command.
- `|action|` stays bounded (1–6 range typical).
- `vel_cmd` in `policy_inference.py` log shows `[0.1, 0.0, 0.0]`.

### E.2 vx = +0.3, then +0.5

Stair-step. Wait 10 s at each step. Look for steady gait, not divergence.

### E.3 Negative vx, then lateral vy, then yaw_rate

In order: `vx=-0.3`, `vx=0 vy=0.2`, `vy=0 yaw_rate=10` (deg/s).

Each step: verify the cyclic motion changes in the expected direction.

### [ABORT — E]
- Action diverges or robot fights itself when command changes → policy
  was probably retrained but the obs spec drifted. Re-export to ONNX from
  the same checkpoint and re-test.

---

## 7. Phase F — First ground-contact, vx = 0 only

Goal: let the robot stand on its feet under policy control with **no**
locomotion command, prove it can balance.

### F.1 Prerequisites
- Phase D passed cleanly for 60+ seconds.
- Operator next to e-stop.
- Hands ready to catch (gently) if it falls — but don't intervene unless
  the robot is about to damage itself.

### F.2 Set `fake_control_agent.toml` to all zeros

```toml
[body]
vx = 0.0  vy = 0.0  vz = 0.0  yaw_rate = 0.0
```

Restart Terminal 2. Lower the robot onto its feet **slowly** (test stand
descent / harness loosen). Keep weight on the harness for first 2–3
seconds; transition to full body weight gradually.

### [PASS — F]
- Robot supports own weight without immediate fall.
- Gait stays bounded; small in-place stepping is expected.
- After 30 s, robot is upright.

### [ABORT — F]
- Robot collapses in first second → action_scale too high or kp too low
  for unloaded → loaded transition. Pull back into harness, increase kp
  via TOML override of `emit_pd_gains_on_wire` (or scale `[control]`
  fields), redeploy and re-Phase-D.
- Robot oscillates / shudders → policy was probably trained on a different
  default pose than what's on the robot. Reconcile `G0_DEFAULT_JOINT_POS`
  in g0.py with the real-HW standing pose.

---

## 8. Phase G — Locomotion on the ground

Only after **F passes for 60+ s**. Begin Phase E sequence but on the ground.
Start with `vx=0.1`, walk a few steps, return to zero, re-evaluate.

---

## 9. Stop / cleanup procedure (use whenever ending a session)

In strict order:

1. `fake_control_agent` (Terminal 2): `Ctrl-C` — operator command stops first.
2. `policy_inference.py` (Terminal 3): `Ctrl-C` — policy stops emitting
   motor_control.
3. Lift robot back into harness / test stand.
4. `policy_control_bridge` (Terminal 1): `Ctrl-C`.
5. Power down the robot last.

This ordering matters: stopping the policy before lifting the robot avoids
the robot snapping back to its last-commanded pose (which the firmware
might do for safety).

---

## 10. Files referenced in this guide

| File                                                              | Role |
|-------------------------------------------------------------------|------|
| `cpp/src/policy_control_bridge.cpp`                               | DDS↔UDS bridge for policy |
| `cpp/src/sniff_motor_control.cpp`                                 | Passive DDS sniffer |
| `cpp/src/fake_control_agent.cpp`                                  | Operator command publisher |
| `../single_joint_test/cpp/src/single_joint_sender.cpp`            | Single-joint test sender |
| `python/policy_inference.py`                                      | ONNX policy runner |
| `config/policy_inference_real_hw.toml`                            | Real-HW config (load explicitly) |
| `config/fake_control_agent.toml`                                  | Operator command TOML |
| `docs/dds_action_*.log`                                           | Reference dumps from sim |

All `--allow-real-hw`-aware binaries print a loud warning on startup when
the flag is active. If you don't see that warning, the binary is **not**
talking to real hardware — re-check the command line.
