"""Read-only connectivity probe for the real G0 over CycloneDDS (mbus).

SAFE ON A POWERED ROBOT: this script ONLY subscribes to `mc/motor_state`.
It NEVER registers a publisher and NEVER writes `mc/motor_control`, so it
cannot move a single motor. Run it FIRST, before anything that commands.

Purpose — verify the three unknowns the RealBackend has TODOs for:
  (1) BUS_MOTOR_ORDER : wiggle one joint by hand, watch which index moves.
  (2) IMU_GYRO_IN_DEG : gyro ~0 at rest; rotate the base and read the peak.
  (3) ACC_SIGN        : standing upright, confirm derived proj_grav ~ [0,0,-1].

Also checks: frame arrival + rate, motor_count, per-motor isvalid/status,
temperatures, acc magnitude (m/s^2 vs g), IMU valid flags.

Usage:
    python -m deploy.tools.real_state_probe                  # 10 s, domain 0
    python -m deploy.tools.real_state_probe --duration 30
    python -m deploy.tools.real_state_probe --watch          # live per-motor pos
"""

import argparse
import math
import os
import sys
import time

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from deploy.common.rotation import (
    quat_apply_inverse_wxyz, quat_to_rpy_wxyz, quat_wxyz_from_gravity)

_RAD2DEG = 180.0 / math.pi


def _import_mbus():
    try:
        from mbus.wrapper import MbusNode
        from mbus.idl.MotorControl import State
        return MbusNode, State
    except Exception as e:
        print("ERROR: mbus python bindings not importable:", e)
        print("  Need: cyclonedds>=11, cyclonedds-python>=11, libmbus>=2.0")
        print("  This script must run on the robot/deploy machine where mbus lives.")
        sys.exit(2)


def _vec(v):
    return np.array([v.x, v.y, v.z], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--domain", type=int, default=0)
    ap.add_argument("--motor-count", type=int, default=22)
    ap.add_argument("--acc-sign", type=float, default=-1.0,
                    help="sign for gravity derivation; flip if upright proj_grav != [0,0,-1]")
    ap.add_argument("--watch", action="store_true",
                    help="live-print all motor positions (for joint-id finding)")
    args = ap.parse_args()

    MbusNode, State = _import_mbus()

    print("=" * 78)
    print("REAL G0 STATE PROBE  (READ-ONLY — never publishes, cannot move motors)")
    print(f"  domain={args.domain}  topic=mc/motor_state  duration={args.duration}s")
    print("=" * 78)

    node = MbusNode(domain_id=args.domain)
    state_topic = node.register_topic(
        "mc/motor_state", State, "mbus::MotorControl_State")

    n = args.motor_count
    t_end = time.monotonic() + args.duration
    n_frames = 0
    last_seq = None
    seq_gaps = 0
    t_first = None
    last_print = 0.0

    # accumulators for the end-of-run verdict
    acc_sum = np.zeros(3)
    gyro_sum = np.zeros(3)
    gyro_absmax = np.zeros(3)
    acc_frames = 0
    imu_valid_frames = 0
    pos_min = np.full(n, np.inf)
    pos_max = np.full(n, -np.inf)
    isvalid_count = np.zeros(n, dtype=np.int64)
    last = None

    print("\nwaiting for first mc/motor_state frame ...")
    while time.monotonic() < t_end:
        samples = state_topic.read(1)
        if not samples:
            time.sleep(0.002)
            continue
        s = samples[0]
        # guard against InvalidSample / partial samples
        try:
            mc = int(s.motor_count)
            seq = int(s.sequence_id)
        except Exception:
            continue

        if t_first is None:
            t_first = time.monotonic()
            print(f"  first frame OK: motor_count={mc}  sequence_id={seq}")
            if mc != n:
                print(f"  WARNING: motor_count={mc} != expected {n}")
        n_frames += 1
        last = s
        if last_seq is not None and seq != ((last_seq + 1) & 0xFFFF):
            seq_gaps += 1
        last_seq = seq

        m = min(mc, n)
        for i in range(m):
            mi = s.motors[i]
            p = float(mi.pos)
            pos_min[i] = min(pos_min[i], p)
            pos_max[i] = max(pos_max[i], p)
            if int(mi.isvalid):
                isvalid_count[i] += 1

        imu = s.imu
        if int(imu.imu_data_valid):
            imu_valid_frames += 1
        a = _vec(imu.acc)
        g = _vec(imu.gyro)
        acc_sum += a
        gyro_sum += g
        gyro_absmax = np.maximum(gyro_absmax, np.abs(g))
        acc_frames += 1

        now = time.monotonic()
        if args.watch and now - last_print > 0.3:
            last_print = now
            pos = np.array([float(s.motors[i].pos) for i in range(m)])
            print("pos(deg): " + " ".join(f"{i:02d}:{pos[i]:+7.1f}" for i in range(m)))
        elif not args.watch and now - last_print > 1.0:
            last_print = now
            rate = n_frames / max(now - t_first, 1e-6)
            print(f"  t={now - t_first:5.1f}s  frames={n_frames}  rate~{rate:5.1f}Hz  "
                  f"acc={np.round(a,2)}  gyro={np.round(g,3)}")

    # ---------------- verdict ----------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if n_frames == 0 or last is None:
        print("  ✗ NO frames received. Check: robot powered + motor service up,")
        print("    same DDS domain (--domain), /etc/mbus/config/*.xml present,")
        print("    network/loopback (CYCLONEDDS_URI), firewall.")
        return

    dur = time.monotonic() - t_first
    print(f"  ✓ frames={n_frames}  rate~{n_frames/max(dur,1e-6):.1f}Hz  "
          f"seq_gaps={seq_gaps}")
    print(f"  motor_count={int(last.motor_count)} (expected {n})")

    valid_motors = int(np.sum(isvalid_count > 0))
    print(f"  motors reporting isvalid>0: {valid_motors}/{n}")
    moved = [i for i in range(n) if (pos_max[i] - pos_min[i]) > 0.5]
    if args.watch:
        print(f"  joints that MOVED during run (range>0.5deg): {moved}")
        print("  -> wiggle ONE joint by hand; the index that moves is its bus slot.")

    acc_mean = acc_sum / max(acc_frames, 1)
    acc_mag = float(np.linalg.norm(acc_mean))
    print("\n  IMU:")
    print(f"    imu_data_valid frames: {imu_valid_frames}/{n_frames}")
    print(f"    acc mean = {np.round(acc_mean,3)}  |acc| = {acc_mag:.3f}")
    if 8.5 < acc_mag < 11.0:
        print("      -> |acc| ~ 9.8  => units are m/s^2")
    elif 0.8 < acc_mag < 1.2:
        print("      -> |acc| ~ 1.0  => units are g")
    else:
        print("      -> unexpected magnitude; robot maybe not at rest / units odd")
    gyro_mean = gyro_sum / max(acc_frames, 1)
    print(f"    gyro mean = {np.round(gyro_mean,4)} (≈0 expected at rest)")
    print(f"    gyro |max| per axis = {np.round(gyro_absmax,3)}")
    print("      -> rotate base ~90deg over ~1s: peak ~1.5 => rad/s, ~90 => deg/s")

    q = quat_wxyz_from_gravity(acc_mean, sign=args.acc_sign)
    pg = quat_to_rpy_wxyz(q)
    pgrav = quat_apply_inverse_wxyz(q, [0, 0, -1.0])
    print(f"\n  derived (acc_sign={args.acc_sign:+.0f}): proj_grav={np.round(pgrav,3)}  "
          f"rpy=({pg[0]*_RAD2DEG:+.1f},{pg[1]*_RAD2DEG:+.1f},{pg[2]*_RAD2DEG:+.1f})deg")
    if abs(pgrav[2] + 1.0) < 0.15:
        print("      ✓ standing upright proj_grav ~ [0,0,-1]  => ACC_SIGN is correct")
    else:
        print("      ✗ proj_grav not ~[0,0,-1]. If robot IS upright, re-run with "
              f"--acc-sign {-args.acc_sign:+.0f}")
    print("\n  next: set BUS_MOTOR_ORDER, IMU_GYRO_IN_DEG, ACC_SIGN in real_backend.py")


if __name__ == "__main__":
    main()
