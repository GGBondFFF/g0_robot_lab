"""Open-loop action sweep for G0 sim2sim.

Validates the full chain:
    policy action index (SDK)  ->  SDK_TO_MJ  ->  MuJoCo ctrl  ->  PD torque  ->  joint response

No ONNX, no observations, no DDS, no real robot. For each SDK index i and each
amplitude a, this script:
  1. Resets the model to the default_stand keyframe.
  2. Sets action[i] = a (all other action entries = 0).
  3. Runs the same PD law that will be used in the closed-loop runner.
  4. Records direction / magnitude / torque saturation / stability per sweep.

Output: a CSV per-sweep summary and a printed pass/fail table.
"""

import argparse
import csv
import os
import sys

import mujoco
import numpy as np


# -----------------------------------------------------------------------------
# Constants (mirrors g0.py / g0_actuators.py / verified by g0_contract_check.py)
# -----------------------------------------------------------------------------

# Isaac Lab action order (= policy.onnx output order).
G0_JOINT_SDK_NAMES = [
    "l_hip_pitch_joint", "l_hip_roll_joint", "l_hip_yaw_joint",
    "l_knee_pitch_joint", "l_ankle_pitch_joint", "l_ankle_roll_joint",
    "r_hip_pitch_joint", "r_hip_roll_joint", "r_hip_yaw_joint",
    "r_knee_pitch_joint", "r_ankle_pitch_joint", "r_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint",
    "l_shoulder_pitch_joint", "l_shoulder_roll_joint",
    "l_shoulder_yaw_joint", "l_elbow_pitch_joint",
    "r_shoulder_pitch_joint", "r_shoulder_roll_joint",
    "r_shoulder_yaw_joint", "r_elbow_pitch_joint",
]

# MuJoCo / URDF traversal order. The patched MJCF lists actuators in this order.
G0_JOINT_NAMES_MJ = [
    "waist_yaw_joint", "waist_roll_joint",
    "l_shoulder_pitch_joint", "l_shoulder_roll_joint",
    "l_shoulder_yaw_joint", "l_elbow_pitch_joint",
    "r_shoulder_pitch_joint", "r_shoulder_roll_joint",
    "r_shoulder_yaw_joint", "r_elbow_pitch_joint",
    "l_hip_pitch_joint", "l_hip_roll_joint", "l_hip_yaw_joint",
    "l_knee_pitch_joint", "l_ankle_pitch_joint", "l_ankle_roll_joint",
    "r_hip_pitch_joint", "r_hip_roll_joint", "r_hip_yaw_joint",
    "r_knee_pitch_joint", "r_ankle_pitch_joint", "r_ankle_roll_joint",
]

# Default standing pose, SDK order. Mirrors G0_DEFAULT_JOINT_POS in g0.py.
G0_DEFAULT_Q_SDK = np.array([
    -0.20, 0.0, 0.0, 0.34, -0.14, 0.0,   # left leg
    -0.20, 0.0, 0.0, 0.34, -0.14, 0.0,   # right leg
     0.0, 0.0,                            # waist
     0.30, 0.25, 0.0, -0.97,              # left arm
     0.30, -0.25, 0.0, -0.97,             # right arm
], dtype=np.float64)

# Frozen permutation (verified by g0_contract_check.py).
SDK_TO_MJ = np.array(
    [10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21,
      0,  1,
      2,  3,  4,  5,
      6,  7,  8,  9],
    dtype=np.int64,
)

# JointPositionActionCfg.scale, from velocity_env_cfg.py.
ACTION_SCALE = 0.12

# Per-joint PD gains, same as g0.py / g0_mujoco_zero_action.py.
KP_SDK = {
    "l_hip_pitch_joint": 4.0, "l_hip_roll_joint": 4.0, "l_hip_yaw_joint": 3.0,
    "l_knee_pitch_joint": 4.0, "l_ankle_pitch_joint": 4.0, "l_ankle_roll_joint": 4.5,
    "r_hip_pitch_joint": 4.0, "r_hip_roll_joint": 4.0, "r_hip_yaw_joint": 3.0,
    "r_knee_pitch_joint": 4.0, "r_ankle_pitch_joint": 4.0, "r_ankle_roll_joint": 4.5,
    "waist_yaw_joint": 2.0, "waist_roll_joint": 2.0,
    "l_shoulder_pitch_joint": 1.5, "l_shoulder_roll_joint": 1.5,
    "l_shoulder_yaw_joint": 1.5, "l_elbow_pitch_joint": 2.0,
    "r_shoulder_pitch_joint": 1.5, "r_shoulder_roll_joint": 1.5,
    "r_shoulder_yaw_joint": 1.5, "r_elbow_pitch_joint": 2.0,
}
KD_SDK = {
    "l_hip_pitch_joint": 0.18, "l_hip_roll_joint": 0.16, "l_hip_yaw_joint": 0.10,
    "l_knee_pitch_joint": 0.26, "l_ankle_pitch_joint": 0.22, "l_ankle_roll_joint": 0.15,
    "r_hip_pitch_joint": 0.18, "r_hip_roll_joint": 0.16, "r_hip_yaw_joint": 0.10,
    "r_knee_pitch_joint": 0.26, "r_ankle_pitch_joint": 0.22, "r_ankle_roll_joint": 0.15,
    "waist_yaw_joint": 0.08, "waist_roll_joint": 0.08,
    "l_shoulder_pitch_joint": 0.06, "l_shoulder_roll_joint": 0.06,
    "l_shoulder_yaw_joint": 0.06, "l_elbow_pitch_joint": 0.08,
    "r_shoulder_pitch_joint": 0.06, "r_shoulder_roll_joint": 0.06,
    "r_shoulder_yaw_joint": 0.06, "r_elbow_pitch_joint": 0.08,
}


# -----------------------------------------------------------------------------
# Setup helpers
# -----------------------------------------------------------------------------

def build_joint_address_table(model: mujoco.MjModel):
    """For each MuJoCo actuator (mj index 0..nu-1), look up the qpos / qvel
    address of the joint it drives. Avoids the qpos[7:] / qvel[6:] shortcut so
    additional unactuated joints in the future would not silently mis-align.
    """
    qadr = np.empty(model.nu, dtype=np.int64)
    dadr = np.empty(model.nu, dtype=np.int64)
    joint_name = []
    for mj_i in range(model.nu):
        jid = int(model.actuator_trnid[mj_i, 0])
        qadr[mj_i] = int(model.jnt_qposadr[jid])
        dadr[mj_i] = int(model.jnt_dofadr[jid])
        joint_name.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid))
    return qadr, dadr, joint_name


def remap_sdk_to_mj(vec_sdk: np.ndarray) -> np.ndarray:
    vec_mj = np.empty_like(vec_sdk)
    vec_mj[SDK_TO_MJ] = vec_sdk
    return vec_mj


# -----------------------------------------------------------------------------
# Single sweep
# -----------------------------------------------------------------------------

def run_one_sweep(
    model,
    data,
    sdk_i: int,
    amplitude: float,
    default_q_mj: np.ndarray,
    kp_mj: np.ndarray,
    kd_mj: np.ndarray,
    ctrl_lo: np.ndarray,
    ctrl_hi: np.ndarray,
    qadr: np.ndarray,
    dadr: np.ndarray,
    duration: float,
    keyframe_id: int,
):
    """Hold action[sdk_i] = amplitude for `duration` seconds. Returns metrics."""
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)

    # Compute target_q in MJ order (only one element differs from default).
    target_q_sdk = G0_DEFAULT_Q_SDK.copy()
    target_q_sdk[sdk_i] += ACTION_SCALE * amplitude
    target_q_mj = remap_sdk_to_mj(target_q_sdk)

    mj_i = int(SDK_TO_MJ[sdk_i])
    target_delta = ACTION_SCALE * amplitude  # expected change at the swept joint

    n_steps = int(round(duration / model.opt.timestep))
    sat_thresh = 0.999

    max_abs_tau_swept = 0.0
    max_abs_tau_any = 0.0
    sat_count = 0
    diverged = False
    nan_seen = False

    for _ in range(n_steps):
        q_mj = data.qpos[qadr]
        dq_mj = data.qvel[dadr]

        if not np.all(np.isfinite(q_mj)) or not np.all(np.isfinite(dq_mj)):
            nan_seen = True
            break

        tau_unclipped = kp_mj * (target_q_mj - q_mj) - kd_mj * dq_mj
        tau = np.clip(tau_unclipped, ctrl_lo, ctrl_hi)

        a_swept = abs(tau[mj_i])
        if a_swept > max_abs_tau_swept:
            max_abs_tau_swept = a_swept
        a_any = float(np.max(np.abs(tau)))
        if a_any > max_abs_tau_any:
            max_abs_tau_any = a_any

        # Saturation accounting on the swept channel only.
        ctrl_max_i = max(abs(ctrl_lo[mj_i]), abs(ctrl_hi[mj_i]))
        if abs(tau_unclipped[mj_i]) >= sat_thresh * ctrl_max_i:
            sat_count += 1

        # Stability sanity: |q| > 20 rad means the model fell or exploded.
        if np.max(np.abs(q_mj)) > 20.0:
            diverged = True
            break

        data.ctrl[:] = tau
        mujoco.mj_step(model, data)

    q_final_mj = data.qpos[qadr].copy()
    measured_delta = float(q_final_mj[mj_i] - default_q_mj[mj_i])

    direction_ok = (np.sign(measured_delta) == np.sign(target_delta)) if target_delta != 0 else (abs(measured_delta) < 1e-3)
    stable = (not diverged) and (not nan_seen) and np.all(np.isfinite(q_final_mj))

    return {
        "sdk_index": sdk_i,
        "sdk_joint_name": G0_JOINT_SDK_NAMES[sdk_i],
        "mj_actuator_index": mj_i,
        "mj_joint_name": G0_JOINT_NAMES_MJ[mj_i],
        "action_amp": amplitude,
        "target_delta": target_delta,
        "measured_delta": measured_delta,
        "direction_ok": bool(direction_ok),
        "max_abs_tau_swept": max_abs_tau_swept,
        "max_abs_tau_any": max_abs_tau_any,
        "torque_saturated_ratio": (sat_count / n_steps) if n_steps else 0.0,
        "stable": bool(stable),
        "diverged": bool(diverged),
        "nan_seen": bool(nan_seen),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--duration", type=float, default=1.5,
                    help="Hold each amplitude for this many sim seconds.")
    ap.add_argument("--amplitudes", default="0.25,-0.25,0.5,-0.5",
                    help="Comma-separated list of action amplitudes to sweep per joint.")
    ap.add_argument("--output", default="logs/sim2sim/g0_action_sweep_summary.csv")
    ap.add_argument("--keyframe", default="default_stand")
    args = ap.parse_args()

    amplitudes = [float(x) for x in args.amplitudes.split(",") if x.strip()]

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    if model.nu != 22:
        raise RuntimeError(f"Expected nu=22, got {model.nu}")

    # Verify actuator order matches our hardcoded G0_JOINT_NAMES_MJ.
    qadr, dadr, joint_per_actuator = build_joint_address_table(model)
    for mj_i, jn in enumerate(joint_per_actuator):
        if jn != G0_JOINT_NAMES_MJ[mj_i]:
            raise RuntimeError(
                f"Actuator order drift at mj_i={mj_i}: model says '{jn}', "
                f"hardcoded says '{G0_JOINT_NAMES_MJ[mj_i]}'"
            )
    # And cross-verify SDK_TO_MJ against names.
    for sdk_i, sdk_n in enumerate(G0_JOINT_SDK_NAMES):
        mj_i = int(SDK_TO_MJ[sdk_i])
        if joint_per_actuator[mj_i] != sdk_n:
            raise RuntimeError(
                f"SDK_TO_MJ inconsistent at sdk_i={sdk_i} ({sdk_n}): "
                f"resolves to mj joint '{joint_per_actuator[mj_i]}'"
            )

    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
    if keyframe_id < 0:
        raise RuntimeError(f"Missing keyframe: {args.keyframe}")

    # Build mj-ordered defaults / gains / ctrlrange once.
    default_q_mj = remap_sdk_to_mj(G0_DEFAULT_Q_SDK)
    kp_sdk = np.array([KP_SDK[n] for n in G0_JOINT_SDK_NAMES], dtype=np.float64)
    kd_sdk = np.array([KD_SDK[n] for n in G0_JOINT_SDK_NAMES], dtype=np.float64)
    kp_mj = remap_sdk_to_mj(kp_sdk)
    kd_mj = remap_sdk_to_mj(kd_sdk)
    ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
    ctrl_hi = model.actuator_ctrlrange[:, 1].copy()

    print("=" * 100)
    print("G0 action sweep")
    print("=" * 100)
    print(f"  model      : {args.model}")
    print(f"  keyframe   : {args.keyframe}")
    print(f"  amplitudes : {amplitudes}")
    print(f"  duration   : {args.duration}s per amplitude")
    print(f"  scale      : {ACTION_SCALE}")
    print(f"  output csv : {args.output}")
    print()

    # Run sweeps.
    rows = []
    n_total = len(G0_JOINT_SDK_NAMES) * len(amplitudes)
    n_done = 0
    for sdk_i in range(len(G0_JOINT_SDK_NAMES)):
        for amp in amplitudes:
            n_done += 1
            r = run_one_sweep(
                model, data,
                sdk_i, amp,
                default_q_mj, kp_mj, kd_mj,
                ctrl_lo, ctrl_hi,
                qadr, dadr,
                args.duration, keyframe_id,
            )
            rows.append(r)
            sys.stdout.write(
                f"\r[{n_done:3d}/{n_total}] sdk={sdk_i:2d} {r['sdk_joint_name']:30s} "
                f"amp={amp:+0.2f} dq={r['measured_delta']:+0.4f} "
                f"dir={'OK' if r['direction_ok'] else 'NO'} "
                f"sat={r['torque_saturated_ratio']*100:5.1f}%   "
            )
            sys.stdout.flush()
    sys.stdout.write("\n\n")

    # Summary table.
    hdr = (f"{'sdk':>3} {'sdk_joint':30s} {'mj':>3} {'mj_joint':30s} "
           f"{'amp':>6} {'tgt_dq':>8} {'mes_dq':>9} {'dir':>4} "
           f"{'max|tau|':>9} {'sat%':>6} {'stable':>6}")
    print(hdr)
    print("-" * len(hdr))
    n_pass = 0
    n_dir_fail = 0
    n_unstable = 0
    for r in rows:
        ok = r["direction_ok"] and r["stable"]
        n_pass += int(ok)
        n_dir_fail += int(not r["direction_ok"])
        n_unstable += int(not r["stable"])
        flag_dir = "OK" if r["direction_ok"] else "FAIL"
        flag_stab = "OK" if r["stable"] else "BAD"
        print(
            f"{r['sdk_index']:>3} {r['sdk_joint_name']:30s} "
            f"{r['mj_actuator_index']:>3} {r['mj_joint_name']:30s} "
            f"{r['action_amp']:>+6.2f} {r['target_delta']:>+8.4f} "
            f"{r['measured_delta']:>+9.4f} {flag_dir:>4} "
            f"{r['max_abs_tau_swept']:>9.4f} "
            f"{r['torque_saturated_ratio']*100:>5.1f}% "
            f"{flag_stab:>6}"
        )

    print()
    print("=" * 100)
    print(f"Total sweeps          : {len(rows)}")
    print(f"Direction OK          : {len(rows) - n_dir_fail}/{len(rows)}")
    print(f"Stable (no nan/blow)  : {len(rows) - n_unstable}/{len(rows)}")
    print(f"Overall pass (dir+stab): {n_pass}/{len(rows)}")
    print("=" * 100)

    # CSV.
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fieldnames = [
        "sdk_index", "sdk_joint_name", "mj_actuator_index", "mj_joint_name",
        "action_amp", "target_delta", "measured_delta", "direction_ok",
        "max_abs_tau_swept", "max_abs_tau_any", "torque_saturated_ratio",
        "stable", "diverged", "nan_seen",
    ]
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
