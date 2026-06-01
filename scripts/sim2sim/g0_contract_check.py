"""Contract check for G0 sim2sim mapping.

Verifies:
  1. MuJoCo actuator type -> data.ctrl semantics (motor=torque vs position=q_target).
  2. SDK_TO_MJ and its inverse MJ_TO_SDK are consistent permutations.
  3. Perturbing SDK action[i] by +1 only touches mj_ctrl[SDK_TO_MJ[i]],
     and that mj actuator drives the joint whose name == G0_JOINT_SDK_NAMES[i].

No ONNX, no closed loop. Static checks only.
"""

import argparse
import numpy as np
import mujoco


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

# Default standing pose, SDK order. Mirrors G0_DEFAULT_JOINT_POS in g0.py.
G0_DEFAULT_Q_SDK = np.array([
    -0.20, 0.0, 0.0, 0.34, -0.14, 0.0,   # left leg
    -0.20, 0.0, 0.0, 0.34, -0.14, 0.0,   # right leg
     0.0, 0.0,                            # waist
     0.30, 0.25, 0.0, -0.97,              # left arm
     0.30, -0.25, 0.0, -0.97,             # right arm
], dtype=np.float64)

# Frozen mapping from earlier comparison step.
SDK_TO_MJ = np.array(
    [10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21,
      0,  1,
      2,  3,  4,  5,
      6,  7,  8,  9],
    dtype=np.int64,
)

ACTION_SCALE = 0.12  # JointPositionActionCfg.scale


def inverse_permutation(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


ACTUATOR_TYPE_NAMES = {
    int(mujoco.mjtTrn.mjTRN_JOINT): "joint",
    int(mujoco.mjtTrn.mjTRN_TENDON): "tendon",
    int(mujoco.mjtTrn.mjTRN_SITE): "site",
}

# Heuristic: classify actuator as motor / position / velocity based on gain/bias.
# - motor:    gainprm = [1,0,0], biasprm = [0,0,0]
# - position: gainprm = [kp,0,0], biasprm = [0,-kp,0]
# - velocity: gainprm = [kv,0,0], biasprm = [0,0,-kv]
def classify_actuator(model: mujoco.MjModel, i: int) -> str:
    g = model.actuator_gainprm[i]
    b = model.actuator_biasprm[i]
    kp_pos = g[0] != 0 and b[1] == -g[0] and b[0] == 0 and b[2] == 0
    kv_vel = g[0] != 0 and b[2] == -g[0] and b[0] == 0 and b[1] == 0
    pure_motor = g[0] == 1 and (b[:3] == 0).all()
    if pure_motor:
        return "motor (torque)"
    if kp_pos:
        return f"position (kp={g[0]:g})"
    if kv_vel:
        return f"velocity (kv={g[0]:g})"
    return f"general (gainprm={g[:3]}, biasprm={b[:3]})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)

    if model.nu != 22:
        raise RuntimeError(f"Expected nu=22, got {model.nu}")

    # --------------------------------------------------------------------- 1
    print("=" * 78)
    print("[1] MuJoCo actuator semantics")
    print("=" * 78)
    types_seen = set()
    print(f"{'idx':>3}  {'name':30s} {'type':32s} ctrlrange")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        kind = classify_actuator(model, i)
        types_seen.add(kind)
        lo, hi = model.actuator_ctrlrange[i]
        print(f"{i:>3}  {name:30s} {kind:32s} [{lo:+.4f}, {hi:+.4f}]")

    print()
    if all(t.startswith("motor") for t in types_seen):
        print(">>> All actuators are <motor>. data.ctrl is TORQUE (Nm).")
        print(">>> Closed loop MUST apply PD in Python:")
        print(">>>     tau = kp * (q_target - q) + kd * (0 - dq)")
        ctrl_is_torque = True
    elif all(t.startswith("position") for t in types_seen):
        print(">>> All actuators are <position>. data.ctrl is q_target (rad).")
        print(">>> No external PD needed; MuJoCo applies internal kp/kv.")
        ctrl_is_torque = False
    else:
        print(f">>> Mixed actuator types: {types_seen}. Handle each group separately.")
        ctrl_is_torque = None

    # --------------------------------------------------------------------- 2
    print()
    print("=" * 78)
    print("[2] Mapping consistency")
    print("=" * 78)
    MJ_TO_SDK = inverse_permutation(SDK_TO_MJ)
    print(f"SDK_TO_MJ = {SDK_TO_MJ.tolist()}")
    print(f"MJ_TO_SDK = {MJ_TO_SDK.tolist()}")

    # Round-trip identity check.
    rt1 = SDK_TO_MJ[MJ_TO_SDK]
    rt2 = MJ_TO_SDK[SDK_TO_MJ]
    assert np.array_equal(rt1, np.arange(22)), f"SDK_TO_MJ ∘ MJ_TO_SDK != identity: {rt1}"
    assert np.array_equal(rt2, np.arange(22)), f"MJ_TO_SDK ∘ SDK_TO_MJ != identity: {rt2}"
    assert sorted(SDK_TO_MJ.tolist()) == list(range(22)), "SDK_TO_MJ is not a permutation"
    print("Permutation round-trip OK.")

    # Cross-check against the model:
    # for each sdk_index i, the actuator at MJ index SDK_TO_MJ[i] must drive
    # the joint whose name equals G0_JOINT_SDK_NAMES[i].
    print()
    print("Joint-name consistency check (using model.actuator_trnid):")
    bad = 0
    for sdk_i in range(22):
        mj_i = int(SDK_TO_MJ[sdk_i])
        jid = int(model.actuator_trnid[mj_i, 0])
        mj_joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        sdk_joint = G0_JOINT_SDK_NAMES[sdk_i]
        ok = mj_joint == sdk_joint
        if not ok:
            bad += 1
            print(f"  MISMATCH  sdk[{sdk_i}]={sdk_joint}  vs  mj[{mj_i}]={mj_joint}")
    if bad == 0:
        print("All 22 SDK indices resolve to the correct MuJoCo joint via SDK_TO_MJ.")
    else:
        raise RuntimeError(f"{bad} mismatches in SDK_TO_MJ")

    # --------------------------------------------------------------------- 3
    print()
    print("=" * 78)
    print("[3] One-hot SDK action perturbation -> single MJ ctrl change")
    print("=" * 78)
    print("Formula: target_q_sdk[i] = action[i] * scale + default_q_sdk[i]")
    print(f"scale = {ACTION_SCALE}")
    print()

    # Baseline: zero action -> default pose, remapped to mj order.
    default_q_mj = np.zeros(22, dtype=np.float64)
    default_q_mj[SDK_TO_MJ] = G0_DEFAULT_Q_SDK

    header = (
        f"{'sdk_i':>5}  {'sdk_joint':30s} "
        f"{'mj_i':>4}  {'mj_joint':30s} "
        f"{'default':>9} {'target':>9} {'delta':>8}  match"
    )
    print(header)
    print("-" * len(header))

    all_ok = True
    for sdk_i in range(22):
        action = np.zeros(22, dtype=np.float64)
        action[sdk_i] = 1.0

        target_q_sdk = action * ACTION_SCALE + G0_DEFAULT_Q_SDK
        target_q_mj = np.zeros(22, dtype=np.float64)
        target_q_mj[SDK_TO_MJ] = target_q_sdk

        delta = target_q_mj - default_q_mj
        changed = np.where(np.abs(delta) > 1e-12)[0]

        # The single mj index that should change.
        expected_mj_i = int(SDK_TO_MJ[sdk_i])

        ok = (len(changed) == 1) and (int(changed[0]) == expected_mj_i)
        all_ok &= ok

        jid = int(model.actuator_trnid[expected_mj_i, 0])
        mj_joint = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)

        print(
            f"{sdk_i:>5}  {G0_JOINT_SDK_NAMES[sdk_i]:30s} "
            f"{expected_mj_i:>4}  {mj_joint:30s} "
            f"{default_q_mj[expected_mj_i]:>+9.4f} "
            f"{target_q_mj[expected_mj_i]:>+9.4f} "
            f"{delta[expected_mj_i]:>+8.4f}  {'OK' if ok else 'FAIL'}"
        )

    print()
    print("Per-index isolation:", "OK" if all_ok else "FAIL")
    print()
    print("=" * 78)
    print("Contract summary")
    print("=" * 78)
    print(f"  ctrl is torque (need external PD)? : {ctrl_is_torque}")
    print(f"  SDK_TO_MJ / MJ_TO_SDK valid?       : True")
    print(f"  one-hot isolation passes?          : {all_ok}")


if __name__ == "__main__":
    main()
