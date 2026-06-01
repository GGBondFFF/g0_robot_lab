"""ONNX policy MuJoCo closed-loop runner with GUI viewer.

Closed-loop chain (per policy step):

    MuJoCo state
      -> single-step 77-D obs (per-term, with the correct scales)
      -> 5-step per-term-grouped, oldest-first 385-D history obs
      -> policy.onnx -> action_sdk (22,)
      -> target_q_sdk = default_q_sdk + ACTION_SCALE * action_sdk
      -> target_q_mj  = remap(SDK_TO_MJ, target_q_sdk)
      -> tau_mj = kp_mj * (target_q_mj - q_mj) + kd_mj * (0 - dq_mj)
      -> tau_mj = clip(tau_mj, ctrl_lo, ctrl_hi)
      -> data.ctrl[:] = tau_mj
      -> mujoco.mj_step()  x decimation

Joint orders:
    SDK   : policy action / last_action  (G0_JOINT_SDK_NAMES)
    MJ    : actuator / ctrl              (G0_JOINT_NAMES_MJ)
    Isaac : obs joint_pos_rel/vel_rel    (G0_JOINT_NAMES_ISAAC)

No DDS. No LowCmd. No real robot. data.ctrl is always torque (MJCF actuators
are <motor>).
"""

import argparse
import csv
import math
import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort


# -----------------------------------------------------------------------------
# Joint orders
# -----------------------------------------------------------------------------

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

# IsaacLab articulation order, taken from g0_obs_isaac.npz dump.
G0_JOINT_NAMES_ISAAC = [
    "l_hip_pitch_joint", "r_hip_pitch_joint", "waist_yaw_joint",
    "l_hip_roll_joint", "r_hip_roll_joint", "waist_roll_joint",
    "l_hip_yaw_joint", "r_hip_yaw_joint",
    "l_shoulder_pitch_joint", "r_shoulder_pitch_joint",
    "l_knee_pitch_joint", "r_knee_pitch_joint",
    "l_shoulder_roll_joint", "r_shoulder_roll_joint",
    "l_ankle_pitch_joint", "r_ankle_pitch_joint",
    "l_shoulder_yaw_joint", "r_shoulder_yaw_joint",
    "l_ankle_roll_joint", "r_ankle_roll_joint",
    "l_elbow_pitch_joint", "r_elbow_pitch_joint",
]

# Default standing pose in SDK order (mirrors g0.py).
G0_DEFAULT_Q_SDK = np.array([
    -0.20, 0.0, 0.0, 0.34, -0.14, 0.0,
    -0.20, 0.0, 0.0, 0.34, -0.14, 0.0,
     0.0, 0.0,
     0.30, 0.25, 0.0, -0.97,
     0.30, -0.25, 0.0, -0.97,
], dtype=np.float64)

SDK_TO_MJ = np.array(
    [10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21,
      0,  1,
      2,  3,  4,  5,
      6,  7,  8,  9],
    dtype=np.int64,
)

ACTION_SCALE = 0.12

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

# Per-term obs scales from velocity_env_cfg.py.
SCALE_BASE_ANG_VEL = 0.2
SCALE_PROJ_GRAV = 1.0
SCALE_VEL_CMD = 1.0
SCALE_JPOS_REL = 1.0
SCALE_JVEL_REL = 0.05
SCALE_LAST_ACT = 1.0
SCALE_GAIT = 1.0

# 5-step history per-term layout.
TERM_DIMS = [3, 3, 3, 22, 22, 22, 2]   # sum = 77
HISTORY_LEN = 5
OBS_DIM_PER_STEP = 77
OBS_DIM_HISTORY = OBS_DIM_PER_STEP * HISTORY_LEN  # 385

# gait period (s) and policy dt come from velocity_env_cfg.py.
GAIT_PERIOD = 0.8
POLICY_DT = 0.02          # Isaac decimation 4 * sim.dt 0.005


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def build_joint_address_table(model):
    qadr = np.empty(model.nu, dtype=np.int64)
    dadr = np.empty(model.nu, dtype=np.int64)
    joint_name = []
    for mj_i in range(model.nu):
        jid = int(model.actuator_trnid[mj_i, 0])
        qadr[mj_i] = int(model.jnt_qposadr[jid])
        dadr[mj_i] = int(model.jnt_dofadr[jid])
        joint_name.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid))
    return qadr, dadr, joint_name


def remap_sdk_to_mj(vec_sdk):
    vec_mj = np.empty_like(vec_sdk)
    vec_mj[SDK_TO_MJ] = vec_sdk
    return vec_mj


def quat_apply_inverse_wxyz(q_wxyz, v):
    w, x, y, z = q_wxyz
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R.T @ np.asarray(v, dtype=np.float64)


def quat_to_rpy_wxyz(q):
    w, x, y, z = q
    # roll (x), pitch (y), yaw (z) -- standard intrinsic XYZ.
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)
    return roll, pitch, yaw


# -----------------------------------------------------------------------------
# Obs construction
# -----------------------------------------------------------------------------

def build_obs_77(
    model, data,
    qadr, dadr,
    isaac_to_mj,
    default_q_isaac, default_dq_isaac,
    velocity_cmd, last_action_sdk,
    gait_phase_ratio,
    base_body_id,
):
    # base_ang_vel in body frame
    vel6 = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_body_id, vel6, 1)
    base_ang_vel_b = vel6[0:3].copy()

    # projected gravity in body frame
    root_quat = data.qpos[3:7].copy()
    proj_grav_b = quat_apply_inverse_wxyz(root_quat, np.array([0.0, 0.0, -1.0]))

    # joints in Isaac order
    q_isaac = np.empty(22, dtype=np.float64)
    dq_isaac = np.empty(22, dtype=np.float64)
    for isaac_i, mj_i in enumerate(isaac_to_mj):
        q_isaac[isaac_i] = data.qpos[qadr[mj_i]]
        dq_isaac[isaac_i] = data.qvel[dadr[mj_i]]
    jpos_rel = q_isaac - default_q_isaac
    jvel_rel = dq_isaac - default_dq_isaac

    gait = np.array([
        math.sin(gait_phase_ratio * 2.0 * math.pi),
        math.cos(gait_phase_ratio * 2.0 * math.pi),
    ], dtype=np.float64)

    obs = np.zeros(77, dtype=np.float64)
    obs[0:3]   = base_ang_vel_b * SCALE_BASE_ANG_VEL
    obs[3:6]   = proj_grav_b    * SCALE_PROJ_GRAV
    obs[6:9]   = velocity_cmd   * SCALE_VEL_CMD
    obs[9:31]  = jpos_rel       * SCALE_JPOS_REL
    obs[31:53] = jvel_rel       * SCALE_JVEL_REL
    obs[53:75] = last_action_sdk * SCALE_LAST_ACT
    obs[75:77] = gait           * SCALE_GAIT
    return obs


def build_obs_385(history_77):
    """history_77 is (HISTORY_LEN, 77), oldest first. Returns (385,) per-term grouped."""
    out = []
    offset = 0
    for d in TERM_DIMS:
        out.append(history_77[:, offset:offset + d].reshape(-1))
        offset += d
    return np.concatenate(out, axis=0)


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--policy", required=True, help="path to policy.onnx")
    ap.add_argument("--mode", choices=("zero_cmd", "cmd"), default="zero_cmd")
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--keyframe", default="default_stand")
    ap.add_argument("--csv", default="logs/sim2sim/g0_onnx_gui_closed_loop_log.csv")
    ap.add_argument("--no-viewer", action="store_true",
                    help="Headless run (no GUI). Useful for CI / quick checks.")
    ap.add_argument("--print-every", type=float, default=0.5,
                    help="Print status every N seconds of sim time.")
    ap.add_argument("--realtime", action="store_true",
                    help="Throttle GUI to wall-clock real time.")
    ap.add_argument("--kp-scale", type=float, default=1.0,
                    help="Multiply all KP gains. Sim2sim gap workaround for implicit vs explicit PD.")
    ap.add_argument("--kd-scale", type=float, default=1.0,
                    help="Multiply all KD gains.")
    ap.add_argument("--joint-damping", type=float, default=None,
                    help="Override MJCF joint damping (per-DOF) for the 22 actuated joints. "
                         "Follows unitree_mujoco convention (G1 uses 0.05). Root freejoint untouched.")
    ap.add_argument("--timestep", type=float, default=None,
                    help="Override MJCF sim timestep. Use to tighten explicit-PD stability margin.")
    ap.add_argument("--joint-frictionloss", type=float, default=None,
                    help="Override MJCF joint frictionloss for the 22 actuated joints. "
                         "Follows unitree_mujoco convention (G1 uses 0.1~0.2). Root freejoint untouched.")
    # Virtual elastic band — ports unitree_mujoco/simulate/src/main.cc:54-86 (ElasticBand class).
    # Per Unitree readme_zh.md:240: "考虑到人形机器人不便于从平地上启动并进行调试,
    # 在仿真中设计了一个虚拟挂带,用于模拟人形机器人的吊起和放下".
    ap.add_argument("--elastic-band", action="store_true",
                    help="Enable virtual elastic-band suspension on base_link (Unitree pattern).")
    ap.add_argument("--band-anchor", default="0 0 2.0",
                    help="World-frame anchor point 'x y z' for the band (m). Unitree default '0 0 3'.")
    ap.add_argument("--band-stiffness", type=float, default=10.0,
                    help="Band spring stiffness (N/m). Unitree G1 uses 200; G0 (1.4 kg) is scaled down ~20x.")
    ap.add_argument("--band-damping", type=float, default=2.0,
                    help="Band spring damping (N*s/m). Unitree G1 uses 100; G0 scaled ~50x.")
    ap.add_argument("--band-length", type=float, default=0.0,
                    help="Band rest length (m). Default 0 matches Unitree convention.")
    ap.add_argument("--no-abort", action="store_true",
                    help="Disable early abort on root_z/rpy/NaN — keep stepping so the GUI run plays out.")
    args = ap.parse_args()

    # ---- model
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    if model.nu != 22:
        raise RuntimeError(f"Expected nu=22, got {model.nu}")

    qadr, dadr, joint_per_actuator = build_joint_address_table(model)
    for mj_i, jn in enumerate(joint_per_actuator):
        if jn != G0_JOINT_NAMES_MJ[mj_i]:
            raise RuntimeError(f"Actuator order drift at mj_i={mj_i}: {jn} vs {G0_JOINT_NAMES_MJ[mj_i]}")

    # ISAAC_TO_MJ: for each Isaac index, the mj actuator index of the same joint.
    mj_index_by_name = {n: i for i, n in enumerate(joint_per_actuator)}
    isaac_to_mj = np.array(
        [mj_index_by_name[n] for n in G0_JOINT_NAMES_ISAAC],
        dtype=np.int64,
    )

    if args.timestep is not None:
        model.opt.timestep = float(args.timestep)

    # Optional MJCF joint property overrides (Unitree-style: small implicit
    # damping + frictionloss on actuated joints for sim2sim stability).
    if args.joint_damping is not None:
        for mj_i in range(model.nu):
            model.dof_damping[int(dadr[mj_i])] = float(args.joint_damping)
    if args.joint_frictionloss is not None:
        for mj_i in range(model.nu):
            model.dof_frictionloss[int(dadr[mj_i])] = float(args.joint_frictionloss)

    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
    if keyframe_id < 0:
        raise RuntimeError(f"Missing keyframe: {args.keyframe}")
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_body_id < 0:
        raise RuntimeError("Missing body 'base_link' in model")

    # ---- defaults & gains
    default_q_mj = remap_sdk_to_mj(G0_DEFAULT_Q_SDK)
    default_q_isaac = np.empty(22, dtype=np.float64)
    for isaac_i, mj_i in enumerate(isaac_to_mj):
        default_q_isaac[isaac_i] = default_q_mj[mj_i]
    default_dq_isaac = np.zeros(22, dtype=np.float64)

    kp_sdk = np.array([KP_SDK[n] for n in G0_JOINT_SDK_NAMES], dtype=np.float64)
    kd_sdk = np.array([KD_SDK[n] for n in G0_JOINT_SDK_NAMES], dtype=np.float64)
    kp_mj = remap_sdk_to_mj(kp_sdk) * args.kp_scale
    kd_mj = remap_sdk_to_mj(kd_sdk) * args.kd_scale
    ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
    ctrl_hi = model.actuator_ctrlrange[:, 1].copy()
    ctrl_max_abs = np.maximum(np.abs(ctrl_lo), np.abs(ctrl_hi))

    # ---- ONNX
    sess = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    inp_shape = sess.get_inputs()[0].shape
    out_name = sess.get_outputs()[0].name
    out_shape = sess.get_outputs()[0].shape
    expected_in = [s for s in inp_shape if isinstance(s, int)]
    if 385 not in expected_in:
        print(f"WARNING: ONNX input shape {inp_shape} does not contain 385.")
    print("=" * 90)
    print("ONNX MuJoCo closed-loop runner (GUI)")
    print("=" * 90)
    print(f"  model    : {args.model}")
    print(f"  policy   : {args.policy}  in={inp_shape} out={out_shape}")
    print(f"  mode     : {args.mode}   cmd=({args.vx},{args.vy},{args.wz})")
    print(f"  duration : {args.duration}s    keyframe : {args.keyframe}")
    print(f"  mj dt    : {model.opt.timestep}   policy dt : {POLICY_DT}")
    print(f"  kp_scale : {args.kp_scale}   kd_scale : {args.kd_scale}")
    print(f"  joint_damping override   : {args.joint_damping}")
    print(f"  joint_frictionloss over. : {args.joint_frictionloss}")
    print()

    # ---- command
    if args.mode == "zero_cmd":
        velocity_cmd = np.zeros(3, dtype=np.float64)
    else:
        velocity_cmd = np.array([args.vx, args.vy, args.wz], dtype=np.float64)

    # ---- reset
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)

    # Decimation: how many mj steps per policy step.
    decimation = max(1, int(round(POLICY_DT / model.opt.timestep)))
    n_policy_steps = int(round(args.duration / POLICY_DT))

    # ---- elastic band setup (mirrors unitree_mujoco/simulate/src/main.cc:54-86)
    band_anchor = np.array([float(x) for x in args.band_anchor.split()], dtype=np.float64)
    assert band_anchor.shape == (3,), "--band-anchor must be 'x y z'"
    band_k = float(args.band_stiffness)
    band_c = float(args.band_damping)
    band_L = float(args.band_length)
    if args.elastic_band:
        x0 = data.qpos[0:3].copy()
        d0 = float(np.linalg.norm(band_anchor - x0))
        f0 = band_k * (d0 - band_L)
        weight = float(np.sum(model.body_mass)) * 9.81
        print(f"  ELASTIC BAND on   : anchor={band_anchor}  k={band_k} N/m  c={band_c} N*s/m  L={band_L} m")
        print(f"    initial dist    : {d0:.3f} m   initial vertical force ≈ {f0:.2f} N  (robot weight={weight:.2f} N)")
        if f0 < 0.5 * weight:
            print(f"    WARNING: band force ({f0:.1f}N) < 50% of weight ({weight:.1f}N) — too weak to suspend.")
        if f0 > 2.0 * weight:
            print(f"    WARNING: band force ({f0:.1f}N) > 2x weight ({weight:.1f}N) — robot may launch upward.")

    # ---- history buffer (oldest first)
    history_77 = np.zeros((HISTORY_LEN, OBS_DIM_PER_STEP), dtype=np.float64)
    last_action_sdk = np.zeros(22, dtype=np.float64)
    target_q_mj_prev = default_q_mj.copy()

    # ---- logging
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    csv_f = open(args.csv, "w", newline="")
    csv_w = csv.writer(csv_f)
    csv_w.writerow([
        "time", "vx", "vy", "wz",
        "root_x", "root_y", "root_z", "roll", "pitch", "yaw",
        "max_abs_action", "mean_abs_action", "max_abs_tau",
        "torque_saturation_ratio", "max_abs_q", "max_abs_dq",
        "obs_min", "obs_max", "stable", "fail_reason",
    ])

    fail_reason = ""
    failed = False
    sat_ratios = []
    max_abs_roll = 0.0
    max_abs_pitch = 0.0
    max_abs_tau_overall = 0.0
    last_print_t = -1e9

    def viewer_ctx():
        if args.no_viewer:
            class _NullViewer:
                def __enter__(self_): return self_
                def __exit__(self_, *a): return False
                def is_running(self_): return True
                def sync(self_): pass
            return _NullViewer()
        return mujoco.viewer.launch_passive(model, data)

    wall_t0 = time.time()
    try:
        with viewer_ctx() as viewer:
            for step_i in range(n_policy_steps):
                t_sim = step_i * POLICY_DT
                gait_phase_ratio = (t_sim % GAIT_PERIOD) / GAIT_PERIOD

                # 1) Build single-step 77-D obs from current MuJoCo state.
                obs_77 = build_obs_77(
                    model, data,
                    qadr, dadr,
                    isaac_to_mj,
                    default_q_isaac, default_dq_isaac,
                    velocity_cmd, last_action_sdk,
                    gait_phase_ratio,
                    base_body_id,
                )

                # 2) Maintain 5-step history (oldest first).
                if step_i == 0:
                    history_77[:] = obs_77
                else:
                    history_77 = np.roll(history_77, -1, axis=0)
                    history_77[-1] = obs_77

                obs_385 = build_obs_385(history_77)
                obs_has_nan = not np.all(np.isfinite(obs_385))

                # 3) Run policy.
                action_sdk = sess.run(
                    [out_name],
                    {inp_name: obs_385.reshape(1, -1).astype(np.float32)},
                )[0].reshape(-1).astype(np.float64)
                action_has_nan = not np.all(np.isfinite(action_sdk))

                # 4) target_q -> PD torque.
                target_q_sdk = G0_DEFAULT_Q_SDK + ACTION_SCALE * action_sdk
                target_q_mj = remap_sdk_to_mj(target_q_sdk)

                # 5) Compute tau once per policy step, hold it for the whole
                #    decimation window. Matches unitree_mujoco/simulate_python/
                #    unitree_sdk2py_bridge.py:LowCmdHandler — DDS-driven PD write
                #    once, then sim_dt steps until next LowCmd arrives.
                q_mj = data.qpos[qadr].copy()
                dq_mj = data.qvel[dadr].copy()
                tau_unclipped = kp_mj * (target_q_mj - q_mj) - kd_mj * dq_mj
                tau = np.clip(tau_unclipped, ctrl_lo, ctrl_hi)
                tau_has_nan = not np.all(np.isfinite(tau))
                data.ctrl[:] = tau
                max_abs_tau = float(np.max(np.abs(tau)))
                sat_ratio = float(np.mean(np.abs(tau_unclipped) >= 0.999 * ctrl_max_abs))
                for _ in range(decimation):
                    # Apply elastic-band external force to base_link each sim step
                    # (per unitree_mujoco/simulate/src/main.cc:489-503).
                    if args.elastic_band:
                        x_base = data.qpos[0:3]
                        v_base = data.qvel[0:3]
                        delta = band_anchor - x_base
                        dist = float(np.linalg.norm(delta))
                        if dist > 1e-9:
                            dir_unit = delta / dist
                            v_along = float(np.dot(v_base, dir_unit))
                            scalar_f = band_k * (dist - band_L) - band_c * v_along
                            data.xfrc_applied[base_body_id, 0:3] = scalar_f * dir_unit
                    mujoco.mj_step(model, data)

                last_action_sdk = action_sdk.copy()
                target_q_mj_prev = target_q_mj.copy()

                # ---- metrics
                root_pos = data.qpos[0:3].copy()
                root_quat = data.qpos[3:7].copy()
                roll, pitch, yaw = quat_to_rpy_wxyz(root_quat)
                max_abs_roll = max(max_abs_roll, abs(roll))
                max_abs_pitch = max(max_abs_pitch, abs(pitch))

                max_abs_action = float(np.max(np.abs(action_sdk))) if action_sdk.size else 0.0
                mean_abs_action = float(np.mean(np.abs(action_sdk))) if action_sdk.size else 0.0
                # max_abs_tau / sat_ratio were already aggregated inside the
                # decimation loop (over the whole policy step).
                max_abs_tau_overall = max(max_abs_tau_overall, max_abs_tau)
                sat_ratios.append(sat_ratio)
                max_abs_q_now = float(np.max(np.abs(data.qpos[qadr])))
                max_abs_dq_now = float(np.max(np.abs(data.qvel[dadr])))
                obs_min = float(np.min(obs_385))
                obs_max = float(np.max(obs_385))

                # ---- failure detection
                if obs_has_nan or action_has_nan or tau_has_nan:
                    fail_reason = "nan"
                    failed = True
                elif root_pos[2] < 0.10:
                    fail_reason = "root_z<0.10"
                    failed = True
                elif abs(roll) > 1.0 or abs(pitch) > 1.0:
                    fail_reason = "rp>1rad"
                    failed = True
                elif max_abs_q_now > 50.0 or max_abs_dq_now > 200.0:
                    fail_reason = "q_or_dq_diverged"
                    failed = True

                stable = not failed
                csv_w.writerow([
                    f"{t_sim:.4f}", velocity_cmd[0], velocity_cmd[1], velocity_cmd[2],
                    root_pos[0], root_pos[1], root_pos[2], roll, pitch, yaw,
                    max_abs_action, mean_abs_action, max_abs_tau,
                    sat_ratio, max_abs_q_now, max_abs_dq_now,
                    obs_min, obs_max, int(stable), fail_reason,
                ])

                if t_sim - last_print_t >= args.print_every:
                    last_print_t = t_sim
                    print(
                        f"t={t_sim:6.2f}s  mode={args.mode}  cmd=({velocity_cmd[0]:+.2f},"
                        f"{velocity_cmd[1]:+.2f},{velocity_cmd[2]:+.2f})  "
                        f"root_z={root_pos[2]:5.3f}  rpy=({roll:+.2f},{pitch:+.2f},{yaw:+.2f})  "
                        f"|act|max={max_abs_action:5.2f} mean={mean_abs_action:5.2f}  "
                        f"|tau|max={max_abs_tau:6.2f}  sat={sat_ratio*100:5.1f}%  "
                        f"obs[{obs_min:+.2f},{obs_max:+.2f}]  "
                        f"nan(obs/act/tau)={int(obs_has_nan)}{int(action_has_nan)}{int(tau_has_nan)}"
                    )

                if failed and not args.no_abort:
                    print(f"\nABORT at t={t_sim:.3f}s: {fail_reason}")
                    break
                if failed and args.no_abort:
                    # Reset failed so we can keep recording / drawing.
                    failed = False
                    fail_reason = ""

                if not args.no_viewer:
                    viewer.sync()
                    if not viewer.is_running():
                        print("Viewer closed by user.")
                        break
                    if args.realtime:
                        target_wall = wall_t0 + t_sim + POLICY_DT
                        slack = target_wall - time.time()
                        if slack > 0:
                            time.sleep(slack)
    finally:
        csv_f.close()

    # ---- summary
    n_done = step_i + 1 if 'step_i' in locals() else 0
    final_root_z = float(data.qpos[2])
    mean_sat = float(np.mean(sat_ratios)) if sat_ratios else 0.0
    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"  duration_sim    : {n_done * POLICY_DT:.2f}s ({n_done} policy steps)")
    print(f"  final_root_z    : {final_root_z:.3f}")
    print(f"  max|roll|       : {max_abs_roll:.3f} rad")
    print(f"  max|pitch|      : {max_abs_pitch:.3f} rad")
    print(f"  max|tau|        : {max_abs_tau_overall:.3f}")
    print(f"  mean saturation : {mean_sat*100:.2f}%")
    print(f"  result          : {'FAILED' if failed else 'PASSED'}")
    if failed:
        print(f"  fail_reason     : {fail_reason}")
    print(f"  csv             : {args.csv}")

    if failed:
        sys.exit(2)


if __name__ == "__main__":
    main()
