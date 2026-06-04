"""G0 sim2sim: closed-loop MuJoCo rollout of the trained locomotion policy.

Structured after the humanoid-gym reference
(https://github.com/roboman-ly/humanoid-gym-modified/blob/main/humanoid/scripts/sim2sim.py):
a ``Sim2simCfg`` block, ``get_obs`` / ``pd_control`` helpers, and a single
``run_mujoco`` loop. The observation/action contract, joint orderings, gains
and scales, however, are G0-specific and mirror the proven runner at
``scripts/sim2sim/g0_mujoco_onnx_gui_runner.py`` (the authoritative source).

Pipeline per policy step (every POLICY_DT = 0.02 s)::

    MuJoCo state
      -> single-step 77-D obs (per-term, scaled)
      -> 5-step per-term-grouped, oldest-first 385-D history obs
      -> policy.onnx -> action_sdk (22,)
      -> target_q_sdk = default_q_sdk + ACTION_SCALE * action_sdk
      -> target_q_mj  = remap(SDK_TO_MJ, target_q_sdk)
      -> tau_mj = kp*(target_q_mj - q_mj) - kd*dq_mj   (clipped to ctrlrange)
      -> data.ctrl[:] = tau_mj
      -> mj_step()  x decimation

Three joint orders are in play:
    SDK   : policy obs/action order        (G0_JOINT_SDK_NAMES)
    MJ    : MuJoCo actuator / motor_id      (G0_JOINT_NAMES_MJ; == actuator order in the MJCF)
    Isaac : IsaacLab articulation obs order (G0_JOINT_NAMES_ISAAC)

The MJCF actuators are <motor>, so data.ctrl is always raw torque.

Example::

    /home/lz/miniconda3/envs/g0_mujoco/bin/python scripts/sim2sim.py \
        --policy logs/rsl_rl/g0_velocity/<run>/exported/policy.onnx \
        --vx 0.3 --duration 20
"""

from __future__ import annotations

import argparse
import math
import os
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np
import onnxruntime as ort

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_MODEL = os.path.join(
    _REPO_ROOT,
    "source/g0_robot_lab/g0_robot_lab/assets/robots/g0/mjcf/g0_mujoco.xml",
)


# -----------------------------------------------------------------------------
# Joint orders  (see g0.py for the canonical lists)
# -----------------------------------------------------------------------------

# Policy obs/action order.
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

# MuJoCo actuator order == motor_id order in g0_mujoco.xml <actuator>.
# We verify this against the loaded model at runtime instead of trusting it.
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

# IsaacLab articulation order used for joint_pos_rel / joint_vel_rel obs terms.
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

# Default standing pose in SDK order (mirrors g0.py G0_DEFAULT_JOINT_POS).
G0_DEFAULT_Q_SDK = np.array([
    -0.20, 0.0, 0.0, 0.34, -0.14, 0.0,   # left leg
    -0.20, 0.0, 0.0, 0.34, -0.14, 0.0,   # right leg
     0.0, 0.0,                           # waist
     0.30, 0.25, 0.0, -0.97,             # left arm
     0.30, -0.25, 0.0, -0.97,            # right arm
], dtype=np.float64)

# For each SDK index, the MJ index of the same joint (target = source[SDK_TO_MJ]
# i.e. vec_mj[SDK_TO_MJ] = vec_sdk). Built once from the name lists.
SDK_TO_MJ = np.array(
    [G0_JOINT_NAMES_MJ.index(n) for n in G0_JOINT_SDK_NAMES], dtype=np.int64
)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

class Sim2simCfg:
    """Single place for the contract constants (matches velocity_env_cfg.py)."""

    # --- timing
    POLICY_DT = 0.02          # IsaacLab decimation 4 * sim.dt 0.005

    # --- action
    NUM_ACTIONS = 22
    ACTION_SCALE = 0.12

    # --- obs layout (per-term, 5-step history)
    TERM_DIMS = [3, 3, 3, 22, 22, 22, 2]   # base_ang_vel, proj_grav, cmd, jpos, jvel, last_act, gait; sum=77
    HISTORY_LEN = 5
    OBS_DIM_PER_STEP = 77
    OBS_DIM_HISTORY = 385

    # --- per-term obs scales
    SCALE_BASE_ANG_VEL = 0.2
    SCALE_PROJ_GRAV = 1.0
    SCALE_VEL_CMD = 1.0
    SCALE_JPOS_REL = 1.0
    SCALE_JVEL_REL = 0.05
    SCALE_LAST_ACT = 1.0
    SCALE_GAIT = 1.0

    # --- gait clock
    GAIT_PERIOD = 0.8

    # --- explicit-PD gains in SDK order (mirror g0.py actuator cfg)
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
# Helpers
# -----------------------------------------------------------------------------

def remap_sdk_to_mj(vec_sdk: np.ndarray) -> np.ndarray:
    """SDK-ordered vector -> MJ-ordered vector."""
    vec_mj = np.empty_like(vec_sdk)
    vec_mj[SDK_TO_MJ] = vec_sdk
    return vec_mj


def quat_apply_inverse_wxyz(q_wxyz, v) -> np.ndarray:
    """Rotate world vector v into the body frame given body quat (w,x,y,z)."""
    w, x, y, z = q_wxyz
    rot = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)
    return rot.T @ np.asarray(v, dtype=np.float64)


def quaternion_to_euler_array(q_wxyz):
    """(w,x,y,z) -> (roll, pitch, yaw), intrinsic XYZ. Used only for logging."""
    w, x, y, z = q_wxyz
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def pd_control(target_q_mj, q_mj, kp_mj, dq_mj, kd_mj, ctrl_lo, ctrl_hi):
    """Explicit joint-space PD, clipped to the actuator ctrlrange (= torque limits)."""
    tau = kp_mj * (target_q_mj - q_mj) - kd_mj * dq_mj
    return np.clip(tau, ctrl_lo, ctrl_hi)


def build_joint_address_table(model):
    """qpos/qvel addresses and joint name for each MuJoCo actuator (motor_id)."""
    qadr = np.empty(model.nu, dtype=np.int64)
    dadr = np.empty(model.nu, dtype=np.int64)
    joint_name = []
    for mj_i in range(model.nu):
        jid = int(model.actuator_trnid[mj_i, 0])
        qadr[mj_i] = int(model.jnt_qposadr[jid])
        dadr[mj_i] = int(model.jnt_dofadr[jid])
        joint_name.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid))
    return qadr, dadr, joint_name


def get_obs(model, data, qadr, dadr, isaac_to_mj, default_q_isaac,
            velocity_cmd, last_action_sdk, gait_phase_ratio, base_body_id, cfg):
    """Single-step 77-D observation from the current MuJoCo state."""
    # base angular velocity in body frame
    vel6 = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_body_id, vel6, 1)
    base_ang_vel_b = vel6[0:3]

    # projected gravity in body frame
    proj_grav_b = quat_apply_inverse_wxyz(data.qpos[3:7], np.array([0.0, 0.0, -1.0]))

    # joints in Isaac order
    q_isaac = np.empty(22, dtype=np.float64)
    dq_isaac = np.empty(22, dtype=np.float64)
    for isaac_i, mj_i in enumerate(isaac_to_mj):
        q_isaac[isaac_i] = data.qpos[qadr[mj_i]]
        dq_isaac[isaac_i] = data.qvel[dadr[mj_i]]
    jpos_rel = q_isaac - default_q_isaac
    jvel_rel = dq_isaac  # default dq is zero

    gait = np.array([
        math.sin(gait_phase_ratio * 2.0 * math.pi),
        math.cos(gait_phase_ratio * 2.0 * math.pi),
    ], dtype=np.float64)

    obs = np.zeros(cfg.OBS_DIM_PER_STEP, dtype=np.float64)
    obs[0:3]   = base_ang_vel_b * cfg.SCALE_BASE_ANG_VEL
    obs[3:6]   = proj_grav_b    * cfg.SCALE_PROJ_GRAV
    obs[6:9]   = velocity_cmd   * cfg.SCALE_VEL_CMD
    obs[9:31]  = jpos_rel       * cfg.SCALE_JPOS_REL
    obs[31:53] = jvel_rel       * cfg.SCALE_JVEL_REL
    obs[53:75] = last_action_sdk * cfg.SCALE_LAST_ACT
    obs[75:77] = gait           * cfg.SCALE_GAIT
    return obs


def build_obs_history(history, cfg):
    """history: (HISTORY_LEN, 77) oldest-first -> (385,) per-term grouped."""
    out, offset = [], 0
    for d in cfg.TERM_DIMS:
        out.append(history[:, offset:offset + d].reshape(-1))
        offset += d
    return np.concatenate(out, axis=0)


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def run_mujoco(args, cfg: Sim2simCfg):
    # ---- model
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    if model.nu != cfg.NUM_ACTIONS:
        raise RuntimeError(f"Expected nu={cfg.NUM_ACTIONS}, got {model.nu}")
    if args.timestep is not None:
        model.opt.timestep = float(args.timestep)

    qadr, dadr, joint_per_actuator = build_joint_address_table(model)
    # Verify motor_id -> joint order matches our hard-coded MJ list.
    for mj_i, jn in enumerate(joint_per_actuator):
        if jn != G0_JOINT_NAMES_MJ[mj_i]:
            raise RuntimeError(
                f"Actuator(motor_id) order drift at {mj_i}: model={jn} vs expected={G0_JOINT_NAMES_MJ[mj_i]}"
            )

    mj_index_by_name = {n: i for i, n in enumerate(joint_per_actuator)}
    isaac_to_mj = np.array([mj_index_by_name[n] for n in G0_JOINT_NAMES_ISAAC], dtype=np.int64)

    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_body_id < 0:
        raise RuntimeError("Missing body 'base_link' in model")

    # ---- defaults & gains (in MJ order)
    default_q_mj = remap_sdk_to_mj(G0_DEFAULT_Q_SDK)
    default_q_isaac = np.array(
        [default_q_mj[mj_index_by_name[n]] for n in G0_JOINT_NAMES_ISAAC], dtype=np.float64
    )
    kp_mj = remap_sdk_to_mj(np.array([cfg.KP_SDK[n] for n in G0_JOINT_SDK_NAMES])) * args.kp_scale
    kd_mj = remap_sdk_to_mj(np.array([cfg.KD_SDK[n] for n in G0_JOINT_SDK_NAMES])) * args.kd_scale
    ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
    ctrl_hi = model.actuator_ctrlrange[:, 1].copy()

    # ---- ONNX policy
    sess = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    if cfg.OBS_DIM_HISTORY not in [s for s in sess.get_inputs()[0].shape if isinstance(s, int)]:
        print(f"WARNING: policy input {sess.get_inputs()[0].shape} does not contain "
              f"{cfg.OBS_DIM_HISTORY}; obs layout may mismatch.")

    # ---- velocity command
    velocity_cmd = (np.zeros(3) if args.mode == "zero_cmd"
                    else np.array([args.vx, args.vy, args.wz], dtype=np.float64))

    # ---- reset to default standing pose (model has no keyframe)
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, args.init_height]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[qadr] = default_q_mj
    mujoco.mj_forward(model, data)

    decimation = max(1, int(round(cfg.POLICY_DT / model.opt.timestep)))
    # duration <= 0 means run unbounded: GUI runs until you close the window
    # (or Ctrl-C headless). Otherwise stop after the requested seconds.
    unbounded = args.duration is None or args.duration <= 0.0
    n_policy_steps = None if unbounded else int(round(args.duration / cfg.POLICY_DT))

    print("=" * 80)
    print("G0 sim2sim closed-loop runner")
    print(f"  model      : {args.model}")
    print(f"  policy     : {args.policy}")
    print(f"  mode/cmd   : {args.mode} ({velocity_cmd[0]:+.2f},{velocity_cmd[1]:+.2f},{velocity_cmd[2]:+.2f})")
    print(f"  mj dt      : {model.opt.timestep}  policy dt: {cfg.POLICY_DT}  decimation: {decimation}")
    print(f"  pd mode    : {'zero-order-hold (legacy)' if args.pd_zero_order_hold else 'per-substep (default)'}")
    print(f"  duration   : {'unbounded (close window / Ctrl-C to stop)' if unbounded else f'{args.duration}s ({n_policy_steps} policy steps)'}")
    print(f"  kp/kd scale: {args.kp_scale}/{args.kd_scale}")
    print("=" * 80)

    # ---- history (oldest first), warm-started with the first obs
    history = np.zeros((cfg.HISTORY_LEN, cfg.OBS_DIM_PER_STEP), dtype=np.float64)
    last_action_sdk = np.zeros(cfg.NUM_ACTIONS, dtype=np.float64)
    obs_hist = deque(maxlen=cfg.HISTORY_LEN)  # kept for parity with humanoid-gym style

    def viewer_ctx():
        if args.no_viewer:
            class _Null:
                def __enter__(self_): return self_
                def __exit__(self_, *a): return False
                def is_running(self_): return True
                def sync(self_): pass
            return _Null()
        return mujoco.viewer.launch_passive(model, data)

    failed, fail_reason, step_i, n_done = False, "", 0, 0
    wall_t0 = time.time()
    with viewer_ctx() as viewer:
        while n_policy_steps is None or step_i < n_policy_steps:
            t_sim = step_i * cfg.POLICY_DT
            gait_phase_ratio = (t_sim % cfg.GAIT_PERIOD) / cfg.GAIT_PERIOD

            # 1) single-step obs
            obs_77 = get_obs(model, data, qadr, dadr, isaac_to_mj, default_q_isaac,
                             velocity_cmd, last_action_sdk, gait_phase_ratio, base_body_id, cfg)

            # 2) maintain 5-step history (oldest first); warm-start on step 0
            if step_i == 0:
                history[:] = obs_77
                obs_hist.extend([obs_77] * cfg.HISTORY_LEN)
            else:
                history = np.roll(history, -1, axis=0)
                history[-1] = obs_77
                obs_hist.append(obs_77)
            obs_385 = build_obs_history(history, cfg)

            # 3) policy
            action_sdk = sess.run(
                [out_name], {inp_name: obs_385.reshape(1, -1).astype(np.float32)}
            )[0].reshape(-1).astype(np.float64)

            # 4) target_q is fixed for this policy step; torque is PD on it.
            target_q_mj = remap_sdk_to_mj(G0_DEFAULT_Q_SDK + cfg.ACTION_SCALE * action_sdk)
            if args.pd_zero_order_hold:
                # Legacy/escape-hatch: compute tau once and freeze it across the
                # decimation window. VERIFIED UNSTABLE even for static standing
                # (stale kd*dq lets the joints oscillate and saturate). Not
                # faithful to Isaac or the real motor PD — kept for comparison.
                tau = pd_control(target_q_mj, data.qpos[qadr].copy(), kp_mj,
                                 data.qvel[dadr].copy(), kd_mj, ctrl_lo, ctrl_hi)
                data.ctrl[:] = tau
                for _ in range(decimation):
                    mujoco.mj_step(model, data)
            else:
                # Default: recompute tau every sim substep from fresh q/dq.
                # Matches Isaac ImplicitActuator (PD at sim.dt) and the real
                # robot's onboard high-rate PD; the held command is q_des/kp/kd,
                # not the torque.
                for _ in range(decimation):
                    tau = pd_control(target_q_mj, data.qpos[qadr], kp_mj,
                                     data.qvel[dadr], kd_mj, ctrl_lo, ctrl_hi)
                    data.ctrl[:] = tau
                    mujoco.mj_step(model, data)
            n_done += 1

            last_action_sdk = action_sdk

            # ---- metrics / failure detection
            root_z = float(data.qpos[2])
            roll, pitch, yaw = quaternion_to_euler_array(data.qpos[3:7])
            if not np.all(np.isfinite(obs_385)) or not np.all(np.isfinite(action_sdk)):
                failed, fail_reason = True, "nan"
            elif root_z < 0.10:
                failed, fail_reason = True, "root_z<0.10"
            elif abs(roll) > 1.0 or abs(pitch) > 1.0:
                failed, fail_reason = True, "rp>1rad"

            if step_i % max(1, int(args.print_every / cfg.POLICY_DT)) == 0:
                print(f"t={t_sim:6.2f}s  root_z={root_z:5.3f}  rpy=({roll:+.2f},{pitch:+.2f},{yaw:+.2f})  "
                      f"|act|max={np.max(np.abs(action_sdk)):5.2f}  |tau|max={np.max(np.abs(tau)):6.3f}")

            if failed and not args.no_abort:
                print(f"\nABORT at t={t_sim:.3f}s: {fail_reason}")
                break
            if failed and args.no_abort:
                failed, fail_reason = False, ""

            if not args.no_viewer:
                viewer.sync()
                if not viewer.is_running():
                    print("Viewer closed by user.")
                    break
                if args.realtime:
                    slack = (wall_t0 + t_sim + cfg.POLICY_DT) - time.time()
                    if slack > 0:
                        time.sleep(slack)

            step_i += 1

        # Bounded run that finished on its own: keep the GUI open so the final
        # pose can be inspected. (Unbounded runs only exit when the window is
        # already closed, so this is a no-op for them.)
        if not args.no_viewer and viewer.is_running():
            print("\nRollout finished. Viewer stays open — close the window to exit.")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.02)

    print("=" * 80)
    print(f"SUMMARY  steps={n_done}  sim_time={n_done * cfg.POLICY_DT:.2f}s  "
          f"final_root_z={float(data.qpos[2]):.3f}  "
          f"result={'FAILED (' + fail_reason + ')' if failed else 'PASSED'}")
    print("=" * 80)
    return 2 if failed else 0


def build_arg_parser():
    ap = argparse.ArgumentParser(description="G0 sim2sim MuJoCo closed-loop runner")
    ap.add_argument("--model", default=_DEFAULT_MODEL, help="path to g0_mujoco.xml")
    ap.add_argument("--policy", required=True, help="path to exported policy.onnx (input 385, output 22)")
    ap.add_argument("--mode", choices=("zero_cmd", "cmd"), default="zero_cmd")
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=0.0,
                    help="rollout seconds; <=0 (default) runs unbounded until you close the GUI window")
    ap.add_argument("--init-height", type=float, default=0.23, help="initial base z (m)")
    ap.add_argument("--kp-scale", type=float, default=1.0)
    ap.add_argument("--kd-scale", type=float, default=1.0)
    ap.add_argument("--timestep", type=float, default=None, help="override MJCF sim timestep")
    ap.add_argument("--print-every", type=float, default=0.5, help="print status every N sim seconds")
    ap.add_argument("--no-viewer", action="store_true", help="headless run (no GUI)")
    ap.add_argument("--realtime", action="store_true", help="throttle GUI playback to wall-clock 1x")
    ap.add_argument("--pd-zero-order-hold", action="store_true",
                    help="legacy: freeze PD torque across the decimation window. "
                         "Verified unstable even at zero action; the default recomputes "
                         "PD every sim substep (≈ Isaac implicit PD / real motor PD).")
    ap.add_argument("--no-abort", action="store_true", help="keep stepping past a fall (for GUI viewing)")
    return ap


def main():
    args = build_arg_parser().parse_args()
    raise SystemExit(run_mujoco(args, Sim2simCfg()))


if __name__ == "__main__":
    main()
