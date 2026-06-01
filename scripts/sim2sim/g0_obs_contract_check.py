"""Observation construction contract check.

Phase-A (isaac):   reset G0-Velocity-v0 once, dump root/joint state, the per-term
                   77-D obs vector (no noise), and the velocity_command sample.
                   Run under the isaaclab conda env.

Phase-B (mujoco):  load the isaac dump, write the SAME state into the MuJoCo
                   model (qpos / qvel), then rebuild the 77-D obs from MuJoCo
                   state alone, segment-by-segment compare to isaac.
                   Run under the mujoco conda env.

This checks the obs *function*, not the simulator transition. No ONNX, no
policy, no DDS, no motor_id.

Usage:

    # phase A (isaaclab env)
    python scripts/sim2sim/g0_obs_contract_check.py --phase isaac \\
        --task G0-Velocity-v0 \\
        --isaac-dump logs/sim2sim/g0_obs_isaac.npz

    # phase B (g0_mujoco env)
    python scripts/sim2sim/g0_obs_contract_check.py --phase mujoco \\
        --model source/g0_robot_lab/g0_robot_lab/assets/robots/g0/mujoco/model_patched.xml \\
        --isaac-dump logs/sim2sim/g0_obs_isaac.npz \\
        --report logs/sim2sim/g0_obs_contract_report.txt \\
        --combined-npz logs/sim2sim/g0_obs_contract_values.npz
"""

import argparse
import os
import sys


# -----------------------------------------------------------------------------
# Constants shared by both phases
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

# Per-term scales from velocity_env_cfg.py PolicyCfg.
SCALES = {
    "base_ang_vel": 0.2,
    "projected_gravity": 1.0,
    "velocity_commands": 1.0,
    "joint_pos_rel": 1.0,
    "joint_vel_rel": 0.05,
    "last_action": 1.0,
    "gait_phase": 1.0,
}

# Slice layout of the single 77-D step.
LAYOUT = [
    ("base_ang_vel",      0,  3),
    ("projected_gravity", 3,  6),
    ("velocity_commands", 6,  9),
    ("joint_pos_rel",     9,  31),
    ("joint_vel_rel",     31, 53),
    ("last_action",       53, 75),
    ("gait_phase",        75, 77),
]


# =============================================================================
# Phase A: IsaacLab
# =============================================================================

def run_isaac(args):
    # AppLauncher MUST be created before importing isaaclab.envs / gym envs.
    from isaaclab.app import AppLauncher

    parser_inner = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser_inner)
    inner_args, _ = parser_inner.parse_known_args([])
    inner_args.headless = True
    inner_args.enable_cameras = False
    app_launcher = AppLauncher(inner_args)
    simulation_app = app_launcher.app

    # Now safe to import isaaclab pieces.
    import math
    import numpy as np
    import torch
    import gymnasium as gym

    # Register G0-Velocity-v0.
    import g0_robot_lab.tasks  # noqa: F401

    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(args.task, device="cpu", num_envs=1)

    # Disable observation corruption so the dumped 77-D matches a noise-free
    # mathematical construction. Keep scales intact.
    pol = cfg.observations.policy
    pol.enable_corruption = False
    pol.history_length = 0  # single-step view; we only need one frame
    pol.concatenate_terms = True

    env = gym.make(args.task, cfg=cfg)
    obs_dict, _ = env.reset(seed=args.seed)

    obs_policy = obs_dict["policy"][0].cpu().numpy()  # (77,)
    assert obs_policy.shape == (77,), f"Expected (77,), got {obs_policy.shape}"

    # Pull raw state from the articulation.
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    rdata = robot.data

    joint_names_isaac = list(robot.joint_names)
    joint_pos       = rdata.joint_pos[0].cpu().numpy()           # (22,)
    joint_vel       = rdata.joint_vel[0].cpu().numpy()           # (22,)
    default_joint_pos = rdata.default_joint_pos[0].cpu().numpy() # (22,)
    default_joint_vel = rdata.default_joint_vel[0].cpu().numpy() # (22,)
    root_pos_w      = rdata.root_pos_w[0].cpu().numpy()          # (3,)
    root_quat_w     = rdata.root_quat_w[0].cpu().numpy()         # (4,) wxyz
    root_lin_vel_w  = rdata.root_lin_vel_w[0].cpu().numpy()      # (3,)
    root_ang_vel_w  = rdata.root_ang_vel_w[0].cpu().numpy()      # (3,)
    root_ang_vel_b  = rdata.root_ang_vel_b[0].cpu().numpy()      # (3,)
    proj_grav_b     = rdata.projected_gravity_b[0].cpu().numpy() # (3,)

    velocity_cmd = base_env.command_manager.get_command("base_velocity")[0].cpu().numpy()  # (3,)
    last_action  = base_env.action_manager.prev_action[0].cpu().numpy()                   # (22,)

    # gait_phase: episode_length_buf == 0 right after reset.
    elen = int(base_env.episode_length_buf[0].item())
    step_dt = float(base_env.step_dt)
    period = 0.8
    g_phase = (elen * step_dt) % period / period
    gait = np.array([math.sin(g_phase * 2.0 * math.pi),
                     math.cos(g_phase * 2.0 * math.pi)], dtype=np.float64)

    # Build manual 77-D (noise-free) for cross-check against env.observation_manager.
    manual = np.zeros(77, dtype=np.float64)
    manual[0:3]   = root_ang_vel_b * SCALES["base_ang_vel"]
    manual[3:6]   = proj_grav_b   * SCALES["projected_gravity"]
    manual[6:9]   = velocity_cmd  * SCALES["velocity_commands"]
    manual[9:31]  = (joint_pos - default_joint_pos) * SCALES["joint_pos_rel"]
    manual[31:53] = (joint_vel - default_joint_vel) * SCALES["joint_vel_rel"]
    manual[53:75] = last_action  * SCALES["last_action"]
    manual[75:77] = gait         * SCALES["gait_phase"]

    diff_manager_vs_manual = np.max(np.abs(obs_policy - manual))
    print(f"max |env.obs[policy]_77  -  manual_77|  =  {diff_manager_vs_manual:.6e}")
    if diff_manager_vs_manual > 1e-5:
        print("WARNING: env observation manager and manual reconstruction differ.")
        print("This usually means scales/order/noise differ from what we assumed.")
    else:
        print("OK: manual reconstruction matches env.observation_manager exactly.")

    print()
    print("articulation.joint_names (Isaac order, len=22):")
    for i, n in enumerate(joint_names_isaac):
        print(f"  [{i:2d}] {n}")
    print()
    print(f"velocity_command (sampled at reset): {velocity_cmd}")
    print(f"episode_length_buf: {elen}, step_dt: {step_dt}, gait_phase ratio: {g_phase}")
    print(f"root_pos_w:  {root_pos_w}")
    print(f"root_quat_w (w,x,y,z): {root_quat_w}")

    os.makedirs(os.path.dirname(args.isaac_dump), exist_ok=True)
    np.savez(
        args.isaac_dump,
        joint_names=np.array(joint_names_isaac),
        joint_pos=joint_pos.astype(np.float64),
        joint_vel=joint_vel.astype(np.float64),
        default_joint_pos=default_joint_pos.astype(np.float64),
        default_joint_vel=default_joint_vel.astype(np.float64),
        root_pos_w=root_pos_w.astype(np.float64),
        root_quat_w=root_quat_w.astype(np.float64),
        root_lin_vel_w=root_lin_vel_w.astype(np.float64),
        root_ang_vel_w=root_ang_vel_w.astype(np.float64),
        root_ang_vel_b=root_ang_vel_b.astype(np.float64),
        projected_gravity_b=proj_grav_b.astype(np.float64),
        velocity_command=velocity_cmd.astype(np.float64),
        last_action=last_action.astype(np.float64),
        gait_phase=gait.astype(np.float64),
        episode_length=np.int64(elen),
        step_dt=np.float64(step_dt),
        obs_policy_77=obs_policy.astype(np.float64),
        obs_manual_77=manual.astype(np.float64),
        sdk_names=np.array(G0_JOINT_SDK_NAMES),
    )
    print(f"\nWrote {args.isaac_dump}")

    env.close()
    simulation_app.close()


# =============================================================================
# Phase B: MuJoCo
# =============================================================================

def quat_apply_inverse_wxyz(q_wxyz, v):
    """Rotate v from world frame into the body frame defined by quat q (w,x,y,z).

    Matches math_utils.quat_apply_inverse in IsaacLab.
    """
    import numpy as np
    w, x, y, z = q_wxyz
    # Rotation matrix R from body to world for q.  Body = R^T @ world.
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R.T @ np.asarray(v, dtype=np.float64)


def run_mujoco(args):
    import math
    import numpy as np
    import mujoco

    dump = np.load(args.isaac_dump, allow_pickle=False)
    joint_names_isaac     = [s.decode() if isinstance(s, bytes) else str(s) for s in dump["joint_names"]]
    joint_pos             = dump["joint_pos"]
    joint_vel             = dump["joint_vel"]
    default_joint_pos     = dump["default_joint_pos"]
    default_joint_vel     = dump["default_joint_vel"]
    root_pos_w            = dump["root_pos_w"]
    root_quat_w           = dump["root_quat_w"]
    root_lin_vel_w        = dump["root_lin_vel_w"]
    root_ang_vel_w        = dump["root_ang_vel_w"]
    root_ang_vel_b_isaac  = dump["root_ang_vel_b"]
    proj_grav_b_isaac     = dump["projected_gravity_b"]
    velocity_cmd          = dump["velocity_command"]
    last_action           = dump["last_action"]
    gait_isaac            = dump["gait_phase"]
    obs_manual_77         = dump["obs_manual_77"]

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    if model.nu != 22:
        raise RuntimeError(f"Expected nu=22, got {model.nu}")

    # Build actuator -> joint name map (and qpos/qvel addresses).
    mj_actuator_joint_names = []
    qadr = np.empty(model.nu, dtype=np.int64)
    dadr = np.empty(model.nu, dtype=np.int64)
    for mj_i in range(model.nu):
        jid = int(model.actuator_trnid[mj_i, 0])
        qadr[mj_i] = int(model.jnt_qposadr[jid])
        dadr[mj_i] = int(model.jnt_dofadr[jid])
        mj_actuator_joint_names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid))

    # Verify isaac joint order == mujoco actuator order. If not, build a remap.
    if joint_names_isaac == mj_actuator_joint_names:
        isaac_to_mj = np.arange(22)
        print("Isaac joint_names == MuJoCo actuator joint order (no remap needed).")
    else:
        isaac_to_mj = np.empty(22, dtype=np.int64)
        mj_index = {n: i for i, n in enumerate(mj_actuator_joint_names)}
        for i, n in enumerate(joint_names_isaac):
            if n not in mj_index:
                raise RuntimeError(f"Isaac joint '{n}' not in MuJoCo actuator list")
            isaac_to_mj[i] = mj_index[n]
        print("WARNING: Isaac and MuJoCo joint orders differ. Will remap.")
        print("  isaac_to_mj =", isaac_to_mj.tolist())

    # ------------------------------------------------------------------ write
    # Write the *same* physical state into MuJoCo: root pose + joint pos/vel.
    # qpos layout: [base x,y,z, qw,qx,qy,qz, joint_q ...]
    # qvel layout: [base vx,vy,vz (world), wx,wy,wz (world for freejoint), joint_dq ...]
    data.qpos[:] = 0
    data.qvel[:] = 0
    data.qpos[0:3] = root_pos_w
    data.qpos[3:7] = root_quat_w     # both isaac and mujoco use w,x,y,z
    for isaac_i, mj_i in enumerate(isaac_to_mj):
        data.qpos[qadr[mj_i]] = joint_pos[isaac_i]
        data.qvel[dadr[mj_i]] = joint_vel[isaac_i]
    # Root velocity: freejoint qvel is in WORLD frame for both linear and angular.
    data.qvel[0:3] = root_lin_vel_w
    data.qvel[3:6] = root_ang_vel_w

    mujoco.mj_forward(model, data)

    # ------------------------------------------------------------------ build
    # 1. base_ang_vel (body frame) — use mj_objectVelocity local for safety.
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    vel6 = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, base_id, vel6, 1)
    # Layout when flg_local=1: [angvel(3), linvel(3)] in body frame.
    base_ang_vel_b_mj = vel6[0:3].copy()

    # 2. projected_gravity (body frame, unit vector).
    proj_grav_b_mj = quat_apply_inverse_wxyz(root_quat_w, np.array([0.0, 0.0, -1.0]))

    # 3. velocity_commands: replay from isaac (sampled at reset).
    velocity_cmd_mj = velocity_cmd.copy()

    # 4. joint_pos_rel (MuJoCo order = G0_JOINT_NAMES).
    q_mj = data.qpos[qadr].copy()
    dq_mj = data.qvel[dadr].copy()
    default_q_mj = np.zeros(22, dtype=np.float64)
    default_dq_mj = np.zeros(22, dtype=np.float64)
    for isaac_i, mj_i in enumerate(isaac_to_mj):
        default_q_mj[mj_i] = default_joint_pos[isaac_i]
        default_dq_mj[mj_i] = default_joint_vel[isaac_i]
    joint_pos_rel_mj = q_mj - default_q_mj
    joint_vel_rel_mj = dq_mj - default_dq_mj

    # IMPORTANT: the policy obs ordering for joint_pos_rel / joint_vel_rel is
    # Isaac articulation order, which is *not* the same as MuJoCo actuator
    # order (= URDF order). Reindex MJ values into Isaac positions.
    joint_pos_rel_isaac_order = np.empty(22, dtype=np.float64)
    joint_vel_rel_isaac_order = np.empty(22, dtype=np.float64)
    for isaac_i, mj_i in enumerate(isaac_to_mj):
        joint_pos_rel_isaac_order[isaac_i] = joint_pos_rel_mj[mj_i]
        joint_vel_rel_isaac_order[isaac_i] = joint_vel_rel_mj[mj_i]

    # 5. last_action (SDK order). Default action at reset is zeros.
    last_action_mj = last_action.copy()

    # 6. gait_phase: ratio = (episode_length * step_dt) % period / period, at reset = 0
    gait_mj = np.array([math.sin(0.0), math.cos(0.0)], dtype=np.float64)

    # Assemble mujoco-side 77-D using ISAAC ordering for joint segments.
    obs_mj_77 = np.zeros(77, dtype=np.float64)
    obs_mj_77[0:3]   = base_ang_vel_b_mj            * SCALES["base_ang_vel"]
    obs_mj_77[3:6]   = proj_grav_b_mj               * SCALES["projected_gravity"]
    obs_mj_77[6:9]   = velocity_cmd_mj              * SCALES["velocity_commands"]
    obs_mj_77[9:31]  = joint_pos_rel_isaac_order    * SCALES["joint_pos_rel"]
    obs_mj_77[31:53] = joint_vel_rel_isaac_order    * SCALES["joint_vel_rel"]
    obs_mj_77[53:75] = last_action_mj      * SCALES["last_action"]
    obs_mj_77[75:77] = gait_mj             * SCALES["gait_phase"]

    # ---------------------------------------------------------------- compare
    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 90)
    emit("G0 obs construction contract check (Isaac vs MuJoCo, same state)")
    emit("=" * 90)
    emit(f"model        : {args.model}")
    emit(f"isaac dump   : {args.isaac_dump}")
    emit(f"root_pos_w   : isaac={root_pos_w}  mujoco_set={data.qpos[0:3]}")
    emit(f"root_quat_w  : isaac={root_quat_w}")
    emit(f"velocity_cmd : {velocity_cmd}")
    emit("")
    emit("Aux sanity:")
    emit(f"  |base_ang_vel_b  isaac vs mj_local |  max = "
         f"{np.max(np.abs(root_ang_vel_b_isaac - base_ang_vel_b_mj)):.3e}")
    emit(f"  |projected_grav_b isaac vs mj_calc |  max = "
         f"{np.max(np.abs(proj_grav_b_isaac - proj_grav_b_mj)):.3e}")
    emit("")
    emit(f"{'segment':22s} {'idx':>8} {'max|diff|':>12} {'mean|diff|':>12}  first mismatching indices")
    emit("-" * 90)

    overall_pass = True
    for name, lo, hi in LAYOUT:
        seg_isaac = obs_manual_77[lo:hi]
        seg_mj    = obs_mj_77[lo:hi]
        diff = seg_mj - seg_isaac
        adiff = np.abs(diff)
        max_d = float(adiff.max()) if adiff.size else 0.0
        mean_d = float(adiff.mean()) if adiff.size else 0.0
        bad_mask = adiff > 1e-5
        bad_idx = np.where(bad_mask)[0]
        seg_pass = (max_d <= 1e-5)
        overall_pass &= seg_pass
        bad_repr = ""
        if bad_idx.size:
            shown = bad_idx[:5]
            bad_repr = "  " + ", ".join(
                f"[{int(i)}] isaac={seg_isaac[int(i)]:+.4f} mj={seg_mj[int(i)]:+.4f}"
                for i in shown
            )
            if bad_idx.size > 5:
                bad_repr += f"  ... (+{bad_idx.size - 5} more)"
        emit(f"{name:22s} {lo:>3}:{hi:<3}  {max_d:>12.3e} {mean_d:>12.3e}{bad_repr}")

    emit("")
    emit("=" * 90)
    emit(f"Overall pass (all segments max|diff| <= 1e-5): {overall_pass}")
    emit("=" * 90)

    # Per the user-stated pass criteria:
    emit("")
    emit("Pass-criteria checks:")
    jpos_rel = obs_mj_77[9:31]
    jvel_rel = obs_mj_77[31:53]
    la = obs_mj_77[53:75]
    gv = obs_mj_77[3:6]
    emit(f"  (1) joint_pos_rel ~ 0  : max|x|={np.max(np.abs(jpos_rel)):.3e}  "
         f"({'OK' if np.max(np.abs(jpos_rel)) < 1e-4 else 'CHECK (init randomization?)'})")
    emit(f"  (2) joint_vel_rel ~ 0  : max|x|={np.max(np.abs(jvel_rel)):.3e}  "
         f"({'OK' if np.max(np.abs(jvel_rel)) < 1e-3 else 'CHECK'})")
    emit(f"  (3) last_action == 0   : max|x|={np.max(np.abs(la)):.3e}  "
         f"({'OK' if np.max(np.abs(la)) == 0 else 'FAIL'})")
    emit(f"  (4) velocity_command   : isaac sampled = {velocity_cmd}  "
         f"(replayed in mujoco)")
    emit(f"  (5) projected_gravity  : mj={gv}  "
         f"({'OK pointing down' if gv[2] < -0.9 else 'CHECK orientation'})")

    # Save report + npz.
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        f.write("\n".join(lines) + "\n")
    emit(f"\nWrote report: {args.report}")

    os.makedirs(os.path.dirname(args.combined_npz), exist_ok=True)
    np.savez(
        args.combined_npz,
        obs_isaac_77=obs_manual_77,
        obs_mujoco_77=obs_mj_77,
        diff=obs_mj_77 - obs_manual_77,
        layout_names=np.array([n for n, _, _ in LAYOUT]),
        layout_lo=np.array([lo for _, lo, _ in LAYOUT]),
        layout_hi=np.array([hi for _, _, hi in LAYOUT]),
    )
    print(f"Wrote values: {args.combined_npz}")

    if not overall_pass:
        sys.exit(1)


# =============================================================================
# Entry point
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("isaac", "mujoco"), required=True)
    p.add_argument("--task", default="G0-Velocity-v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="source/g0_robot_lab/g0_robot_lab/assets/robots/g0/mujoco/model_patched.xml")
    p.add_argument("--isaac-dump", default="logs/sim2sim/g0_obs_isaac.npz")
    p.add_argument("--report", default="logs/sim2sim/g0_obs_contract_report.txt")
    p.add_argument("--combined-npz", default="logs/sim2sim/g0_obs_contract_values.npz")
    args = p.parse_args()

    if args.phase == "isaac":
        run_isaac(args)
    else:
        run_mujoco(args)


if __name__ == "__main__":
    main()
