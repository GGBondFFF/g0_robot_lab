"""Diagnostic dump + cross-check for ONNX action at reset.

Goal: rule in / rule out an obs construction gap vs an Isaac-vs-MuJoCo physics
gap by comparing the policy's action on the *same conceptual reset state*.

Two phases:

  --phase isaac      (run in g0_isaaclab env)
      Reset G0-Velocity-v0 with history_length=5 in the policy group, evaluate
      the trained policy (rsl_rl) on the reset obs, dump:
          obs_385 (per-term grouped, oldest-first, as the policy actually sees it)
          action  (22,)
          single-step obs_77 + raw state for reference

  --phase mujoco     (run in g0_mujoco env)
      Three checks:
       (1) Load the dumped obs_385, send it to policy.onnx, compare action.
       (2) Reset MJCF to default_stand, build obs_385 from MuJoCo state using
           the same logic as g0_mujoco_onnx_gui_runner, compare to dumped obs_385.
       (3) Send the MuJoCo-built obs_385 through policy.onnx, compare action to
           the dumped action.

Outputs a side-by-side report. No DDS, no real robot, no policy training.
"""

import argparse
import os
import sys


# Per-term layout shared with the runner.
TERM_DIMS = [3, 3, 3, 22, 22, 22, 2]
HISTORY_LEN = 5
OBS_DIM_PER_STEP = 77


# =============================================================================
# Phase A: Isaac
# =============================================================================

def run_isaac(args):
    from isaaclab.app import AppLauncher
    import argparse as _argparse

    parser_inner = _argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser_inner)
    inner_args, _ = parser_inner.parse_known_args([])
    inner_args.headless = True
    inner_args.enable_cameras = False
    app_launcher = AppLauncher(inner_args)
    simulation_app = app_launcher.app

    import numpy as np
    import torch
    import gymnasium as gym

    import g0_robot_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    # ---- env (history_length=5 in policy group, no corruption)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    pol = cfg.observations.policy
    pol.enable_corruption = False
    pol.history_length = HISTORY_LEN
    pol.concatenate_terms = True

    env = gym.make(args.task, cfg=cfg)
    obs_dict, _ = env.reset(seed=args.seed)
    obs_385 = obs_dict["policy"][0].cpu().numpy().astype(np.float64)
    assert obs_385.shape == (385,), f"expected (385,), got {obs_385.shape}"

    # ---- load exported jit policy.pt (already verified equal to policy.onnx)
    policy_pt = torch.jit.load(args.policy_pt, map_location=args.device).eval()
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    rdata = robot.data
    joint_names_isaac = list(robot.joint_names)
    default_joint_pos = rdata.default_joint_pos[0].cpu().numpy().astype(np.float64)
    default_joint_vel = rdata.default_joint_vel[0].cpu().numpy().astype(np.float64)

    # ---- record initial state (step 0)
    def snapshot():
        return dict(
            joint_pos=rdata.joint_pos[0].cpu().numpy().astype(np.float64),
            joint_vel=rdata.joint_vel[0].cpu().numpy().astype(np.float64),
            root_pos_w=rdata.root_pos_w[0].cpu().numpy().astype(np.float64),
            root_quat_w=rdata.root_quat_w[0].cpu().numpy().astype(np.float64),
            root_lin_vel_w=rdata.root_lin_vel_w[0].cpu().numpy().astype(np.float64),
            root_ang_vel_w=rdata.root_ang_vel_w[0].cpu().numpy().astype(np.float64),
            velocity_command=base_env.command_manager.get_command("base_velocity")[0].cpu().numpy().astype(np.float64),
            last_action=base_env.action_manager.prev_action[0].cpu().numpy().astype(np.float64),
        )

    # Step 0: state BEFORE first action; action computed from obs at reset.
    n_steps = max(1, int(args.rollout_steps))
    obs_traj = np.zeros((n_steps, 385), dtype=np.float64)
    action_traj = np.zeros((n_steps, 22), dtype=np.float64)
    state_traj = {k: [] for k in ["joint_pos", "joint_vel",
                                   "root_pos_w", "root_quat_w",
                                   "root_lin_vel_w", "root_ang_vel_w",
                                   "velocity_command", "last_action"]}

    for t in range(n_steps):
        snap = snapshot()
        for k, v in snap.items():
            state_traj[k].append(v)
        obs_t = obs_dict["policy"][0].cpu().numpy().astype(np.float64)
        assert obs_t.shape == (385,), obs_t.shape
        obs_traj[t] = obs_t
        with torch.no_grad():
            a = policy_pt(torch.from_numpy(obs_t).to(args.device).unsqueeze(0).float())
            a_np = a.cpu().numpy().reshape(-1).astype(np.float64)
        action_traj[t] = a_np
        obs_dict, _, _, _, _ = env.step(a)
    # State AFTER last action — useful so the replay can verify the final step too.
    state_post = snapshot()
    for k, v in state_post.items():
        state_traj[k].append(v)

    # Backward-compat: single-step variables (= step 0)
    obs_385 = obs_traj[0]
    action = action_traj[0]
    joint_pos = state_traj["joint_pos"][0]
    joint_vel = state_traj["joint_vel"][0]
    root_pos_w = state_traj["root_pos_w"][0]
    root_quat_w = state_traj["root_quat_w"][0]
    root_lin_vel_w = state_traj["root_lin_vel_w"][0]
    root_ang_vel_w = state_traj["root_ang_vel_w"][0]
    velocity_cmd = state_traj["velocity_command"][0]
    last_action = state_traj["last_action"][0]

    print("=" * 90)
    print("Phase A (Isaac) — diagnose dump")
    print("=" * 90)
    print(f"  task         : {args.task}")
    print(f"  policy.pt    : {args.policy_pt}")
    print(f"  obs_385 shape: {obs_385.shape}   [min={obs_385.min():+.3f}, max={obs_385.max():+.3f}]")
    print(f"  action shape : {action.shape}    [min={action.min():+.3f}, max={action.max():+.3f}]")
    print(f"  |action|max  : {abs(action).max():.4f}")
    print(f"  velocity_cmd : {velocity_cmd}")
    print(f"  root_pos_w   : {root_pos_w}")
    print(f"  root_quat_w  : {root_quat_w}")
    print(f"  episode_len  : {int(base_env.episode_length_buf[0].item())}")
    print(f"  rollout_steps: {n_steps}")
    if n_steps > 1:
        # Brief trajectory summary
        rp = np.stack(state_traj["root_pos_w"])
        rq = np.stack(state_traj["root_quat_w"])
        print(f"  root_z range : [{rp[:,2].min():.3f}, {rp[:,2].max():.3f}]")
        print(f"  |action|max  trajectory: t=0 {abs(action_traj[0]).max():.3f}  "
              f"t={n_steps//2} {abs(action_traj[n_steps//2]).max():.3f}  "
              f"t={n_steps-1} {abs(action_traj[n_steps-1]).max():.3f}")

    os.makedirs(os.path.dirname(args.dump), exist_ok=True)
    np.savez(
        args.dump,
        # single-step (step 0) keys preserved for backward compat
        obs_385=obs_385,
        action=action,
        joint_names=np.array(joint_names_isaac),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        default_joint_pos=default_joint_pos,
        default_joint_vel=default_joint_vel,
        root_pos_w=root_pos_w,
        root_quat_w=root_quat_w,
        root_lin_vel_w=root_lin_vel_w,
        root_ang_vel_w=root_ang_vel_w,
        velocity_command=velocity_cmd,
        last_action=last_action,
        # full rollout
        traj_obs_385=obs_traj,
        traj_action=action_traj,
        traj_joint_pos=np.stack(state_traj["joint_pos"]),
        traj_joint_vel=np.stack(state_traj["joint_vel"]),
        traj_root_pos_w=np.stack(state_traj["root_pos_w"]),
        traj_root_quat_w=np.stack(state_traj["root_quat_w"]),
        traj_root_lin_vel_w=np.stack(state_traj["root_lin_vel_w"]),
        traj_root_ang_vel_w=np.stack(state_traj["root_ang_vel_w"]),
        traj_velocity_command=np.stack(state_traj["velocity_command"]),
        traj_last_action=np.stack(state_traj["last_action"]),
    )
    print(f"\nWrote {args.dump}")

    env.close()
    simulation_app.close()


# =============================================================================
# Phase B: MuJoCo + ONNX cross-check
# =============================================================================

def run_mujoco(args):
    import math
    import numpy as np
    import mujoco
    import onnxruntime as ort

    # Import constants from the runner so we stay in sync.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import g0_mujoco_onnx_gui_runner as R

    dump = np.load(args.dump, allow_pickle=False)
    obs_385_isaac = dump["obs_385"].astype(np.float64)
    action_isaac = dump["action"].astype(np.float64)
    velocity_cmd = dump["velocity_command"].astype(np.float64)

    # ---- 1. Replay dumped obs_385 through our ONNX session
    sess = ort.InferenceSession(args.policy, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    out = sess.get_outputs()[0].name
    action_onnx_replay = sess.run(
        [out], {inp: obs_385_isaac.reshape(1, -1).astype(np.float32)}
    )[0].reshape(-1).astype(np.float64)

    # ---- 2. Build obs_385 fresh from MuJoCo state at default_stand
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    qadr, dadr, joint_per_actuator = R.build_joint_address_table(model)
    mj_index_by_name = {n: i for i, n in enumerate(joint_per_actuator)}
    isaac_to_mj = np.array(
        [mj_index_by_name[n] for n in R.G0_JOINT_NAMES_ISAAC], dtype=np.int64
    )
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
    if key_id < 0:
        raise RuntimeError(f"missing keyframe: {args.keyframe}")
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    if args.match_state:
        # Force MuJoCo state to exactly match the Isaac dump.
        joint_pos_isaac = dump["joint_pos"]
        joint_vel_isaac = dump["joint_vel"]
        root_pos_w = dump["root_pos_w"]
        root_quat_w = dump["root_quat_w"]
        root_lin_vel_w = dump["root_lin_vel_w"]
        root_ang_vel_w = dump["root_ang_vel_w"]
        data.qpos[0:3] = root_pos_w
        data.qpos[3:7] = root_quat_w
        for isaac_i, mj_i in enumerate(isaac_to_mj):
            data.qpos[qadr[mj_i]] = joint_pos_isaac[isaac_i]
            data.qvel[dadr[mj_i]] = joint_vel_isaac[isaac_i]
        data.qvel[0:3] = root_lin_vel_w
        data.qvel[3:6] = root_ang_vel_w
    mujoco.mj_forward(model, data)

    default_q_mj = R.remap_sdk_to_mj(R.G0_DEFAULT_Q_SDK)
    default_q_isaac = np.empty(22, dtype=np.float64)
    for isaac_i, mj_i in enumerate(isaac_to_mj):
        default_q_isaac[isaac_i] = default_q_mj[mj_i]
    default_dq_isaac = np.zeros(22, dtype=np.float64)

    obs_77_mj = R.build_obs_77(
        model, data, qadr, dadr, isaac_to_mj,
        default_q_isaac, default_dq_isaac,
        velocity_cmd, np.zeros(22),
        0.0,
        base_body_id,
    )
    history_77 = np.tile(obs_77_mj, (HISTORY_LEN, 1))
    obs_385_mj = R.build_obs_385(history_77)

    # ---- 3. Send the MuJoCo obs_385 through ONNX
    action_onnx_mj = sess.run(
        [out], {inp: obs_385_mj.reshape(1, -1).astype(np.float32)}
    )[0].reshape(-1).astype(np.float64)

    # ---- Report
    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 90)
    emit("Phase B (MuJoCo) — ONNX action cross-check at reset")
    emit("=" * 90)
    emit(f"  dump        : {args.dump}")
    emit(f"  policy      : {args.policy}")
    emit(f"  model       : {args.model}  keyframe={args.keyframe}")
    emit(f"  velocity_cmd: {velocity_cmd}")
    emit("")

    def diff_stats(name, a, b):
        d = a - b
        emit(f"  {name}: shape={a.shape}  max|diff|={np.max(np.abs(d)):.6e}  "
             f"mean|diff|={np.mean(np.abs(d)):.6e}  "
             f"rel(L2)={np.linalg.norm(d) / (np.linalg.norm(b) + 1e-12):.6e}")

    emit("[1] action_onnx(replay isaac obs_385)  vs  action_isaac (from policy.pt in isaac):")
    diff_stats("    action diff", action_onnx_replay, action_isaac)
    emit("")
    emit("[2] obs_385_mujoco (built from MJCF default_stand)  vs  obs_385_isaac:")
    diff_stats("    obs diff   ", obs_385_mj, obs_385_isaac)
    emit("")
    emit("[3] action_onnx(mujoco obs_385)  vs  action_isaac:")
    diff_stats("    action diff", action_onnx_mj, action_isaac)
    emit("")

    # Per-term obs diff
    emit("Per-term obs_385 diff (mujoco - isaac), per history step:")
    offset_isaac = 0
    term_names = ["base_ang_vel", "proj_grav", "vel_cmd",
                  "jpos_rel", "jvel_rel", "last_action", "gait"]
    for d, name in zip(TERM_DIMS, term_names):
        seg_size = d * HISTORY_LEN
        a = obs_385_mj[offset_isaac:offset_isaac + seg_size].reshape(HISTORY_LEN, d)
        b = obs_385_isaac[offset_isaac:offset_isaac + seg_size].reshape(HISTORY_LEN, d)
        adiff = np.abs(a - b)
        emit(f"  {name:13s}  per-slot max|diff| = " +
             " ".join(f"{float(adiff[k].max()):.3e}" for k in range(HISTORY_LEN)))
        offset_isaac += seg_size

    emit("")
    emit("First 22 action values, side by side:")
    emit(f"  {'sdk_i':>5} {'sdk_joint':30s} {'isaac':>10} {'onnx_replay':>12} {'onnx_mj':>10}")
    for i, n in enumerate(R.G0_JOINT_SDK_NAMES):
        emit(f"  {i:>5} {n:30s} "
             f"{action_isaac[i]:+.4f}   {action_onnx_replay[i]:+.4f}     {action_onnx_mj[i]:+.4f}")

    emit("")
    emit("=" * 90)
    # Pass criteria
    eps = 1e-4
    p1 = np.max(np.abs(action_onnx_replay - action_isaac)) <= eps
    p2 = np.max(np.abs(obs_385_mj - obs_385_isaac)) <= eps
    p3 = np.max(np.abs(action_onnx_mj - action_isaac)) <= eps
    emit(f"  [1] ONNX(isaac obs) == policy.pt(isaac obs)        : {'PASS' if p1 else 'FAIL'}")
    emit(f"  [2] obs_385 mujoco == obs_385 isaac                : {'PASS' if p2 else 'FAIL'}")
    emit(f"  [3] ONNX(mujoco obs) == policy.pt(isaac obs)       : {'PASS' if p3 else 'FAIL'}")
    emit("=" * 90)

    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\nWrote report: {args.report}")


# =============================================================================
# Phase C: Replay Isaac action trajectory in MuJoCo (pure physics replay)
# =============================================================================

def run_replay_rollout(args):
    """Initialize MuJoCo to Isaac dump's step-0 state, then for each step apply
    Isaac's recorded action[t] (NO ONNX inference) and step physics. Compare
    state trajectories step-by-step to localize physics-level divergence.
    """
    import numpy as np
    import mujoco

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import g0_mujoco_onnx_gui_runner as R

    dump = np.load(args.dump, allow_pickle=False)
    if "traj_action" not in dump.files:
        raise RuntimeError("dump has no rollout trajectory; re-run isaac phase with --rollout-steps N")
    traj_action_isaac = dump["traj_action"].astype(np.float64)         # (T, 22)  SDK order
    traj_root_pos_isaac = dump["traj_root_pos_w"].astype(np.float64)   # (T+1, 3)
    traj_root_quat_isaac = dump["traj_root_quat_w"].astype(np.float64) # (T+1, 4)
    traj_joint_pos_isaac = dump["traj_joint_pos"].astype(np.float64)   # (T+1, 22)  Isaac order
    init_root_pos = dump["root_pos_w"].astype(np.float64)
    init_root_quat = dump["root_quat_w"].astype(np.float64)
    init_joint_pos = dump["joint_pos"].astype(np.float64)
    init_joint_vel = dump["joint_vel"].astype(np.float64)
    init_root_lin_vel = dump["root_lin_vel_w"].astype(np.float64)
    init_root_ang_vel = dump["root_ang_vel_w"].astype(np.float64)

    T = traj_action_isaac.shape[0]
    print(f"replay-rollout: T={T} steps, comparing MuJoCo physics vs Isaac trajectory")
    print(f"  dump   : {args.dump}")
    print(f"  model  : {args.model}")

    # ---- load mujoco
    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)
    if args.timestep is not None:
        model.opt.timestep = float(args.timestep)
    qadr, dadr, joint_per_actuator = R.build_joint_address_table(model)
    mj_index_by_name = {n: i for i, n in enumerate(joint_per_actuator)}
    isaac_to_mj = np.array(
        [mj_index_by_name[n] for n in R.G0_JOINT_NAMES_ISAAC], dtype=np.int64
    )

    default_q_mj = R.remap_sdk_to_mj(R.G0_DEFAULT_Q_SDK)
    kp_sdk = np.array([R.KP_SDK[n] for n in R.G0_JOINT_SDK_NAMES], dtype=np.float64)
    kd_sdk = np.array([R.KD_SDK[n] for n in R.G0_JOINT_SDK_NAMES], dtype=np.float64)
    kp_mj = R.remap_sdk_to_mj(kp_sdk)
    kd_mj = R.remap_sdk_to_mj(kd_sdk)
    ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
    ctrl_hi = model.actuator_ctrlrange[:, 1].copy()

    POLICY_DT = R.POLICY_DT
    decimation = max(1, int(round(POLICY_DT / model.opt.timestep)))

    # ---- initialize MuJoCo state = Isaac step-0 state
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    data.qpos[0:3] = init_root_pos
    data.qpos[3:7] = init_root_quat
    for isaac_i, mj_i in enumerate(isaac_to_mj):
        data.qpos[qadr[mj_i]] = init_joint_pos[isaac_i]
        data.qvel[dadr[mj_i]] = init_joint_vel[isaac_i]
    data.qvel[0:3] = init_root_lin_vel
    data.qvel[3:6] = init_root_ang_vel
    mujoco.mj_forward(model, data)

    # ---- rollout
    mj_root_pos = np.zeros((T + 1, 3))
    mj_root_quat = np.zeros((T + 1, 4))
    mj_joint_pos_isaac = np.zeros((T + 1, 22))
    mj_max_abs_tau = np.zeros(T)

    def record(t):
        mj_root_pos[t] = data.qpos[0:3]
        mj_root_quat[t] = data.qpos[3:7]
        for isaac_i, mj_i in enumerate(isaac_to_mj):
            mj_joint_pos_isaac[t, isaac_i] = data.qpos[qadr[mj_i]]

    record(0)
    for t in range(T):
        action_sdk = traj_action_isaac[t]
        target_q_sdk = R.G0_DEFAULT_Q_SDK + R.ACTION_SCALE * action_sdk
        target_q_mj = R.remap_sdk_to_mj(target_q_sdk)
        q_mj = data.qpos[qadr].copy()
        dq_mj = data.qvel[dadr].copy()
        tau = np.clip(kp_mj * (target_q_mj - q_mj) - kd_mj * dq_mj, ctrl_lo, ctrl_hi)
        mj_max_abs_tau[t] = float(np.max(np.abs(tau)))
        data.ctrl[:] = tau
        for _ in range(decimation):
            mujoco.mj_step(model, data)
        record(t + 1)

    # ---- compare
    drz = mj_root_pos[:, 2] - traj_root_pos_isaac[:, 2]
    dxy = np.linalg.norm(mj_root_pos[:, :2] - traj_root_pos_isaac[:, :2], axis=1)
    # roll/pitch difference via quat→rpy
    def rpy_arr(q_arr):
        out = np.zeros((q_arr.shape[0], 3))
        for i in range(q_arr.shape[0]):
            out[i] = R.quat_to_rpy_wxyz(q_arr[i])
        return out
    rpy_mj = rpy_arr(mj_root_quat)
    rpy_isaac = rpy_arr(traj_root_quat_isaac)
    drpy = rpy_mj - rpy_isaac
    djp = mj_joint_pos_isaac - traj_joint_pos_isaac
    djp_max = np.max(np.abs(djp), axis=1)

    print()
    print("=" * 110)
    print(f"{'t':>4} {'cmd_dt':>7} | {'rz_is':>6} {'rz_mj':>6} {'Δrz':>7} | "
          f"{'roll_is':>7} {'roll_mj':>7} {'pitch_is':>8} {'pitch_mj':>8} | "
          f"{'Δroll':>7} {'Δpitch':>7} | {'max|Δq|':>8} | {'|tau|max':>9} {'|act|max':>9}")
    print("-" * 110)
    stride = max(1, T // 25)
    for t in [0] + list(range(stride, T + 1, stride)):
        if t > T: break
        amax = abs(traj_action_isaac[t-1]).max() if 0 < t <= T else 0.0
        tau_max = mj_max_abs_tau[t-1] if 0 < t <= T else 0.0
        print(f"{t:>4} {t*POLICY_DT:>7.3f} | {traj_root_pos_isaac[t,2]:>6.3f} {mj_root_pos[t,2]:>6.3f} {drz[t]:>+7.3f} | "
              f"{rpy_isaac[t,0]:>+7.3f} {rpy_mj[t,0]:>+7.3f} {rpy_isaac[t,1]:>+8.3f} {rpy_mj[t,1]:>+8.3f} | "
              f"{drpy[t,0]:>+7.3f} {drpy[t,1]:>+7.3f} | {djp_max[t]:>8.4f} | "
              f"{tau_max:>9.3f} {amax:>9.3f}")
    print()
    # Find first step where |Δpitch|, |Δroll| or max|Δq| exceeds eps
    eps_rpy = 0.05      # rad
    eps_q = 0.02        # rad
    first = None
    for t in range(1, T + 1):
        if abs(drpy[t, 0]) > eps_rpy or abs(drpy[t, 1]) > eps_rpy or djp_max[t] > eps_q:
            first = t
            break
    if first is None:
        print(f"PASS: trajectories within eps (Δrpy<{eps_rpy} rad, max|Δq|<{eps_q} rad) over all {T} steps.")
    else:
        print(f"First divergence at t={first} (sim_time {first*POLICY_DT:.3f}s):")
        print(f"  Δrz={drz[first]:+.4f}  Δroll={drpy[first,0]:+.4f}  Δpitch={drpy[first,1]:+.4f}  max|Δq|={djp_max[first]:.4f}")
        # show top-3 joints by |Δq|
        order = np.argsort(-np.abs(djp[first]))[:5]
        for o in order:
            print(f"    isaac_joint[{o}] {R.G0_JOINT_NAMES_ISAAC[o]:30s} "
                  f"isaac={traj_joint_pos_isaac[first,o]:+.4f}  mj={mj_joint_pos_isaac[first,o]:+.4f}  "
                  f"Δ={djp[first,o]:+.4f}")

    # save the trajectories for later plotting
    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        with open(args.report, "w") as f:
            f.write(f"# replay-rollout report  T={T}  policy_dt={POLICY_DT}\n")
            f.write("# t  rz_is  rz_mj  d_rz  roll_is  roll_mj  d_roll  pitch_is  pitch_mj  d_pitch  max|Δq|  max|tau|_mj  max|action|_is\n")
            for t in range(T + 1):
                amax = abs(traj_action_isaac[t-1]).max() if 0 < t <= T else 0.0
                tau_max = mj_max_abs_tau[t-1] if 0 < t <= T else 0.0
                f.write(f"{t} {traj_root_pos_isaac[t,2]:.5f} {mj_root_pos[t,2]:.5f} {drz[t]:+.5f} "
                        f"{rpy_isaac[t,0]:+.5f} {rpy_mj[t,0]:+.5f} {drpy[t,0]:+.5f} "
                        f"{rpy_isaac[t,1]:+.5f} {rpy_mj[t,1]:+.5f} {drpy[t,1]:+.5f} "
                        f"{djp_max[t]:.5f} {tau_max:.5f} {amax:.5f}\n")
        print(f"\nWrote {args.report}")


# =============================================================================
# Entry
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=("isaac", "mujoco", "replay-rollout"), required=True)
    # isaac phase
    p.add_argument("--task", default="G0-Velocity-v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--policy-pt", default="logs/rsl_rl/g0_velocity/2026-05-26_18-36-57/exported/policy.pt")
    p.add_argument("--dump", default="logs/sim2sim/g0_action_diagnose_dump.npz")
    # mujoco phase
    p.add_argument("--model", default="source/g0_robot_lab/g0_robot_lab/assets/robots/g0/mujoco/model_patched.xml")
    p.add_argument("--policy", default="logs/rsl_rl/g0_velocity/2026-05-26_18-36-57/exported/policy.onnx")
    p.add_argument("--keyframe", default="default_stand")
    p.add_argument("--report", default="logs/sim2sim/g0_action_diagnose_report.txt")
    p.add_argument("--match-state", action="store_true",
                   help="Phase B: copy Isaac dump's root pose + joints into MuJoCo before building obs.")
    p.add_argument("--rollout-steps", type=int, default=1,
                   help="Phase isaac: number of env.step() iterations to record (default 1 = single-step contract).")
    p.add_argument("--timestep", type=float, default=None,
                   help="Phase replay-rollout: override MuJoCo sim timestep.")
    args = p.parse_args()
    if args.phase == "isaac":
        run_isaac(args)
    elif args.phase == "mujoco":
        run_mujoco(args)
    else:
        run_replay_rollout(args)


if __name__ == "__main__":
    main()
