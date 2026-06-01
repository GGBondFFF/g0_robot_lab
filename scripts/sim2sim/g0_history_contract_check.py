"""History layout contract check for G0 policy obs.

After reading IsaacLab's ObservationManager + CircularBuffer source, the
candidate layouts for the 385-D policy obs are:

  A.  per-step, newest first:    [obs_t, obs_t-1, obs_t-2, obs_t-3, obs_t-4]
  B.  per-step, oldest first:    [obs_t-4, obs_t-3, obs_t-2, obs_t-1, obs_t]
  C.  per-term, oldest first:    for each term k in declaration order,
                                 [term_k_t-4, term_k_t-3, ..., term_k_t]
  D.  per-term, newest first:    for each term k,
                                 [term_k_t, term_k_t-1, ..., term_k_t-4]

CircularBuffer.buffer returns (batch, max_len, dim) with index 0 = oldest, and
ObservationManager flattens via .reshape(batch, -1) before cat'ing terms in
declaration order. From the source we expect layout C, but we verify
empirically against the actual 385-D returned by env.step().

We also verify the initial-buffer behavior: at reset, the first observation
should be replicated 5 times in every buffer slot.

This script runs entirely inside the isaaclab conda env. No ONNX. No MuJoCo.
No closed loop. Always feeds zero actions.
"""

import argparse
import math
import os
import sys


# Single-step obs term layout (matches velocity_env_cfg.py PolicyCfg).
TERM_LAYOUT = [
    ("base_ang_vel",      3,  0.2 ),
    ("projected_gravity", 3,  1.0 ),
    ("velocity_commands", 3,  1.0 ),
    ("joint_pos_rel",     22, 1.0 ),
    ("joint_vel_rel",     22, 0.05),
    ("last_action",       22, 1.0 ),
    ("gait_phase",        2,  1.0 ),
]
SINGLE_STEP_DIM = sum(d for _, d, _ in TERM_LAYOUT)  # 77
HISTORY_LEN = 5
EXPECTED_TOTAL = SINGLE_STEP_DIM * HISTORY_LEN        # 385


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="G0-Velocity-v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-steps", type=int, default=6)
    p.add_argument("--report", default="logs/sim2sim/g0_history_contract_report.txt")
    p.add_argument("--values-npz", default="logs/sim2sim/g0_history_contract_values.npz")
    args = p.parse_args()

    from isaaclab.app import AppLauncher
    inner = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(inner)
    inner_args, _ = inner.parse_known_args([])
    inner_args.headless = True
    inner_args.enable_cameras = False
    app_launcher = AppLauncher(inner_args)
    sim_app = app_launcher.app

    import numpy as np
    import torch
    import gymnasium as gym
    import g0_robot_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(args.task, device="cpu", num_envs=1)
    pol = cfg.observations.policy
    pol.enable_corruption = False
    # Keep history_length = 5 (the trained config). Verify total dim.
    if pol.history_length != HISTORY_LEN:
        print(f"NOTE: policy.history_length = {pol.history_length}, overriding to {HISTORY_LEN}")
        pol.history_length = HISTORY_LEN
    pol.concatenate_terms = True

    env = gym.make(args.task, cfg=cfg)
    obs_dict, _ = env.reset(seed=args.seed)
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    rdata = robot.data
    step_dt = float(base_env.step_dt)
    gait_period = 0.8

    def grab_single_step_obs():
        """Build the noise-free 77-D obs from the current articulation state."""
        joint_pos       = rdata.joint_pos[0].cpu().numpy()
        joint_vel       = rdata.joint_vel[0].cpu().numpy()
        d_pos           = rdata.default_joint_pos[0].cpu().numpy()
        d_vel           = rdata.default_joint_vel[0].cpu().numpy()
        root_ang_vel_b  = rdata.root_ang_vel_b[0].cpu().numpy()
        proj_grav_b     = rdata.projected_gravity_b[0].cpu().numpy()
        cmd             = base_env.command_manager.get_command("base_velocity")[0].cpu().numpy()
        last_action     = base_env.action_manager.prev_action[0].cpu().numpy()
        elen            = int(base_env.episode_length_buf[0].item())
        g_phase         = (elen * step_dt) % gait_period / gait_period
        gait = np.array([math.sin(g_phase * 2.0 * math.pi),
                         math.cos(g_phase * 2.0 * math.pi)], dtype=np.float64)
        out = np.zeros(SINGLE_STEP_DIM, dtype=np.float64)
        out[0:3]   = root_ang_vel_b * 0.2
        out[3:6]   = proj_grav_b    * 1.0
        out[6:9]   = cmd            * 1.0
        out[9:31]  = (joint_pos - d_pos) * 1.0
        out[31:53] = (joint_vel - d_vel) * 0.05
        out[53:75] = last_action    * 1.0
        out[75:77] = gait           * 1.0
        return out, elen, cmd

    # Step 0: post-reset state.
    obs_385_0 = obs_dict["policy"][0].cpu().numpy()
    assert obs_385_0.shape == (EXPECTED_TOTAL,), f"unexpected 385-D shape: {obs_385_0.shape}"
    single_0, elen_0, cmd_0 = grab_single_step_obs()

    single_steps = [single_0]
    policy_obs   = [obs_385_0]
    episode_lens = [elen_0]

    # Roll forward with zero actions.
    zero_action = torch.zeros((1, 22), dtype=torch.float32)
    for t in range(1, args.num_steps + 1):
        result = env.step(zero_action)
        obs_t_dict = result[0]
        single_t, elen_t, _ = grab_single_step_obs()
        single_steps.append(single_t)
        policy_obs.append(obs_t_dict["policy"][0].cpu().numpy())
        episode_lens.append(elen_t)

    single_steps = np.stack(single_steps, axis=0)   # (T, 77)
    policy_obs   = np.stack(policy_obs,   axis=0)   # (T, 385)
    episode_lens = np.array(episode_lens, dtype=np.int64)
    T = len(single_steps)

    # Verify each step is internally consistent: the policy_obs should be a
    # known reordering of the previous 5 single-step obs (with replication
    # before the buffer is full).
    def history_window(t):
        """Return the 5 single-step obs that *should* be in the history buffer
        at time t (after t-th compute). Index 0 = oldest, 4 = newest. Before
        the buffer is full, the very first observation is replicated forward
        until it is naturally pushed out."""
        # CircularBuffer.append() fills all slots with the first push.
        # Thereafter each new push overwrites the oldest slot.
        # At time t (with t pushes done, so num_pushes = t+1), buffer holds
        # the most recent 5 entries; if t < HISTORY_LEN-1, missing slots are
        # the first obs replicated.
        win = []
        for k in range(HISTORY_LEN):
            # k = 0 -> oldest (t - 4), k = 4 -> newest (t)
            idx = t - (HISTORY_LEN - 1 - k)
            if idx < 0:
                idx = 0
            win.append(single_steps[idx])
        return np.stack(win, axis=0)  # (5, 77)

    def build_candidate(layout: str, t: int) -> np.ndarray:
        win = history_window(t)  # (5, 77), [oldest, ..., newest]
        if layout == "C_per_term_oldest_first":
            # for each term k, take its slice across all 5 history steps in
            # oldest->newest order, then flatten in that order, then concat.
            out = []
            offset = 0
            for _, dim, _ in TERM_LAYOUT:
                # win[:, offset:offset+dim] has shape (5, dim) -> flatten row-major
                out.append(win[:, offset:offset + dim].reshape(-1))
                offset += dim
            return np.concatenate(out, axis=0)
        if layout == "D_per_term_newest_first":
            out = []
            offset = 0
            for _, dim, _ in TERM_LAYOUT:
                out.append(win[::-1, offset:offset + dim].reshape(-1))
                offset += dim
            return np.concatenate(out, axis=0)
        if layout == "A_per_step_newest_first":
            return win[::-1].reshape(-1)
        if layout == "B_per_step_oldest_first":
            return win.reshape(-1)
        raise ValueError(layout)

    CANDIDATES = [
        "A_per_step_newest_first",
        "B_per_step_oldest_first",
        "C_per_term_oldest_first",
        "D_per_term_newest_first",
    ]

    # Compare each candidate to the actual 385-D at each time step.
    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit("=" * 90)
    emit("G0 history layout contract check")
    emit("=" * 90)
    emit(f"task          : {args.task}")
    emit(f"num_steps     : {args.num_steps}  (total time-points T = {T})")
    emit(f"history_len   : {HISTORY_LEN}")
    emit(f"single_step_d : {SINGLE_STEP_DIM}")
    emit(f"total_d       : {EXPECTED_TOTAL}")
    emit(f"step_dt       : {step_dt}")
    emit(f"velocity cmd  : {cmd_0}  (sampled at reset, constant within 6 steps)")
    emit("")
    emit(f"episode_length_buf per step: {episode_lens.tolist()}")
    emit("")

    candidate_max_diff = {c: [] for c in CANDIDATES}
    for t in range(T):
        for c in CANDIDATES:
            cand = build_candidate(c, t)
            d = float(np.max(np.abs(cand - policy_obs[t])))
            candidate_max_diff[c].append(d)

    emit(f"{'t':>2}  " + "  ".join(f"{c:32s}" for c in CANDIDATES))
    emit("-" * (4 + 34 * len(CANDIDATES)))
    for t in range(T):
        row = f"{t:>2}  " + "  ".join(
            f"{candidate_max_diff[c][t]:>10.3e}{'':22s}" for c in CANDIDATES
        )
        emit(row)

    emit("")
    winners = []
    for c in CANDIDATES:
        passes = all(d <= 1e-5 for d in candidate_max_diff[c])
        emit(f"  {c:32s}  -> {'MATCH' if passes else 'differs'}")
        if passes:
            winners.append(c)

    emit("")
    if len(winners) == 1:
        emit(f"==> Confirmed history layout: {winners[0]}")
    elif len(winners) > 1:
        emit(f"==> Multiple layouts match (ambiguous): {winners}")
        emit("    This usually means the obs are degenerate (e.g. constant)")
        emit("    over the test window. Run with more diverse states.")
    else:
        emit("==> No candidate matches. Inspect raw per-step diffs.")

    # Initial-buffer behavior: is t=0 actually obs_0 replicated 5x?
    emit("")
    emit("Initial buffer behavior (t=0 just after reset):")
    if winners:
        cand0 = build_candidate(winners[0], 0)
        d0 = float(np.max(np.abs(cand0 - policy_obs[0])))
        emit(f"  build_candidate('{winners[0]}', t=0) reproduces policy_obs[0]?  "
             f"max|diff| = {d0:.3e}  ({'YES' if d0 <= 1e-5 else 'NO'})")
        emit(f"  -> Initial buffer = first obs replicated {HISTORY_LEN} times: "
             f"{'YES' if d0 <= 1e-5 else 'NO'}")
    else:
        # As a fallback, just check directly.
        rep = np.tile(single_steps[0], HISTORY_LEN)
        d_rep = float(np.max(np.abs(rep - policy_obs[0])))
        emit(f"  flat replicate single_step[0] x 5 matches policy_obs[0]? "
             f"max|diff| = {d_rep:.3e}")

    # ---------------------------------------------------------------- save
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        f.write("\n".join(lines) + "\n")
    emit(f"\nWrote {args.report}")

    os.makedirs(os.path.dirname(args.values_npz), exist_ok=True)
    np.savez(
        args.values_npz,
        single_steps=single_steps,
        policy_obs=policy_obs,
        episode_lens=episode_lens,
        candidate_max_diff_A=np.array(candidate_max_diff["A_per_step_newest_first"]),
        candidate_max_diff_B=np.array(candidate_max_diff["B_per_step_oldest_first"]),
        candidate_max_diff_C=np.array(candidate_max_diff["C_per_term_oldest_first"]),
        candidate_max_diff_D=np.array(candidate_max_diff["D_per_term_newest_first"]),
        winner=np.array(winners[0] if winners else "none"),
        term_names=np.array([n for n, _, _ in TERM_LAYOUT]),
        term_dims=np.array([d for _, d, _ in TERM_LAYOUT]),
    )
    print(f"Wrote {args.values_npz}")

    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
