"""Dry-run the G0-Velocity-v0 env to verify EventCfg wiring.

Run in g0_isaaclab env. Loads the env with num_envs=4, resets, prints:
  - The EventCfg entries (term names, modes, params).
  - For each randomized quantity, the values realized per env after reset
    (so we can verify P1 / P2 / P4 actually fire and produce the expected
    distribution).

No training, no policy, no rollout — just a configuration sanity check.
"""

import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="G0-Velocity-v0")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from isaaclab.app import AppLauncher
    import argparse as _ap

    inner = _ap.ArgumentParser()
    AppLauncher.add_app_launcher_args(inner)
    iargs, _ = inner.parse_known_args([])
    iargs.headless = True
    iargs.enable_cameras = False
    app_launcher = AppLauncher(iargs)
    sim_app = app_launcher.app

    import numpy as np
    import torch
    import gymnasium as gym

    import g0_robot_lab.tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    cfg = parse_env_cfg(args.task, device="cpu", num_envs=args.num_envs)

    # ---------------- EventCfg summary -------------
    print("=" * 90)
    print(f"EventCfg for task={args.task}")
    print("=" * 90)
    for name in cfg.events.__dict__:
        term = getattr(cfg.events, name)
        if term is None:
            continue
        func = getattr(term, "func", None)
        mode = getattr(term, "mode", None)
        interval_range_s = getattr(term, "interval_range_s", None)
        fname = getattr(func, "__name__", str(func)) if func else "<None>"
        print(f"  {name:32s}  func={fname:38s}  mode={mode!s:8s} interval={interval_range_s}")
        params = getattr(term, "params", {}) or {}
        for k, v in params.items():
            short = repr(v)
            if len(short) > 80:
                short = short[:77] + "..."
            print(f"      {k:30s} = {short}")
        print()

    # ---------------- noise summary -------------
    print("=" * 90)
    print("ObservationsCfg.policy term noise")
    print("=" * 90)
    pol = cfg.observations.policy
    for term_name in pol.__dict__:
        t = getattr(pol, term_name)
        if not hasattr(t, "noise"):
            continue
        noise = getattr(t, "noise", None)
        scale = getattr(t, "scale", None)
        print(f"  {term_name:25s}  scale={scale}  noise={noise}")
    print(f"  enable_corruption = {pol.enable_corruption}    history_length = {pol.history_length}")

    # ---------------- make + reset env -------------
    env = gym.make(args.task, cfg=cfg)
    base = env.unwrapped
    robot = base.scene["robot"]
    rdata = robot.data

    print()
    print("=" * 90)
    print("Pre-reset (URDF nominal) values")
    print("=" * 90)
    # Save nominal stiffness/damping/default_joint_pos BEFORE any randomization.
    nom_stiff = []
    nom_damp = []
    for act in robot.actuators.values():
        nom_stiff.append((act.cfg.joint_names_expr, act.stiffness[0].cpu().numpy().copy()))
        nom_damp.append((act.cfg.joint_names_expr, act.damping[0].cpu().numpy().copy()))
    nom_default_q = rdata.default_joint_pos[0].cpu().numpy().copy()
    print(f"  default_joint_pos[0] = {nom_default_q}")
    print(f"  joint_names_actuator-order = {robot.joint_names}")

    # Drive a reset to trigger startup + reset events
    obs_dict, _ = env.reset(seed=args.seed)

    print()
    print("=" * 90)
    print(f"Post-reset per-env values (num_envs={args.num_envs})")
    print("=" * 90)

    # P1 — actuator gains
    print("[P1] actuator stiffness/damping per env (scale relative to nominal):")
    for act_name, actuator in robot.actuators.items():
        kp = actuator.stiffness.cpu().numpy()
        kd = actuator.damping.cpu().numpy()
        # ratio per env per joint
        kp0 = kp[0].copy()
        # find any nonzero col to compute ratio safely
        ratios_kp = np.zeros_like(kp)
        ratios_kd = np.zeros_like(kd)
        # Use env 0's first-reset stiffness as denominator (post-randomize),
        # then compare across envs. We instead use the original
        # default_joint_stiffness from rdata which is the URDF/g0 nominal.
        nom_kp = rdata.default_joint_stiffness.cpu().numpy()[0]
        nom_kd = rdata.default_joint_damping.cpu().numpy()[0]
        for e in range(args.num_envs):
            for j_in_act, j_idx in enumerate(actuator.joint_indices):
                nk = nom_kp[j_idx]
                nd = nom_kd[j_idx]
                ratios_kp[e, j_in_act] = kp[e, j_in_act] / nk if nk != 0 else 0
                ratios_kd[e, j_in_act] = kd[e, j_in_act] / nd if nd != 0 else 0
        print(f"  actuator '{act_name}'  joint_indices={list(actuator.joint_indices)}")
        print(f"    kp_scale per env  : min={ratios_kp.min():.3f}  max={ratios_kp.max():.3f}  "
              f"mean={ratios_kp.mean():.3f}   shape={ratios_kp.shape}")
        print(f"    kd_scale per env  : min={ratios_kd.min():.3f}  max={ratios_kd.max():.3f}  "
              f"mean={ratios_kd.mean():.3f}")
    print()

    # P2 — default_joint_pos shift per env
    print("[P2] default_joint_pos shift relative to nominal:")
    dq = rdata.default_joint_pos.cpu().numpy()
    nom = getattr(rdata, "default_joint_pos_nominal", None)
    if nom is None:
        print("  (default_joint_pos_nominal not set — P2 may not have fired)")
    else:
        nom_np = nom.cpu().numpy() if hasattr(nom, "cpu") else np.array(nom)
        for e in range(args.num_envs):
            diff = dq[e] - nom_np
            print(f"  env {e}: max|Δ|={np.max(np.abs(diff)):.4f}  mean|Δ|={np.mean(np.abs(diff)):.4f}  "
                  f"sample(first 4 jts)={diff[:4]}")
    print()

    # P4 — base mass + yaw
    print("[P4] base mass (torso_link) and root yaw per env:")
    body_id = robot.body_names.index("torso_link")
    masses = robot.root_physx_view.get_masses().cpu().numpy()[:, body_id]
    print(f"  torso mass per env: {masses}")
    root_quat = rdata.root_quat_w.cpu().numpy()  # (E, 4) wxyz
    # yaw from quat
    def quat_yaw(q):
        w, x, y, z = q
        return float(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    yaws = np.array([quat_yaw(root_quat[e]) for e in range(args.num_envs)])
    print(f"  root yaw per env (rad) : {yaws}")
    print(f"  root xy per env       :\n{rdata.root_pos_w[:, :2].cpu().numpy()}")
    print()

    # Other DR (physics_material, push_robot) — printed via cfg block above.

    print("=" * 90)
    print("Dry-run completed successfully.")
    print("=" * 90)
    env.close()
    sim_app.close()


if __name__ == "__main__":
    main()
