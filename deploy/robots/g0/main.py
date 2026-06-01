"""G0 deploy entry — Python, MuJoCo-backed.

Pipeline:
    deploy.yaml -> MujocoBackend -> ManagerBasedRLEnv -> CtrlFSM(Passive|FixStand|RLBase)
    keyboard (via viewer key_callback) -> RemoteController -> CtrlFSM

Usage:
    python -m deploy.robots.g0.main \
        --config deploy/robots/g0/config/policy/velocity/v0/deploy.yaml

Keys (defaults from yaml):
    p  -> Passive          (always allowed)
    f  -> FixStand         (from Passive only; via Passive from RLBase)
    r  -> RLBase           (from FixStand, after ramp completes)
    0  -> zero velocity command
    w/s -> vx +/-     a/d -> vy +/-     q/e -> wz +/-
"""

import argparse
import os
import sys
import time

import numpy as np
import yaml

# Make sibling 'deploy' package importable when invoked as a script.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEPLOY_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))  # repo root
if _DEPLOY_ROOT not in sys.path:
    sys.path.insert(0, _DEPLOY_ROOT)

from deploy.backends.mujoco_backend import MujocoBackend
from deploy.common.elastic_band import ElasticBand
from deploy.common.remote_controller import RemoteController
from deploy.isaaclab.algorithms.ort_runner import OrtRunner
from deploy.isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from deploy.isaaclab.managers.action_manager import JointPositionAction
from deploy.isaaclab.managers.command_manager import VelocityCommandManager
from deploy.isaaclab.managers.observation_manager import ObservationManager


# MJ actuator order is fixed by model.xml; we cross-check at boot.
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


def _build_sdk_to_mj(joint_names_sdk):
    mj_index = {n: i for i, n in enumerate(G0_JOINT_NAMES_MJ)}
    return np.array([mj_index[n] for n in joint_names_sdk], dtype=np.int64)


def _remap_sdk_to_mj(vec_sdk, sdk_to_mj):
    out = np.empty_like(vec_sdk)
    out[sdk_to_mj] = vec_sdk
    return out


def _resolve_path(p, repo_root):
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) else os.path.join(repo_root, p)


def tick(cmd_mgr, rc, band, backend, fsm):
    """One policy step. Public so tests/soaks reuse the exact production path.

    Order matters:
        1. Pull latest keyboard cmd into the command manager.
        2. Refresh the elastic band's xfrc_applied for THIS step. MuJoCo
           does not clear xfrc_applied between mj_step calls, but the
           force is a function of (base_pos, base_lin_vel) so we must
           recompute each policy tick or the support force goes stale —
           this was an actual bug in earlier viewer-mode runs.
        3. Run the FSM (which calls env.step_*, which calls mj_step xN).
    """
    cmd_mgr.set(*rc.get_cmd())
    band.update(backend)
    fsm.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--repo-root", default=None,
                    help="Repo root for resolving relative paths in yaml.")
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--realtime", action="store_true")
    ap.add_argument("--initial-state", default="fix_stand",
                    choices=("passive", "fix_stand", "rl_base"))
    ap.add_argument("--elastic-band", choices=("on", "off"), default=None,
                    help="Override yaml elastic_band.enabled.")
    args = ap.parse_args()

    repo_root = args.repo_root or os.path.abspath(
        os.path.join(_THIS_DIR, "..", "..", "..")
    )

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    mjcf_path = _resolve_path(cfg["mjcf_path"], repo_root)
    onnx_path = _resolve_path(cfg["policy"]["onnx_path"], repo_root)

    joint_names_sdk = cfg["joint_names_sdk"]
    joint_names_isaac = cfg["joint_names_isaac"]

    # ---- backend
    backend = MujocoBackend(
        mjcf_path=mjcf_path,
        sim_dt=cfg.get("sim_dt"),
        keyframe=cfg.get("keyframe"),
        joint_names_mj=G0_JOINT_NAMES_MJ,
    )

    # ---- defaults & gains, all in SDK / MJ order
    default_q_sdk = np.array(
        [cfg["default_joint_pos"][n] for n in joint_names_sdk], dtype=np.float64
    )
    kp_sdk = np.array([cfg["stiffness"][n] for n in joint_names_sdk], dtype=np.float64)
    kd_sdk = np.array([cfg["damping"][n] for n in joint_names_sdk], dtype=np.float64)
    sdk_to_mj = _build_sdk_to_mj(joint_names_sdk)
    default_q_mj = _remap_sdk_to_mj(default_q_sdk, sdk_to_mj)
    kp_mj = _remap_sdk_to_mj(kp_sdk, sdk_to_mj)
    kd_mj = _remap_sdk_to_mj(kd_sdk, sdk_to_mj)

    # default_q in Isaac order for jpos_rel
    isaac_in_mj = {n: i for i, n in enumerate(G0_JOINT_NAMES_MJ)}
    isaac_to_mj = np.array([isaac_in_mj[n] for n in joint_names_isaac], dtype=np.int64)
    default_q_isaac = default_q_mj[isaac_to_mj]

    # ---- managers
    cmd_mgr = VelocityCommandManager(
        default=cfg["commands"]["base_velocity"]["default"],
        ranges={
            k: tuple(v) for k, v in cfg["commands"]["base_velocity"]["ranges"].items()
        },
    )
    act_mgr = JointPositionAction(
        default_q_sdk=default_q_sdk,
        scale=cfg["actions"]["joint_pos"]["scale"],
        sdk_to_mj=sdk_to_mj, kp_mj=kp_mj, kd_mj=kd_mj,
    )
    obs_mgr = ObservationManager(
        cfg=cfg["observations"],
        joint_names_sdk=joint_names_sdk,
        joint_names_isaac=joint_names_isaac,
        joint_names_mj=G0_JOINT_NAMES_MJ,
        default_q_isaac=default_q_isaac,
    )
    env = ManagerBasedRLEnv(
        backend=backend,
        command_manager=cmd_mgr,
        action_manager=act_mgr,
        observation_manager=obs_mgr,
        decimation=int(cfg["decimation"]),
    )
    env.reset()

    # ---- policy
    ort = OrtRunner(onnx_path, expected_input_dim=obs_mgr.obs_dim_flat)

    # ---- elastic band
    eb_cfg = cfg.get("elastic_band", {}) or {}
    eb_enabled = eb_cfg.get("enabled", False)
    if args.elastic_band is not None:
        eb_enabled = (args.elastic_band == "on")
    band = ElasticBand(
        anchor=eb_cfg.get("anchor", (0.0, 0.0, 2.0)),
        stiffness=eb_cfg.get("stiffness", 50.0),
        damping=eb_cfg.get("damping", 10.0),
        rest_length=eb_cfg.get("rest_length", 0.0),
        enabled=eb_enabled,
        one_sided=eb_cfg.get("one_sided", True),
        mode=eb_cfg.get("mode", "rope"),
    )
    print(f"  elastic_band : enabled={band.enabled}  mode={band.mode}  "
          f"one_sided={band.one_sided}  anchor={band.anchor.tolist()}  "
          f"k={band.k}  c={band.c}  L0={band.L}")

    # ---- RC + FSM
    rc = RemoteController(cfg["keys"])

    from deploy.fsm.state_passive import StatePassive
    from deploy.fsm.state_fix_stand import StateFixStand
    from deploy.fsm.state_rl_base import StateRLBase
    from deploy.fsm.ctrl_fsm import CtrlFSM

    states = {
        "passive": StatePassive(env, rc=rc),
        "fix_stand": StateFixStand(
            env, rc=rc, default_q_mj=default_q_mj,
            ramp_time_s=cfg["fix_stand"]["ramp_time_s"],
            step_dt=cfg["step_dt"],
            band=band,
        ),
        "rl_base": StateRLBase(env, ort_runner=ort, rc=rc),
    }
    fsm = CtrlFSM(states, initial=args.initial_state, rc=rc)

    print("=" * 90)
    print("G0 deploy (Python/MuJoCo)  policy = {}".format(onnx_path))
    print("=" * 90)
    print("Keys: p=Passive  f=FixStand  r=RLBase  0=zero cmd")
    print("      w/s=vx+/-  a/d=vy+/-  q/e=wz+/-")
    print()

    step_dt = float(cfg["step_dt"])

    if args.no_viewer:
        n_steps = int(args.duration / step_dt)
        t0 = time.time()
        for i in range(n_steps):
            tick(cmd_mgr, rc, band, backend, fsm)
            if (i % 50) == 0:
                s = backend.read_state()
                print(f"  t={s['sim_time']:6.2f}  state={fsm.current.name:10s}  cmd={rc.get_cmd()}")
        print(f"[done] wall_s={time.time() - t0:.2f}")
    else:
        import mujoco
        import mujoco.viewer

        def key_callback(keycode):
            try:
                ch = chr(keycode).lower()
            except ValueError:
                return
            rc.on_key(ch)

        with mujoco.viewer.launch_passive(
            backend.model, backend.data, key_callback=key_callback
        ) as viewer:
            t0 = time.time()
            sim_t0 = backend.data.time
            while viewer.is_running() and (backend.data.time - sim_t0) < args.duration:
                tick(cmd_mgr, rc, band, backend, fsm)
                viewer.sync()
                if args.realtime:
                    wall = time.time() - t0
                    sim_elapsed = backend.data.time - sim_t0
                    delta = sim_elapsed - wall
                    if delta > 0:
                        time.sleep(delta)
            print(f"[done] sim_s={backend.data.time - sim_t0:.2f}  "
                  f"wall_s={time.time() - t0:.2f}")
        # mujoco.viewer + onnxruntime can race during interpreter teardown.
        # Force a clean exit before GC kicks in.
        os._exit(0)


if __name__ == "__main__":
    main()
