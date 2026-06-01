"""Headless automation of the Unitree-aligned G0 staging SOP (Task 7).

Drives the same MuJoCo+FSM+ONNX pipeline as deploy/robots/g0/main.py with
deploy_staging.yaml, scripted through the operator sequence:

    f  -> FixStand (ramp + band calibrate)
    8  -> loosen band x8 (feet settle on ground)
    g  -> confirm ground
    r  -> RLBase (gated until ground confirmed)
    9  -> disable band

Acceptance levels exercised here:
    * L1 (process)   -> test_staging_sop_process  [MUST PASS]
    * L3 (no band)   -> test_staging_sop_rlbase_30s_no_band  [xfail until Task 10]

L1 is deterministic process logic. L3 depends on the v0 policy's MuJoCo
standing, which is a known limitation (g0_onnx_closed_loop_gui_test_zh.md §8),
so it is marked xfail(strict=False) until Phase 5 / Task 10 fixes it.

    pytest tests/deployment/test_g0_staging_sop.py -v
"""

import os
import sys

import numpy as np
import pytest
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from deploy.common.staging_context import StagingContext  # noqa: E402

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

STAGING_YAML = os.path.join(
    REPO_ROOT, "deploy", "robots", "g0", "config", "policy", "velocity", "v0",
    "deploy_staging.yaml",
)


@pytest.fixture(scope="module")
def staging_cfg():
    with open(STAGING_YAML, "r") as f:
        return yaml.safe_load(f)


def _resolve(p):
    p = os.path.expanduser(p)
    return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)


def _build_staging_pipeline(cfg):
    from deploy.backends.mujoco_backend import MujocoBackend
    from deploy.common.elastic_band import ElasticBand
    from deploy.common.remote_controller import RemoteController
    from deploy.isaaclab.algorithms.ort_runner import OrtRunner
    from deploy.isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
    from deploy.isaaclab.managers.action_manager import JointPositionAction
    from deploy.isaaclab.managers.command_manager import VelocityCommandManager
    from deploy.isaaclab.managers.observation_manager import ObservationManager
    from deploy.fsm.ctrl_fsm import CtrlFSM
    from deploy.fsm.state_passive import StatePassive
    from deploy.fsm.state_fix_stand import StateFixStand
    from deploy.fsm.state_rl_base import StateRLBase

    mjcf_path = _resolve(cfg["mjcf_path"])
    onnx_path = _resolve(cfg["policy"]["onnx_path"])
    joint_names_sdk = cfg["joint_names_sdk"]
    joint_names_isaac = cfg["joint_names_isaac"]

    backend = MujocoBackend(
        mjcf_path=mjcf_path, sim_dt=cfg.get("sim_dt"),
        keyframe=cfg.get("keyframe"), joint_names_mj=G0_JOINT_NAMES_MJ,
    )

    default_q_sdk = np.array(
        [cfg["default_joint_pos"][n] for n in joint_names_sdk], dtype=np.float64)
    kp_sdk = np.array([cfg["stiffness"][n] for n in joint_names_sdk], dtype=np.float64)
    kd_sdk = np.array([cfg["damping"][n] for n in joint_names_sdk], dtype=np.float64)

    mj_index = {n: i for i, n in enumerate(G0_JOINT_NAMES_MJ)}
    sdk_to_mj = np.array([mj_index[n] for n in joint_names_sdk], dtype=np.int64)
    isaac_to_mj = np.array([mj_index[n] for n in joint_names_isaac], dtype=np.int64)

    def _remap(vec, idx):
        out = np.empty_like(vec)
        out[idx] = vec
        return out

    default_q_mj = _remap(default_q_sdk, sdk_to_mj)
    kp_mj = _remap(kp_sdk, sdk_to_mj)
    kd_mj = _remap(kd_sdk, sdk_to_mj)
    default_q_isaac = default_q_mj[isaac_to_mj]

    cmd_mgr = VelocityCommandManager(
        default=cfg["commands"]["base_velocity"]["default"],
        ranges={k: tuple(v)
                for k, v in cfg["commands"]["base_velocity"]["ranges"].items()},
    )
    act_mgr = JointPositionAction(
        default_q_sdk=default_q_sdk, scale=cfg["actions"]["joint_pos"]["scale"],
        sdk_to_mj=sdk_to_mj, kp_mj=kp_mj, kd_mj=kd_mj,
    )
    obs_mgr = ObservationManager(
        cfg=cfg["observations"], joint_names_sdk=joint_names_sdk,
        joint_names_isaac=joint_names_isaac, joint_names_mj=G0_JOINT_NAMES_MJ,
        default_q_isaac=default_q_isaac,
    )
    env = ManagerBasedRLEnv(
        backend=backend, command_manager=cmd_mgr, action_manager=act_mgr,
        observation_manager=obs_mgr, decimation=int(cfg["decimation"]),
    )
    env.reset()

    ort = OrtRunner(onnx_path, expected_input_dim=obs_mgr.obs_dim_flat)

    eb_cfg = cfg.get("elastic_band", {}) or {}
    band = ElasticBand(
        anchor=eb_cfg.get("anchor", (0.0, 0.0, 2.0)),
        stiffness=eb_cfg.get("stiffness", 50.0),
        damping=eb_cfg.get("damping", 10.0),
        rest_length=eb_cfg.get("rest_length", 0.0),
        enabled=True, one_sided=eb_cfg.get("one_sided", True),
        mode=eb_cfg.get("mode", "rope"),
    )
    staging = StagingContext(cfg.get("staging"))

    rc = RemoteController(cfg["keys"])
    states = {
        "passive": StatePassive(env, rc=rc),
        "fix_stand": StateFixStand(
            env, rc=rc, default_q_mj=default_q_mj,
            ramp_time_s=cfg["fix_stand"]["ramp_time_s"],
            step_dt=cfg["step_dt"], band=band, staging=staging,
        ),
        "rl_base": StateRLBase(env, ort_runner=ort, rc=rc),
    }
    fsm = CtrlFSM(states, initial="passive", rc=rc)
    return backend, fsm, band, cmd_mgr, rc, staging, float(cfg["step_dt"])


def _run_sop_through_band_disable(cfg):
    """Execute the SOP from f -> ... -> 9 (band off). Returns the live pipeline
    plus a few measured signals for assertions. Shared by L1 and L3 tests."""
    from deploy.robots.g0.main import tick
    backend, fsm, band, cmd_mgr, rc, staging, step_dt = _build_staging_pipeline(cfg)
    step_m = cfg["elastic_band"]["length_step_m"]
    keys = cfg["keys"]

    signals = {"band_force_max_fix_stand": 0.0, "all_finite": True}

    def _tick():
        tick(cmd_mgr, rc, band, backend, fsm, length_step_m=step_m, staging=staging)
        if not (np.all(np.isfinite(backend.data.qpos))
                and np.all(np.isfinite(backend.data.qvel))
                and np.all(np.isfinite(backend.data.ctrl))):
            signals["all_finite"] = False

    # 1) f -> FixStand, run past the ramp.
    rc.on_key(keys["fix_stand"])
    for _ in range(int(2.5 / step_dt)):
        _tick()
        if fsm.current.name == "fix_stand":
            f = float(np.linalg.norm(
                backend.data.xfrc_applied[backend.base_body_id, :3]))
            signals["band_force_max_fix_stand"] = max(
                signals["band_force_max_fix_stand"], f)
    assert fsm.current.name == "fix_stand"

    L_before = band.L
    # 3) press 8 x8 -> loosen band; feet settle onto the ground.
    for _ in range(8):
        rc.on_key(keys["band_loosen"])
        _tick()
    signals["L_after_loosen"] = band.L
    signals["L_before_loosen"] = L_before
    signals["base_z_after_loosen"] = float(backend.data.qpos[2])

    # 4) g -> confirm ground.
    rc.on_key(keys["confirm_ground"])
    _tick()
    signals["feet_on_ground"] = staging.feet_on_ground

    # 5) r -> RLBase (gated until ground). Drive a few seconds with band on.
    rc.on_key(keys["rl_base"])
    for _ in range(int(3.0 / step_dt)):
        _tick()
    signals["state_after_r"] = fsm.current.name

    # 6) 9 -> disable band.
    rc.on_key(keys["band_toggle"])
    _tick()
    signals["band_enabled_after_toggle"] = band.enabled

    return backend, fsm, band, cmd_mgr, rc, staging, step_dt, signals, _tick


def test_staging_sop_process(staging_cfg):
    """L1 — process logic must hold deterministically (no policy dependence)."""
    (backend, fsm, band, cmd_mgr, rc, staging, step_dt,
     sig, _tick) = _run_sop_through_band_disable(staging_cfg)

    # Band was actually loading the base while suspended in FixStand.
    assert sig["band_force_max_fix_stand"] > 0.1, (
        f"band never produced force in FixStand: {sig['band_force_max_fix_stand']}")
    # Key 8 loosened the band by ~8 * length_step_m.
    step_m = staging_cfg["elastic_band"]["length_step_m"]
    assert sig["L_after_loosen"] == pytest.approx(
        sig["L_before_loosen"] + 8 * step_m, abs=1e-6)
    # Feet are near the ground after lowering.
    assert sig["base_z_after_loosen"] < staging_cfg["staging"]["ground_base_z_max"]
    # Ground confirm + gate let RLBase through.
    assert sig["feet_on_ground"] is True
    assert sig["state_after_r"] == "rl_base", (
        "RLBase not reached after ground confirm")
    # Band disabled by key 9.
    assert sig["band_enabled_after_toggle"] is False
    # Whole band-on staging run stayed finite.
    assert sig["all_finite"] is True


@pytest.mark.xfail(strict=False,
                   reason="v0 policy mistracks standing in MuJoCo without the "
                          "band (~1.4s fall, see g0_onnx_closed_loop_gui_test_zh "
                          "§8); fixed in Phase 5 / Task 10")
def test_staging_sop_rlbase_30s_no_band(staging_cfg):
    """L3 — after the band is disabled, RLBase must keep standing >=30 s."""
    (backend, fsm, band, cmd_mgr, rc, staging, step_dt,
     sig, _tick) = _run_sop_through_band_disable(staging_cfg)
    assert band.enabled is False
    assert fsm.current.name == "rl_base"

    z_min = float(backend.data.qpos[2])
    for _ in range(int(30.0 / step_dt)):
        _tick()
        z_min = min(z_min, float(backend.data.qpos[2]))
    assert fsm.current.name == "rl_base"
    assert z_min > 0.15, f"robot fell without band: root_z_min={z_min:.4f}"
