"""FSM ground-gate test for the Unitree-aligned staging SOP.

Boots the same MuJoCo+FSM+ONNX pipeline as deploy/robots/g0/main.py but with
deploy_staging.yaml + a StagingContext, then proves:

    after FixStand ramp completes:
      press r            -> REJECTED (feet not confirmed on ground)
      press g, press r   -> rl_base

Run from repo root with the g0_mujoco conda env:

    pytest tests/deployment/test_g0_staging_fsm_gate.py -v
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

# Match deploy/robots/g0/main.py G0_JOINT_NAMES_MJ.
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
    """Mirror main.py's wiring with a StagingContext gating rl_base."""
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


def test_rl_base_gated_until_feet_on_ground(staging_cfg):
    from deploy.robots.g0.main import tick
    backend, fsm, band, cmd_mgr, rc, staging, step_dt = _build_staging_pipeline(
        staging_cfg)
    step_m = staging_cfg["elastic_band"]["length_step_m"]

    def _tick():
        tick(cmd_mgr, rc, band, backend, fsm,
             length_step_m=step_m, staging=staging)

    # Enter FixStand and run past the ramp.
    rc.on_key(staging_cfg["keys"]["fix_stand"])
    for _ in range(int(2.0 / step_dt)):
        _tick()
    assert fsm.current.name == "fix_stand"
    assert staging.feet_on_ground is False

    # Request RLBase WITHOUT confirming ground -> must be rejected.
    rc.on_key(staging_cfg["keys"]["rl_base"])
    _tick()
    assert fsm.current.name == "fix_stand", (
        "rl_base must be gated until feet_on_ground is confirmed")

    # Confirm ground via the 'g' key, then request RLBase -> allowed.
    rc.on_key(staging_cfg["keys"]["confirm_ground"])
    _tick()
    assert staging.feet_on_ground is True
    rc.on_key(staging_cfg["keys"]["rl_base"])
    _tick()
    assert fsm.current.name == "rl_base", (
        "rl_base should be allowed once ground is confirmed and ramp is done")
