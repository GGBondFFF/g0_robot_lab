"""Optional `suspended_stand` keyframe for the staging SOP (Task 6).

Verifies the patched MJCF exposes a `suspended_stand` keyframe with the base
raised into [0.45, 0.55] m (same joint pose as `default_stand`) and that the
elastic band can hold the robot suspended there under a passive (tau=0) sim.

This keyframe is opt-in: deploy_staging.yaml stays on `default_stand` by
default; set `keyframe: suspended_stand` to demonstrate the full band-descend
sequence.

    pytest tests/deployment/test_g0_suspended_stand_keyframe.py -v
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

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

MJCF = os.path.join(
    REPO_ROOT, "source", "g0_robot_lab", "g0_robot_lab", "assets", "robots",
    "g0", "mujoco", "model_patched.xml",
)


def _keyframe_qpos(name):
    root = ET.parse(MJCF).getroot()
    kf = root.find("keyframe")
    assert kf is not None, "MJCF has no <keyframe>"
    for key in kf.findall("key"):
        if key.attrib.get("name") == name:
            return np.array([float(x) for x in key.attrib["qpos"].split()])
    raise AssertionError(f"keyframe {name!r} not found in {MJCF}")


def test_suspended_stand_root_z_in_range():
    q = _keyframe_qpos("suspended_stand")
    root_z = q[2]
    assert 0.45 <= root_z <= 0.55, f"suspended_stand root_z={root_z} not in [0.45, 0.55]"


def test_suspended_stand_joints_match_default_stand():
    sus = _keyframe_qpos("suspended_stand")
    deflt = _keyframe_qpos("default_stand")
    # Joints (everything past the 7-dof freejoint root) must match default_stand.
    np.testing.assert_allclose(sus[7:], deflt[7:], atol=1e-8)


def test_band_holds_robot_at_suspended_stand():
    from deploy.backends.mujoco_backend import MujocoBackend
    from deploy.common.elastic_band import ElasticBand

    backend = MujocoBackend(
        mjcf_path=MJCF, sim_dt=0.002, keyframe="suspended_stand",
        joint_names_mj=G0_JOINT_NAMES_MJ,
    )
    band = ElasticBand(
        anchor=(0.0, 0.0, 2.0), stiffness=50.0, damping=10.0,
        rest_length=0.0, enabled=True, one_sided=True, mode="rope",
    )
    band.calibrate(backend)  # snap slack at the suspended pose

    z0 = float(backend.read_state()["base_pos_w"][2])
    assert z0 >= 0.45, f"sim did not start at suspended pose (z0={z0})"

    tau0 = np.zeros(backend.model.nu, dtype=np.float64)
    # ~2 s of passive (tau=0) sim. With the default staging band params
    # (one-sided, k=50, calibrated slack at the suspended pose) the band
    # catches the falling robot and settles it near standing height
    # (default_stand base = 0.23); it must NOT collapse to the floor. Holding
    # the robot fully at 0.5 would need a higher anchor / stiffness — see the
    # suspend-tuning note in docs/sim2sim/g0_unitree_staging_sop_en.md.
    z_min = z0
    for _ in range(int(2.0 / 0.02)):
        band.update(backend)
        backend.apply_torque(tau0, n_substeps=10)
        z = float(backend.read_state()["base_pos_w"][2])
        z_min = min(z_min, z)
        assert np.all(np.isfinite(backend.data.qpos)), "non-finite qpos"
    assert z_min > 0.18, (
        f"band failed to hold suspended robot off the floor: "
        f"base_z_min={z_min:.3f}")
