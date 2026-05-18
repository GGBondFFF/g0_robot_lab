from __future__ import annotations

import numpy as np

from scripts.sim2sim import g0_sim2sim_config as cfg


def test_default_pose_array_matches_policy_joint_order() -> None:
    default = cfg.get_default_joint_pos_array()
    by_name = dict(zip(cfg.get_joint_names(), default, strict=True))
    assert default.shape == (cfg.get_action_dim(),)
    assert np.isclose(by_name["l_hip_pitch_joint"], -0.20)
    assert np.isclose(by_name["r_hip_pitch_joint"], 0.20)
    assert np.isclose(by_name["l_knee_pitch_joint"], -0.34)
    assert np.isclose(by_name["r_knee_pitch_joint"], 0.34)
    assert np.isclose(by_name["l_elbow_pitch_joint"], 0.97)
    assert np.isclose(by_name["r_elbow_pitch_joint"], -0.97)


def test_zero_action_target_is_default_pose() -> None:
    np.testing.assert_allclose(
        cfg.compute_target_joint_pos(np.zeros(cfg.get_action_dim())),
        cfg.get_default_joint_pos_array(),
    )
