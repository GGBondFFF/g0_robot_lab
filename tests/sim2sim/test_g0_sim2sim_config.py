from __future__ import annotations

import numpy as np

from scripts.sim2sim import g0_sim2sim_config as cfg


def test_g0_interface_constants() -> None:
    joint_names = cfg.get_joint_names()
    assert cfg.get_action_dim() == 22
    assert len(joint_names) == 22
    assert cfg.get_default_joint_pos_array().shape == (22,)
    assert np.isclose(cfg.ACTION_SCALE, 0.12)
    assert np.isclose(cfg.CONTROL_DT, 0.02)
    assert joint_names[0] == "l_hip_pitch_joint"
    assert joint_names[-1] == "r_elbow_pitch_joint"


def test_policy_observation_dims() -> None:
    assert cfg.get_single_frame_observation_dim() == 77
    assert cfg.get_policy_observation_dim() == 77 * cfg.POLICY_HISTORY_LENGTH

