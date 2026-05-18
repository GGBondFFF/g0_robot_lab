from __future__ import annotations

import numpy as np

from scripts.sim2sim import g0_sim2sim_config as cfg
from scripts.sim2sim.g0_mujoco_interface import G0MuJoCoInterface


def test_observation_term_dims_sum_to_frame_width() -> None:
    assert sum(cfg.POLICY_OBS_TERM_DIMS.values()) == cfg.get_single_frame_observation_dim()
    assert set(cfg.POLICY_OBS_TERM_DIMS) == set(cfg.POLICY_OBS_TERMS)


def test_single_frame_observation_can_be_split_by_term() -> None:
    frame = G0MuJoCoInterface.build_observation_frame(
        joint_pos=cfg.get_default_joint_pos_array(),
        joint_vel=np.zeros(cfg.get_action_dim()),
        last_action=np.zeros(cfg.get_action_dim()),
        command=np.zeros(3),
        sim_time=0.0,
    )
    terms = cfg.split_policy_observation(frame)
    assert terms["base_ang_vel"].shape == (1, 3)
    assert terms["joint_pos_rel"].shape == (1, cfg.get_action_dim())
    assert terms["last_action"].shape == (1, cfg.get_action_dim())
    assert terms["gait_phase"].shape == (1, 2)
