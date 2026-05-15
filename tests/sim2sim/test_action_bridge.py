from __future__ import annotations

import numpy as np
import pytest

from scripts.sim2sim import g0_sim2sim_config as cfg


def test_zero_action_returns_default_joint_pos() -> None:
    action = np.zeros(cfg.get_action_dim())
    np.testing.assert_allclose(cfg.compute_target_joint_pos(action), cfg.get_default_joint_pos_array())


def test_positive_action_adds_action_scale() -> None:
    action = np.ones(cfg.get_action_dim())
    np.testing.assert_allclose(
        cfg.compute_target_joint_pos(action),
        cfg.get_default_joint_pos_array() + cfg.ACTION_SCALE,
    )


def test_negative_action_subtracts_action_scale() -> None:
    action = -np.ones(cfg.get_action_dim())
    np.testing.assert_allclose(
        cfg.compute_target_joint_pos(action),
        cfg.get_default_joint_pos_array() - cfg.ACTION_SCALE,
    )


def test_action_is_clipped() -> None:
    action = np.full(cfg.get_action_dim(), 10.0)
    np.testing.assert_allclose(
        cfg.compute_target_joint_pos(action),
        cfg.get_default_joint_pos_array() + cfg.ACTION_SCALE,
    )


def test_invalid_action_shape_raises() -> None:
    with pytest.raises(ValueError):
        cfg.compute_target_joint_pos(np.zeros((1, cfg.get_action_dim())))

