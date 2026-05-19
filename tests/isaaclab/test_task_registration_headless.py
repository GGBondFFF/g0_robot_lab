from __future__ import annotations

import pytest


pytestmark = pytest.mark.isaaclab


def test_g0_velocity_task_is_registered_headless(isaac_sim_app):
    import gymnasium as gym
    import g0_robot_lab.tasks  # noqa: F401

    assert "G0-Velocity-v0" in gym.registry
