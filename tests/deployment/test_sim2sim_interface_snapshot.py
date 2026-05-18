from __future__ import annotations

import pytest

from tests.conftest import REPO_ROOT


pytestmark = [pytest.mark.deployment_dryrun, pytest.mark.hardware_forbidden]


def test_sim2sim_snapshot_is_documentation_only():
    assert not (REPO_ROOT / "scripts" / "sim2sim").exists()
    assert not (REPO_ROOT / "mujoco").exists()


def test_sim2sim_docs_keep_interface_anchors():
    text = (REPO_ROOT / "docs" / "sim2sim_isaaclab_to_mujoco.md").read_text(encoding="utf-8")
    required = [
        "G0-Velocity-v0",
        "G0_JOINT_SDK_NAMES",
        "0.12",
        "target_joint_pos = default_joint_pos + action_scale * policy_action",
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos_rel",
        "joint_vel_rel",
        "last_action",
        "gait_phase",
        "sim.dt = 0.005",
        "decimation = 4",
        "control dt = 0.02",
    ]
    for item in required:
        assert item in text
