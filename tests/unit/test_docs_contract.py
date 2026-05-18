from __future__ import annotations

import pytest

from tests.conftest import APPROVED_SPEC, REPO_ROOT


pytestmark = pytest.mark.unit


def test_approved_design_spec_exists_at_actual_path():
    assert APPROVED_SPEC.is_file()


def test_docs_reference_policy_and_action_contracts():
    docs = [
        REPO_ROOT / "docs" / "observation_action_interface.md",
        REPO_ROOT / "docs" / "g0_actuator_parameters.md",
        REPO_ROOT / "docs" / "sim2sim_isaaclab_to_mujoco.md",
    ]
    joined = "\n".join(path.read_text(encoding="utf-8") for path in docs)
    for required in [
        "G0_JOINT_SDK_NAMES",
        "preserve_order=True",
        "0.12",
        "target_joint_pos = default_joint_pos + action_scale * policy_action",
        "output_speed = standard_speed * 6 / 7",
        "output_torque = standard_torque * 7 / 6",
    ]:
        assert required in joined


def test_implementation_plan_command_tiers_are_documented():
    plan_path = REPO_ROOT / "docs" / "superpowers" / "specs" / "2026-05-18-pre-deployment-validation-implementation-plan.md"
    if not plan_path.exists():
        pytest.skip("Implementation plan has not been created in this checkout.")
    plan = plan_path.read_text(encoding="utf-8")
    expected_commands = [
        'python -m pytest tests/unit -m "unit"',
        'python -m pytest tests/deployment -m "deployment_dryrun and hardware_forbidden"',
        '/home/lz/IsaacLab/isaaclab.sh -p -m pytest tests/isaaclab -m "isaaclab"',
        '/home/lz/IsaacLab/isaaclab.sh -p -m pytest tests -m "release_gate"',
    ]
    for command in expected_commands:
        assert command in plan
