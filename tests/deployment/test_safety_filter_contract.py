from __future__ import annotations

import math

import pytest

from scripts.validation._lowcmd_mapping import build_default_safety_limits
from scripts.validation._safety_filter import SafetyFilter, SafetyFilterError, SafetyState
from tests.conftest import G0_SOURCE
from tests.helpers.static_contract import literal_assignment, load_module_ast


pytestmark = [pytest.mark.deployment_dryrun, pytest.mark.hardware_forbidden]


def _contract_values():
    module = load_module_ast(G0_SOURCE)
    joint_order = literal_assignment(module, "G0_JOINT_SDK_NAMES")
    default_pose = literal_assignment(module, "G0_DEFAULT_JOINT_POS")
    return joint_order, default_pose


def _limits(joint_order):
    return build_default_safety_limits(
        joint_order,
        max_action_delta=0.5,
        max_target_delta=0.1,
        max_command_age_s=0.05,
    )


def test_zero_action_yields_default_targets(dryrun_required):
    joint_order, default_pose = _contract_values()
    result = SafetyFilter().apply(
        raw_action=[0.0] * 22,
        joint_order=joint_order,
        default_joint_pos=default_pose,
        action_scale=0.12,
        limits=_limits(joint_order),
        state=SafetyState(last_obs_time=1.0),
        now=1.0,
        emergency_stop=False,
        hardware_enabled=False,
    )
    for joint_name in joint_order:
        assert result.target_by_joint[joint_name] == pytest.approx(default_pose[joint_name])
        assert result.clipped_action_by_joint[joint_name] == 0.0


def test_negative_one_action_shifts_target_by_scale(dryrun_required):
    joint_order, default_pose = _contract_values()
    result = SafetyFilter().apply(
        raw_action=[-1.0] + [0.0] * 21,
        joint_order=joint_order,
        default_joint_pos=default_pose,
        action_scale=0.12,
        limits=_limits(joint_order),
        state=SafetyState(last_obs_time=1.0),
        now=1.0,
        emergency_stop=False,
        hardware_enabled=False,
    )
    first_joint = joint_order[0]
    assert result.target_by_joint[first_joint] == pytest.approx(default_pose[first_joint] - 0.12)


def test_overflow_action_is_clipped_before_target_generation(dryrun_required):
    joint_order, default_pose = _contract_values()
    result = SafetyFilter().apply(
        raw_action=[10.0] + [0.0] * 21,
        joint_order=joint_order,
        default_joint_pos=default_pose,
        action_scale=0.12,
        limits=_limits(joint_order),
        state=SafetyState(last_obs_time=1.0),
        now=1.0,
        emergency_stop=False,
        hardware_enabled=False,
    )
    first_joint = joint_order[0]
    assert result.clipped_action_by_joint[first_joint] == 1.0
    assert result.target_by_joint[first_joint] == pytest.approx(default_pose[first_joint] + 0.12)
    assert result.target_by_joint[first_joint] != pytest.approx(default_pose[first_joint] + 1.2)


def test_nan_and_inf_actions_are_rejected(dryrun_required):
    joint_order, default_pose = _contract_values()
    limits = _limits(joint_order)
    state = SafetyState(last_obs_time=1.0)
    with pytest.raises(SafetyFilterError, match="non-finite"):
        SafetyFilter().apply(
            raw_action=[math.nan] + [0.0] * 21,
            joint_order=joint_order,
            default_joint_pos=default_pose,
            action_scale=0.12,
            limits=limits,
            state=state,
            now=1.0,
            emergency_stop=False,
            hardware_enabled=False,
        )
    with pytest.raises(SafetyFilterError, match="non-finite"):
        SafetyFilter().apply(
            raw_action=[math.inf] + [0.0] * 21,
            joint_order=joint_order,
            default_joint_pos=default_pose,
            action_scale=0.12,
            limits=limits,
            state=state,
            now=1.0,
            emergency_stop=False,
            hardware_enabled=False,
        )


def test_fast_action_jump_is_clamped_by_previous_action_and_target(dryrun_required):
    joint_order, default_pose = _contract_values()
    result = SafetyFilter().apply(
        raw_action=[1.0] * 22,
        joint_order=joint_order,
        default_joint_pos=default_pose,
        action_scale=0.12,
        limits=_limits(joint_order),
        state=SafetyState(
            prev_action={joint_name: 0.0 for joint_name in joint_order},
            prev_target={joint_name: float(default_pose[joint_name]) for joint_name in joint_order},
            last_obs_time=1.0,
        ),
        now=1.0,
        emergency_stop=False,
        hardware_enabled=False,
    )
    for joint_name in joint_order:
        assert result.clipped_action_by_joint[joint_name] == pytest.approx(0.5)
        assert abs(result.target_by_joint[joint_name] - default_pose[joint_name]) <= 0.1 + 1e-9


def test_stale_observation_is_rejected(dryrun_required):
    joint_order, default_pose = _contract_values()
    with pytest.raises(SafetyFilterError, match="Stale observation"):
        SafetyFilter().apply(
            raw_action=[0.0] * 22,
            joint_order=joint_order,
            default_joint_pos=default_pose,
            action_scale=0.12,
            limits=_limits(joint_order),
            state=SafetyState(last_obs_time=1.0),
            now=1.2,
            emergency_stop=False,
            hardware_enabled=False,
        )


def test_emergency_stop_returns_safe_hold_output(dryrun_required):
    joint_order, default_pose = _contract_values()
    limits = _limits(joint_order)
    prev_target = {joint_name: float(default_pose[joint_name]) + 0.03 for joint_name in joint_order}
    result = SafetyFilter().apply(
        raw_action=[1.0] * 22,
        joint_order=joint_order,
        default_joint_pos=default_pose,
        action_scale=0.12,
        limits=limits,
        state=SafetyState(
            prev_action={joint_name: 0.25 for joint_name in joint_order},
            prev_target=prev_target,
            last_obs_time=0.0,
        ),
        now=100.0,
        emergency_stop=True,
        hardware_enabled=False,
    )
    assert result.safe_mode == "emergency_stop_hold"
    for joint_name in joint_order:
        expected = max(limits.pos_lower[joint_name], min(limits.pos_upper[joint_name], prev_target[joint_name]))
        assert result.target_by_joint[joint_name] == pytest.approx(expected)


def test_hardware_enabled_is_rejected(dryrun_required):
    joint_order, default_pose = _contract_values()
    with pytest.raises(SafetyFilterError, match="hardware_enabled=True"):
        SafetyFilter().apply(
            raw_action=[0.0] * 22,
            joint_order=joint_order,
            default_joint_pos=default_pose,
            action_scale=0.12,
            limits=_limits(joint_order),
            state=SafetyState(last_obs_time=1.0),
            now=1.0,
            emergency_stop=False,
            hardware_enabled=True,
        )
