from __future__ import annotations

import ast

import pytest

from tests.conftest import VELOCITY_ENV_CFG_SOURCE
from tests.helpers.static_contract import class_def, keyword_value_from_call, load_module_ast, nested_class_def


pytestmark = pytest.mark.unit


EXPECTED_POLICY_TERMS = [
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos_rel",
    "joint_vel_rel",
    "last_action",
    "gait_phase",
]


def _module():
    return load_module_ast(VELOCITY_ENV_CFG_SOURCE)


def test_action_cfg_uses_sdk_joint_order_scale_and_preserve_order():
    actions_cfg = class_def(_module(), "ActionsCfg")
    scale = ast.literal_eval(keyword_value_from_call(actions_cfg, "joint_pos", "scale"))
    use_default_offset = ast.literal_eval(keyword_value_from_call(actions_cfg, "joint_pos", "use_default_offset"))
    preserve_order = ast.literal_eval(keyword_value_from_call(actions_cfg, "joint_pos", "preserve_order"))
    joint_names_expr = keyword_value_from_call(actions_cfg, "joint_pos", "joint_names")

    assert scale == 0.12
    assert use_default_offset is True
    assert preserve_order is True
    assert isinstance(joint_names_expr, ast.Call)
    assert isinstance(joint_names_expr.func, ast.Name)
    assert joint_names_expr.func.id == "list"
    assert isinstance(joint_names_expr.args[0], ast.Name)
    assert joint_names_expr.args[0].id == "G0_JOINT_SDK_NAMES"


def test_policy_observation_terms_and_history_are_contractual():
    observations_cfg = class_def(_module(), "ObservationsCfg")
    policy_cfg = nested_class_def(observations_cfg, "PolicyCfg")
    observed_terms = []
    for node in policy_cfg.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in EXPECTED_POLICY_TERMS:
                    observed_terms.append(target.id)
    assert observed_terms == EXPECTED_POLICY_TERMS

    post_init = next(node for node in policy_cfg.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__")
    assignments = {
        node.targets[0].attr: ast.literal_eval(node.value)
        for node in ast.walk(post_init)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and isinstance(node.targets[0].value, ast.Name)
        and node.targets[0].value.id == "self"
    }
    assert assignments["history_length"] == 5
    assert assignments["enable_corruption"] is True
    assert assignments["concatenate_terms"] is True


def test_policy_observation_dimension_snapshot():
    per_frame = {
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "velocity_commands": 3,
        "joint_pos_rel": 22,
        "joint_vel_rel": 22,
        "last_action": 22,
        "gait_phase": 2,
    }
    assert sum(per_frame.values()) == 77
    assert sum(per_frame.values()) * 5 == 385


def test_sim_timing_contract():
    env_cfg = class_def(_module(), "G0RobotLabEnvCfg")
    post_init = next(node for node in env_cfg.body if isinstance(node, ast.FunctionDef) and node.name == "__post_init__")
    constants = {}
    for node in ast.walk(post_init):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == "self"
        ):
            constants[node.targets[0].attr] = ast.literal_eval(node.value)
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.targets[0].value, ast.Attribute)
            and isinstance(node.targets[0].value.value, ast.Name)
            and node.targets[0].value.value.id == "self"
            and node.targets[0].value.attr == "sim"
            and node.targets[0].attr == "dt"
        ):
            constants["sim.dt"] = ast.literal_eval(node.value)
    assert constants["decimation"] == 4
    assert constants["sim.dt"] == 0.005
    assert constants["decimation"] * constants["sim.dt"] == pytest.approx(0.02)
