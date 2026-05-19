from __future__ import annotations

import pytest

from tests.conftest import RAW_RSL_RL_CHECKPOINT, RAW_RSL_RL_CHECKPOINT_SHA256


pytestmark = [pytest.mark.release_gate, pytest.mark.slow]


def test_policy_rollout_safety_release_gate_500_steps(isaac_sim_app):
    del isaac_sim_app

    from scripts.validation._rollout_core import run_policy_rollout_validation

    payload = run_policy_rollout_validation(
        task="G0-Velocity-v0",
        checkpoint=RAW_RSL_RL_CHECKPOINT,
        steps=500,
        num_envs=1,
        seed=42,
        root_z=0.233,
        effort_ratio_threshold=0.9,
        write_json=False,
    )

    assert payload["checkpoint"]["sha256"] == RAW_RSL_RL_CHECKPOINT_SHA256
    assert payload["result"]["contract_pass"], payload["result"]["errors"]

    metrics = payload["metrics"]
    raw_action = metrics["raw_policy_action"]
    clipped_action = metrics["clipped_action"]
    terminations = metrics["terminations"]

    assert raw_action["shape"] == [1, 22]
    assert terminations["base_height"] == 0
    assert terminations["bad_orientation"] == 0
    assert metrics["resets"] == 0

    tolerance = 1.0e-6
    assert clipped_action["min"] is not None
    assert clipped_action["max"] is not None
    assert clipped_action["min"] >= -1.0 - tolerance
    assert clipped_action["max"] <= 1.0 + tolerance
