from __future__ import annotations

import pytest

from tests.conftest import RAW_RSL_RL_CHECKPOINT, RAW_RSL_RL_CHECKPOINT_SHA256


pytestmark = [pytest.mark.release_gate, pytest.mark.slow]


def test_lowcmd_mapping_chain_release_gate_500_steps(isaac_sim_app):
    del isaac_sim_app

    from scripts.validation.validate_g0_lowcmd_mapping import run_isaac_policy_sample

    exit_code, payload = run_isaac_policy_sample(
        task="G0-Velocity-v0",
        checkpoint=RAW_RSL_RL_CHECKPOINT,
        steps=500,
        num_envs=1,
        seed=42,
        root_z=0.233,
        emit_json=None,
        device=None,
    )

    assert exit_code == 0, payload.get("errors")
    assert payload["result"] == "PASS", payload.get("errors")
    assert payload["checkpoint"]["sha256"] == RAW_RSL_RL_CHECKPOINT_SHA256
    assert payload["dry_run_only"] is True
    assert payload["no_hardware_confirmation"] is True
    assert payload["joint_count"] == 22
    assert payload["command_count"] == 500

    metrics = payload["metrics"]
    raw_action = metrics["raw_action"]
    clipped_action = metrics["clipped_action"]

    assert raw_action["count"] == 500 * 22
    assert clipped_action["count"] == 500 * 22
    assert clipped_action["min"] is not None
    assert clipped_action["max"] is not None
    tolerance = 1.0e-6
    assert clipped_action["min"] >= -1.0 - tolerance
    assert clipped_action["max"] <= 1.0 + tolerance

    assert metrics["max_target_mapping_error"] <= 1.0e-6
    assert metrics["rejected_step_count"] == 0
    assert metrics["emergency_stop_count"] == 0
    assert metrics["stale_observation_count"] == 0
    assert metrics["motor_id_mismatch_count"] == 0
    assert metrics["non_finite_command_count"] == 0

    assert payload["errors"] == []
