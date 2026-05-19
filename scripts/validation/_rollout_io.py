from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def default_json_path(repo_root: Path) -> Path:
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return repo_root / "logs" / "validation" / f"policy_rollout_{timestamp}.json"


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def print_summary(payload: dict[str, Any]) -> None:
    def emit(line: str) -> None:
        print(line, flush=True)

    command = payload.get("command", [])
    checkpoint = payload["checkpoint"]
    config = payload["config"]
    result = payload["result"]
    metrics = payload["metrics"]
    warnings = payload.get("warnings", [])
    json_out = payload.get("json_out")

    emit("=== Policy Rollout Safety (Phase 1) ===")
    emit(f"RESULT: {'PASS (contract)' if result['contract_pass'] else 'FAIL (contract)'}")
    emit(f"command: {' '.join(command)}")
    emit(f"task: {config['task']}")
    emit(f"checkpoint: {checkpoint['path']}")
    emit(f"sha256: {checkpoint['sha256']}")
    emit(
        f"steps: {config['steps']}  num_envs: {config['num_envs']}  seed: {config['seed']}  device: {config['device']}"
    )

    raw_action = metrics["raw_policy_action"]
    emit(
        "raw_policy_action: "
        f"shape={raw_action['shape']} min={raw_action['min']} max={raw_action['max']} mean={raw_action['mean']} "
        f"std={raw_action['std']} out_of_range={raw_action['out_of_range_count']}"
    )
    emit(
        "raw_action_worst_case: "
        f"min={raw_action['raw_action_min_value']} step={raw_action['raw_action_min_step']} "
        f"joint_index={raw_action['raw_action_min_joint_index']} joint_name={raw_action['raw_action_min_joint_name']} "
        f"clipped={raw_action['clipped_action_at_raw_min']} | "
        f"max={raw_action['raw_action_max_value']} step={raw_action['raw_action_max_step']} "
        f"joint_index={raw_action['raw_action_max_joint_index']} joint_name={raw_action['raw_action_max_joint_name']} "
        f"clipped={raw_action['clipped_action_at_raw_max']}"
    )

    clipped_action = metrics["clipped_action"]
    emit(
        "clipped_action: "
        f"shape={clipped_action['shape']} min={clipped_action['min']} max={clipped_action['max']} "
        f"mean={clipped_action['mean']} std={clipped_action['std']} clip_range={clipped_action['clip_range']}"
    )

    root_z = metrics["root_z"]
    emit(f"root_z: min={root_z['min']} max={root_z['max']} mean={root_z['mean']}")

    terminations = metrics["terminations"]
    emit(
        "terminations: "
        f"base_height={terminations['base_height']} "
        f"bad_orientation={terminations['bad_orientation']} "
        f"time_out={terminations['time_out']} "
        f"resets={metrics['resets']}"
    )

    effort = metrics["effort"]
    if effort["available"]:
        emit(
            "effort: "
            f"ratio_max={effort['max']} steps_above_{effort['threshold']}={effort['steps_above_threshold']}"
        )
        emit(
            "effort_worst_case: "
            f"value={effort['effort_ratio_worst_value']} step={effort['effort_ratio_worst_step']} "
            f"joint_index={effort['effort_ratio_worst_joint_index']} "
            f"joint_name={effort['effort_ratio_worst_joint_name']} torque={effort['torque_at_worst']} "
            f"effort_limit={effort['effort_limit_at_worst']}"
        )
    else:
        emit("effort: unavailable")

    target_delta = metrics["target_delta_abs"]
    emit(
        "target_delta_abs: "
        f"available={target_delta['available']} max={target_delta['max']} mean={target_delta['mean']}"
    )
    if target_delta["available"]:
        emit(
            "target_delta_worst_case: "
            f"value={target_delta['target_delta_worst_value']} step={target_delta['target_delta_worst_step']} "
            f"joint_index={target_delta['target_delta_worst_joint_index']} "
            f"joint_name={target_delta['target_delta_worst_joint_name']}"
        )

    torque = metrics["torque"]
    emit(f"torque: available={torque['available']} min={torque['min']} max={torque['max']}")

    margin = metrics["joint_limit_margin_min"]
    emit(f"joint_limit_margin_min: {margin}")
    joint_limit = metrics["joint_limit_margin"]
    emit(
        "joint_limit_margin_worst_case: "
        f"value={joint_limit['joint_limit_margin_worst_value']} step={joint_limit['joint_limit_margin_worst_step']} "
        f"joint_index={joint_limit['joint_limit_margin_worst_joint_index']} "
        f"joint_name={joint_limit['joint_limit_margin_worst_joint_name']} "
        f"joint_pos={joint_limit['joint_pos_at_worst']} "
        f"lower={joint_limit['joint_lower_limit_at_worst']} "
        f"upper={joint_limit['joint_upper_limit_at_worst']}"
    )
    emit(f"json_out: {json_out}")

    if result.get("errors"):
        emit("ERRORS:")
        for item in result["errors"]:
            emit(f"- {item}")

    if warnings:
        emit("WARNINGS:")
        for item in warnings:
            emit(f"- {item}")
    else:
        emit("WARNINGS: none")
