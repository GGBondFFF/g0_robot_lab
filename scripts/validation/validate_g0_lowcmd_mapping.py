from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.validation._joint_contract_tables import JointContractTables, load_joint_contract_tables
from scripts.validation._lowcmd_mapping import build_default_safety_limits, map_policy_to_lowcmd_dry_run
from scripts.validation._safety_filter import SafetyFilterError, SafetyState


EXIT_OK = 0
EXIT_CONTRACT_FAILURE = 2
DEFAULT_JSON_PATH = REPO_ROOT / "logs" / "validation" / "lowcmd_mapping_offline.json"
DEFAULT_ISAAC_JSON_PATH = REPO_ROOT / "logs" / "validation" / "lowcmd_mapping_isaac_500.json"
DEFAULT_TASK = "G0-Velocity-v0"
DEFAULT_CHECKPOINT = REPO_ROOT / "logs" / "rsl_rl" / "g0_velocity" / "2026-05-14_18-29-19" / "model_9999.pt"
DEFAULT_STEPS = 500
DEFAULT_NUM_ENVS = 1
DEFAULT_SEED = 42
DEFAULT_ROOT_Z = 0.233
DEFAULT_ACTION_SCALE = 0.12


@dataclass(frozen=True)
class CaseExpectation:
    name: str
    raw_action: list[float]
    state: SafetyState
    now: float
    emergency_stop: bool = False
    hardware_enabled: bool = False
    expect_reject: bool = False
    expect_warning: bool = False


@dataclass
class RunningStats:
    count: int = 0
    sum: float = 0.0
    sum_sq: float = 0.0
    min: float | None = None
    max: float | None = None

    def update(self, values: list[float]) -> None:
        for value in values:
            value = float(value)
            self.count += 1
            self.sum += value
            self.sum_sq += value * value
            self.min = value if self.min is None else min(self.min, value)
            self.max = value if self.max is None else max(self.max, value)

    def as_dict(self) -> dict[str, float | int | None]:
        mean = self.sum / self.count if self.count else None
        variance = self.sum_sq / self.count - mean * mean if self.count and mean is not None else None
        std = math.sqrt(max(variance, 0.0)) if variance is not None else None
        return {
            "count": self.count,
            "min": self.min,
            "max": self.max,
            "mean": mean,
            "std": std,
        }


def _finite_case_actions(value: float, count: int) -> list[float]:
    return [float(value)] * count


def _validate_motor_fields(
    *,
    command,
    tables: JointContractTables,
    action_scale: float,
    expected_target_by_joint: dict[str, float] | None,
    case_name: str,
) -> tuple[bool, list[str], float]:
    errors: list[str] = []
    max_error = 0.0
    if command.dry_run is not True:
        errors.append(f"{case_name}: dry_run must be True")
    if len(command.motors) != len(tables.joint_order):
        errors.append(f"{case_name}: expected {len(tables.joint_order)} motors, got {len(command.motors)}")
    for index, motor in enumerate(command.motors):
        if motor.motor_id != index:
            errors.append(f"{case_name}: motor_id mismatch at index {index}: got {motor.motor_id}")
        if motor.joint_name != tables.joint_order[index]:
            errors.append(
                f"{case_name}: joint order mismatch at index {index}: got {motor.joint_name}, expected {tables.joint_order[index]}"
            )
        if not all(
            math.isfinite(value)
            for value in (
                motor.q,
                motor.dq,
                motor.kp,
                motor.kd,
                motor.tau_ff,
                motor.effort_limit,
                motor.source_raw_action,
                motor.source_clipped_action,
            )
        ):
            errors.append(f"{case_name}: non-finite command field on joint {motor.joint_name}")
        lower = tables.pos_lower[motor.joint_name]
        upper = tables.pos_upper[motor.joint_name]
        if motor.q < lower - 1e-9 or motor.q > upper + 1e-9:
            errors.append(f"{case_name}: target outside limits for {motor.joint_name}: {motor.q}")
        if expected_target_by_joint is not None:
            error = abs(motor.q - expected_target_by_joint[motor.joint_name])
            max_error = max(max_error, error)
            if error > 1e-9:
                errors.append(
                    f"{case_name}: target mismatch for {motor.joint_name}: got {motor.q}, expected {expected_target_by_joint[motor.joint_name]}"
                )
        if motor.source_clipped_action < -1.0 - 1e-9 or motor.source_clipped_action > 1.0 + 1e-9:
            errors.append(f"{case_name}: clipped action out of range for {motor.joint_name}: {motor.source_clipped_action}")
        if motor.tau_ff < -motor.effort_limit - 1e-9 or motor.tau_ff > motor.effort_limit + 1e-9:
            errors.append(f"{case_name}: tau_ff exceeds effort limit for {motor.joint_name}")
    return not errors, errors, max_error


def _build_cases(tables: JointContractTables) -> list[CaseExpectation]:
    default_state = SafetyState(last_obs_time=0.0)
    prev_zero_action = {joint_name: 0.0 for joint_name in tables.joint_order}
    prev_target = {joint_name: tables.default_joint_pos[joint_name] for joint_name in tables.joint_order}
    return [
        CaseExpectation("zero_action", _finite_case_actions(0.0, len(tables.joint_order)), default_state, now=0.0),
        CaseExpectation("all_plus_one", _finite_case_actions(1.0, len(tables.joint_order)), default_state, now=0.0),
        CaseExpectation("all_minus_one", _finite_case_actions(-1.0, len(tables.joint_order)), default_state, now=0.0),
        CaseExpectation("overflow_action_ten", _finite_case_actions(10.0, len(tables.joint_order)), default_state, now=0.0),
        CaseExpectation("nan_action", [math.nan] + _finite_case_actions(0.0, len(tables.joint_order) - 1), default_state, now=0.0, expect_reject=True),
        CaseExpectation("inf_action", [math.inf] + _finite_case_actions(0.0, len(tables.joint_order) - 1), default_state, now=0.0, expect_reject=True),
        CaseExpectation(
            "fast_action_jump",
            _finite_case_actions(1.0, len(tables.joint_order)),
            SafetyState(prev_target=dict(prev_target), prev_action=dict(prev_zero_action), last_obs_time=0.0),
            now=0.0,
            expect_warning=True,
        ),
        CaseExpectation(
            "stale_observation",
            _finite_case_actions(0.0, len(tables.joint_order)),
            SafetyState(last_obs_time=0.0),
            now=1.0,
            expect_reject=True,
        ),
        CaseExpectation(
            "emergency_stop",
            _finite_case_actions(1.0, len(tables.joint_order)),
            SafetyState(
                prev_target={joint_name: tables.default_joint_pos[joint_name] + 0.02 for joint_name in tables.joint_order},
                prev_action={joint_name: 0.25 for joint_name in tables.joint_order},
                last_obs_time=0.0,
            ),
            now=1.0,
            emergency_stop=True,
            expect_warning=True,
        ),
        CaseExpectation(
            "hardware_enabled_true",
            _finite_case_actions(0.0, len(tables.joint_order)),
            default_state,
            now=0.0,
            hardware_enabled=True,
            expect_reject=True,
        ),
    ]


def run_offline_contract(*, emit_json: Path | None) -> tuple[int, dict[str, Any]]:
    os.environ["G0_ALLOW_HARDWARE"] = "0"
    tables = load_joint_contract_tables()
    limits = build_default_safety_limits(tables.joint_order)
    action_scale = 0.12
    rejected_cases: list[str] = []
    warning_cases: list[str] = []
    passed_cases: list[str] = []
    case_reports: list[dict[str, Any]] = []
    errors: list[str] = []
    target_mapping_max_error = 0.0
    clipping_confirmed = False

    for case in _build_cases(tables):
        try:
            command = map_policy_to_lowcmd_dry_run(
                case.raw_action,
                joint_order=tables.joint_order,
                default_joint_pos=tables.default_joint_pos,
                action_scale=action_scale,
                limits=limits,
                state=case.state,
                now=case.now,
                emergency_stop=case.emergency_stop,
                hardware_enabled=case.hardware_enabled,
            )
        except SafetyFilterError as exc:
            case_reports.append({"name": case.name, "result": "rejected", "message": str(exc)})
            if case.expect_reject:
                rejected_cases.append(case.name)
                continue
            errors.append(f"{case.name}: unexpected reject: {exc}")
            continue

        if case.expect_reject:
            errors.append(f"{case.name}: expected reject but command was produced")
            continue

        expected_target_by_joint: dict[str, float] | None = None
        if case.name in {"zero_action", "all_plus_one", "all_minus_one", "overflow_action_ten"}:
            expected_target_by_joint = {}
            for joint_name, motor in zip(tables.joint_order, command.motors):
                clipped = max(-1.0, min(1.0, motor.source_raw_action))
                expected_target_by_joint[joint_name] = max(
                    tables.pos_lower[joint_name],
                    min(tables.pos_upper[joint_name], tables.default_joint_pos[joint_name] + action_scale * clipped),
                )
            if case.name == "overflow_action_ten":
                clipping_confirmed = all(motor.source_clipped_action == 1.0 for motor in command.motors)
        ok, case_errors, case_error_max = _validate_motor_fields(
            command=command,
            tables=tables,
            action_scale=action_scale,
            expected_target_by_joint=expected_target_by_joint,
            case_name=case.name,
        )
        target_mapping_max_error = max(target_mapping_max_error, case_error_max)
        if not ok:
            errors.extend(case_errors)
            continue
        if case.expect_warning or case.emergency_stop:
            warning_cases.append(case.name)
        passed_cases.append(case.name)
        case_reports.append(
            {
                "name": case.name,
                "result": "passed",
                "command_count": len(command.motors),
                "dry_run": command.dry_run,
                "diagnostics": [],
            }
        )

    payload = {
        "mode": "offline-contract",
        "result": "PASS" if not errors else "FAIL",
        "joint_count": len(tables.joint_order),
        "command_count": len(tables.joint_order),
        "passed_cases": passed_cases,
        "rejected_cases": rejected_cases,
        "warning_cases": warning_cases,
        "motor_id_convention": "motor_id = index (software convention only)",
        "no_hardware_confirmation": True,
        "action_clipping_confirmation": clipping_confirmed,
        "target_mapping_max_error": target_mapping_max_error,
        "expected_reject_case_count": 4,
        "case_reports": case_reports,
        "errors": errors,
    }

    if emit_json is not None:
        emit_json.parent.mkdir(parents=True, exist_ok=True)
        emit_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload["json_path"] = str(emit_json)

    return (EXIT_OK if not errors else EXIT_CONTRACT_FAILURE), payload


def _print_summary(payload: dict[str, Any]) -> None:
    title = "G0 LowCmd Mapping Offline Validation" if payload["mode"] == "offline-contract" else "G0 LowCmd Mapping Isaac Policy Sampling"
    print(title)
    print(f"mode: {payload['mode']}")
    print(f"result: {payload['result']}")
    print(f"joint count: {payload['joint_count']}")
    print(f"command count: {payload['command_count']}")
    if payload["mode"] == "offline-contract":
        print(f"passed cases: {', '.join(payload['passed_cases']) if payload['passed_cases'] else 'none'}")
        print(f"rejected cases: {', '.join(payload['rejected_cases']) if payload['rejected_cases'] else 'none'}")
        print(f"warning cases: {', '.join(payload['warning_cases']) if payload['warning_cases'] else 'none'}")
    else:
        metrics = payload["metrics"]
        raw_action = metrics["raw_action"]
        clipped_action = metrics["clipped_action"]
        print(
            "raw action stats: "
            f"min={raw_action['min']}, max={raw_action['max']}, mean={raw_action['mean']}, std={raw_action['std']}"
        )
        print(f"raw action out_of_range_count: {metrics['raw_action_out_of_range_count']}")
        print(f"clipped action range: [{clipped_action['min']}, {clipped_action['max']}]")
        print(f"rejected step count: {metrics['rejected_step_count']}")
        print(f"emergency stop count: {metrics['emergency_stop_count']}")
        print(f"stale observation count: {metrics['stale_observation_count']}")
        print(f"motor_id mismatch count: {metrics['motor_id_mismatch_count']}")
        print(f"non-finite command count: {metrics['non_finite_command_count']}")
        print(f"worst raw action: {_summarize_worst(metrics['worst_raw_action'])}")
        print(f"worst target delta: {_summarize_worst(metrics['worst_target_delta'])}")
        print(f"worst safety clamp: {_summarize_worst(metrics['worst_safety_clamp'])}")
    print(f"motor_id convention: {payload['motor_id_convention']}")
    print(f"no-hardware confirmation: {payload['no_hardware_confirmation']}")
    if payload["mode"] == "offline-contract":
        print(f"action clipping confirmation: {payload['action_clipping_confirmation']}")
        print(f"target mapping max error: {payload['target_mapping_max_error']:.6g}")
    else:
        print(f"dry_run_only: {payload['dry_run_only']}")
        print(f"action clipping confirmation: {payload['metrics']['clipped_action']['min'] is not None}")
        print(f"target mapping max error: {payload['metrics']['max_target_mapping_error']:.6g}")
        print(f"max safety clamp correction: {payload['metrics']['max_safety_clamp_correction']:.6g}")
    if payload.get("json_path"):
        print(f"json: {payload['json_path']}")
    if payload["errors"]:
        print("errors:")
        for error in payload["errors"]:
            print(f"  - {error}")


def _summarize_worst(record: dict[str, Any]) -> str:
    if record.get("value") is None:
        return "none"
    return f"step={record.get('step')}, joint={record.get('joint_name')}, value={record.get('value')}"


def _flatten_env_actions(action_tensor: Any) -> list[list[float]]:
    if hasattr(action_tensor, "detach"):
        rows = action_tensor.detach().cpu().tolist()
    else:
        rows = action_tensor
    return [[float(value) for value in row] for row in rows]


def _make_expected_safe_target(
    *,
    joint_name: str,
    default_joint_pos: float,
    action_scale: float,
    raw_action: float,
    limits,
) -> tuple[float, float]:
    clipped = max(-1.0, min(1.0, float(raw_action)))
    pre_safety = float(default_joint_pos) + action_scale * clipped
    safe = max(limits.pos_lower[joint_name], min(limits.pos_upper[joint_name], pre_safety))
    return pre_safety, safe


def run_isaac_policy_sample(
    *,
    task: str,
    checkpoint: Path,
    steps: int,
    num_envs: int,
    seed: int,
    root_z: float,
    emit_json: Path | None,
    device: str | None,
) -> tuple[int, dict[str, Any]]:
    import gc

    import torch

    from scripts.validation._rollout_core import (
        _iter_obs_tensors,
        _load_policy,
        _reset_env,
        _resolve_checkpoint_path,
        _step_env,
        verify_checkpoint,
    )
    from tests.helpers.isaaclab_runtime import get_robot_joint_names, resolve_action_joint_order

    os.environ["G0_ALLOW_HARDWARE"] = "0"

    tables = load_joint_contract_tables()
    limits = build_default_safety_limits(tables.joint_order)
    checkpoint_path = _resolve_checkpoint_path(checkpoint)
    errors: list[str] = []
    rejected_step_count = 0
    stale_observation_count = 0
    emergency_stop_count = 0
    motor_id_mismatch_count = 0
    non_finite_command_count = 0
    lowcmd_command_count = 0
    raw_action_out_of_range_count = 0
    max_target_mapping_error = 0.0
    max_safety_clamp_correction = 0.0
    worst_raw_action = {"value": None, "step": None, "joint_name": None}
    worst_target_delta = {"value": None, "step": None, "joint_name": None}
    worst_safety_clamp = {"value": None, "step": None, "joint_name": None}
    raw_stats = RunningStats()
    clipped_stats = RunningStats()
    control_dt = 0.02
    env_states = [SafetyState(last_obs_time=0.0) for _ in range(num_envs)]
    env = None
    policy = None
    policy_reset = None
    runner = None
    base_env = None
    robot = None
    resolved_device = device

    try:
        sha256 = verify_checkpoint(checkpoint_path)
        env, policy, policy_reset, runner, _env_cfg, _agent_cfg = _load_policy(
            task=task,
            checkpoint=checkpoint_path,
            seed=seed,
            num_envs=num_envs,
            root_z=root_z,
            device=device,
        )
        resolved_device = str(env.unwrapped.device)
        _reset_env(env)
        obs = env.get_observations()
        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        robot_joint_names = get_robot_joint_names(robot)
        action_joint_order = resolve_action_joint_order(base_env.action_manager, robot_joint_names)
        if action_joint_order != list(tables.joint_order):
            errors.append(
                f"joint order mismatch: runtime={action_joint_order}, expected={list(tables.joint_order)}"
            )

        for step in range(steps):
            obs_tensors = list(_iter_obs_tensors(obs))
            if not obs_tensors:
                errors.append("Could not resolve policy observation tensor for finiteness check.")
                break
            if not all(torch.isfinite(obs_tensor).all().item() for obs_tensor in obs_tensors):
                errors.append(f"Non-finite observation at step {step}")
                break

            with torch.inference_mode():
                actions = policy(obs)
            if not torch.isfinite(actions).all():
                errors.append(f"Non-finite action at step {step}")
                break
            if tuple(actions.shape) != (num_envs, len(tables.joint_order)):
                errors.append(
                    f"Action shape mismatch at step {step}: got {list(actions.shape)}, expected {[num_envs, len(tables.joint_order)]}"
                )
                break

            raw_rows = _flatten_env_actions(actions)
            clipped_rows = [[max(-1.0, min(1.0, value)) for value in row] for row in raw_rows]
            raw_stats.update([value for row in raw_rows for value in row])
            clipped_stats.update([value for row in clipped_rows for value in row])
            raw_action_out_of_range_count += sum(1 for row in raw_rows for value in row if value < -1.0 or value > 1.0)

            for env_index, raw_row in enumerate(raw_rows):
                state = env_states[env_index]
                try:
                    command = map_policy_to_lowcmd_dry_run(
                        raw_row,
                        joint_order=tables.joint_order,
                        default_joint_pos=tables.default_joint_pos,
                        action_scale=DEFAULT_ACTION_SCALE,
                        limits=limits,
                        state=state,
                        now=float(step) * control_dt,
                        emergency_stop=False,
                        hardware_enabled=False,
                    )
                except SafetyFilterError as exc:
                    rejected_step_count += 1
                    if "Stale observation" in str(exc):
                        stale_observation_count += 1
                    errors.append(f"step {step} env {env_index}: unexpected mapping reject: {exc}")
                    continue

                if command.dry_run is not True:
                    errors.append(f"step {step} env {env_index}: FakeLowCmd.dry_run is not True")
                if len(command.motors) != len(tables.joint_order):
                    errors.append(
                        f"step {step} env {env_index}: motor count {len(command.motors)} != {len(tables.joint_order)}"
                    )
                lowcmd_command_count += 1

                next_prev_action: dict[str, float] = {}
                next_prev_target: dict[str, float] = {}
                for motor_index, motor in enumerate(command.motors):
                    joint_name = tables.joint_order[motor_index]
                    if motor.motor_id != motor_index:
                        motor_id_mismatch_count += 1
                        errors.append(
                            f"step {step} env {env_index}: motor_id mismatch at index {motor_index}: got {motor.motor_id}"
                        )
                    if motor.joint_name != joint_name:
                        errors.append(
                            f"step {step} env {env_index}: joint name mismatch at index {motor_index}: got {motor.joint_name}, expected {joint_name}"
                        )
                    fields = (
                        motor.q,
                        motor.dq,
                        motor.kp,
                        motor.kd,
                        motor.tau_ff,
                        motor.effort_limit,
                        motor.source_raw_action,
                        motor.source_clipped_action,
                    )
                    if not all(math.isfinite(value) for value in fields):
                        non_finite_command_count += 1
                        errors.append(f"step {step} env {env_index}: non-finite command field for {joint_name}")
                    if motor.source_clipped_action < -1.0 - 1e-9 or motor.source_clipped_action > 1.0 + 1e-9:
                        errors.append(
                            f"step {step} env {env_index}: clipped action out of range for {joint_name}: {motor.source_clipped_action}"
                        )
                    if motor.q < limits.pos_lower[joint_name] - 1e-9 or motor.q > limits.pos_upper[joint_name] + 1e-9:
                        errors.append(f"step {step} env {env_index}: q outside safety limits for {joint_name}: {motor.q}")

                    pre_safety, position_clamped_target = _make_expected_safe_target(
                        joint_name=joint_name,
                        default_joint_pos=tables.default_joint_pos[joint_name],
                        action_scale=DEFAULT_ACTION_SCALE,
                        raw_action=motor.source_raw_action,
                        limits=limits,
                    )
                    reconstructed_pre_safety = (
                        tables.default_joint_pos[joint_name] + DEFAULT_ACTION_SCALE * motor.source_clipped_action
                    )
                    mapping_error = abs(pre_safety - reconstructed_pre_safety)
                    max_target_mapping_error = max(max_target_mapping_error, mapping_error)
                    if mapping_error > 1e-9:
                        errors.append(
                            f"step {step} env {env_index}: target mapping mismatch for {joint_name}: got {reconstructed_pre_safety}, expected {pre_safety}"
                        )
                    clamp_correction = abs(motor.q - pre_safety)
                    if clamp_correction > max_safety_clamp_correction:
                        max_safety_clamp_correction = clamp_correction
                        worst_safety_clamp = {"value": clamp_correction, "step": step, "joint_name": joint_name}

                    raw_value = motor.source_raw_action
                    if worst_raw_action["value"] is None or abs(raw_value) > abs(float(worst_raw_action["value"])):
                        worst_raw_action = {"value": raw_value, "step": step, "joint_name": joint_name}

                    if state.prev_target and joint_name in state.prev_target:
                        target_delta = abs(motor.q - state.prev_target[joint_name])
                        if worst_target_delta["value"] is None or target_delta > float(worst_target_delta["value"]):
                            worst_target_delta = {"value": target_delta, "step": step, "joint_name": joint_name}

                    next_prev_action[joint_name] = motor.source_clipped_action
                    next_prev_target[joint_name] = motor.q

                state.prev_action = next_prev_action
                state.prev_target = next_prev_target
                state.last_obs_time = float(step) * control_dt

            dones = None
            obs, _reward, dones, _terminated, _truncated, _info = _step_env(env, actions)
            if policy_reset is not None and dones is not None:
                policy_reset(dones)

            if errors:
                break

        payload = {
            "mode": "isaac-policy-sample",
            "result": "PASS" if not errors else "FAIL",
            "checkpoint": {"path": str(checkpoint_path), "sha256": sha256},
            "task": task,
            "steps": steps,
            "num_envs": num_envs,
            "device": resolved_device,
            "joint_count": len(tables.joint_order),
            "command_count": lowcmd_command_count,
            "metrics": {
                "raw_action": raw_stats.as_dict(),
                "clipped_action": clipped_stats.as_dict(),
                "raw_action_out_of_range_count": raw_action_out_of_range_count,
                "max_target_mapping_error": max_target_mapping_error,
                "max_safety_clamp_correction": max_safety_clamp_correction,
                "rejected_step_count": rejected_step_count,
                "emergency_stop_count": emergency_stop_count,
                "stale_observation_count": stale_observation_count,
                "worst_raw_action": worst_raw_action,
                "worst_target_delta": worst_target_delta,
                "worst_safety_clamp": worst_safety_clamp,
                "motor_id_mismatch_count": motor_id_mismatch_count,
                "non_finite_command_count": non_finite_command_count,
            },
            "motor_id_convention": "motor_id = index (software convention only)",
            "no_hardware_confirmation": True,
            "dry_run_only": True,
            "errors": errors,
        }
    except Exception as exc:
        payload = {
            "mode": "isaac-policy-sample",
            "result": "FAIL",
            "checkpoint": {"path": str(checkpoint_path), "sha256": "unverified"},
            "task": task,
            "steps": steps,
            "num_envs": num_envs,
            "device": resolved_device,
            "joint_count": len(load_joint_contract_tables().joint_order),
            "command_count": lowcmd_command_count,
            "metrics": {
                "raw_action": raw_stats.as_dict(),
                "clipped_action": clipped_stats.as_dict(),
                "raw_action_out_of_range_count": raw_action_out_of_range_count,
                "max_target_mapping_error": max_target_mapping_error,
                "max_safety_clamp_correction": max_safety_clamp_correction,
                "rejected_step_count": rejected_step_count,
                "emergency_stop_count": emergency_stop_count,
                "stale_observation_count": stale_observation_count,
                "worst_raw_action": worst_raw_action,
                "worst_target_delta": worst_target_delta,
                "worst_safety_clamp": worst_safety_clamp,
                "motor_id_mismatch_count": motor_id_mismatch_count,
                "non_finite_command_count": non_finite_command_count,
            },
            "motor_id_convention": "motor_id = index (software convention only)",
            "no_hardware_confirmation": True,
            "dry_run_only": True,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    finally:
        if env is not None:
            env.close()
        del robot
        del base_env
        del policy_reset
        del policy
        del runner
        gc.collect()

    if emit_json is not None:
        emit_json.parent.mkdir(parents=True, exist_ok=True)
        emit_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        payload["json_path"] = str(emit_json)

    return (EXIT_OK if not payload["errors"] else EXIT_CONTRACT_FAILURE), payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LowCmd mapping validator for G0.")
    parser.add_argument("--mode", choices=("offline-contract", "isaac-policy-sample"), required=True)
    parser.add_argument("--emit-json", type=Path, default=None)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--root-z", type=float, default=DEFAULT_ROOT_Z)
    parser.add_argument("--device", default=None)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "offline-contract":
        emit_json = args.emit_json or DEFAULT_JSON_PATH
        exit_code, payload = run_offline_contract(emit_json=emit_json)
    elif args.mode == "isaac-policy-sample":
        from isaaclab.app import AppLauncher

        emit_json = args.emit_json or DEFAULT_ISAAC_JSON_PATH
        os.environ["G0_ALLOW_HARDWARE"] = "0"
        app_launcher = AppLauncher({"headless": args.headless})
        simulation_app = app_launcher.app
        try:
            exit_code, payload = run_isaac_policy_sample(
                task=args.task,
                checkpoint=args.checkpoint,
                steps=args.steps,
                num_envs=args.num_envs,
                seed=args.seed,
                root_z=args.root_z,
                emit_json=emit_json,
                device=args.device,
            )
        finally:
            simulation_app.close()
    else:
        print(f"Unsupported mode: {args.mode}")
        return EXIT_CONTRACT_FAILURE
    _print_summary(payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
