from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
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
    print("G0 LowCmd Mapping Offline Validation")
    print(f"mode: {payload['mode']}")
    print(f"result: {payload['result']}")
    print(f"joint count: {payload['joint_count']}")
    print(f"command count: {payload['command_count']}")
    print(f"passed cases: {', '.join(payload['passed_cases']) if payload['passed_cases'] else 'none'}")
    print(f"rejected cases: {', '.join(payload['rejected_cases']) if payload['rejected_cases'] else 'none'}")
    print(f"warning cases: {', '.join(payload['warning_cases']) if payload['warning_cases'] else 'none'}")
    print(f"motor_id convention: {payload['motor_id_convention']}")
    print(f"no-hardware confirmation: {payload['no_hardware_confirmation']}")
    print(f"action clipping confirmation: {payload['action_clipping_confirmation']}")
    print(f"target mapping max error: {payload['target_mapping_max_error']:.6g}")
    if payload.get("json_path"):
        print(f"json: {payload['json_path']}")
    if payload["errors"]:
        print("errors:")
        for error in payload["errors"]:
            print(f"  - {error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline LowCmd mapping validator for G0.")
    parser.add_argument("--mode", choices=("offline-contract",), required=True)
    parser.add_argument("--emit-json", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode != "offline-contract":
        print(f"Unsupported mode: {args.mode}")
        return EXIT_CONTRACT_FAILURE
    exit_code, payload = run_offline_contract(emit_json=args.emit_json)
    _print_summary(payload)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
