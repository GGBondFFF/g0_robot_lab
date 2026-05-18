#!/usr/bin/env python3
"""Run the G0 deploy-style MuJoCo validation matrix."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sim2sim.policy_io import policy_metadata, require_absolute_path


COMMANDS = [
    (0.0, 0.0, 0.0),
    (0.05, 0.0, 0.0),
    (0.1, 0.0, 0.0),
    (0.0, 0.0, 0.1),
]


@dataclass(frozen=True)
class Case:
    action_kind: str
    control_mode: str
    command: tuple[float, float, float]

    @property
    def name(self) -> str:
        cmd = "_".join(format_component(value) for value in self.command)
        return f"{self.action_kind}_{self.control_mode}_{cmd}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mujoco/g0.xml", help="Path to MuJoCo XML model.")
    parser.add_argument("--deploy-cfg", default="logs/sim2sim/g0_deploy/params/deploy.yaml", help="Path to deploy.yaml.")
    parser.add_argument("--policy", default=None, help="Absolute path to policy/checkpoint. Required for policy matrix cases.")
    parser.add_argument("--steps", type=int, default=200, help="Control steps per case.")
    parser.add_argument("--output-dir", default="logs/sim2sim/g0_deploy/validation_matrix", help="Directory for rollouts and checks.")
    parser.add_argument("--report", default="docs/g0_deploy_sim2sim_validation_matrix_report.md", help="Markdown summary report path.")
    return parser.parse_args()


def format_component(value: float) -> str:
    text = f"{value:.3g}".replace("-", "m").replace(".", "p")
    return f"c{text}"


def load_deploy_cfg(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def as_vector(values: Any, expected: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == 1 and expected > 1:
        array = np.repeat(array, expected)
    if array.shape != (expected,):
        raise ValueError(f"expected vector ({expected},), got {array.shape}")
    return array


def action_cfg(deploy_cfg: dict[str, Any]) -> dict[str, Any]:
    actions = deploy_cfg.get("actions") or {}
    if "JointPositionAction" in actions:
        return actions["JointPositionAction"]
    if len(actions) == 1:
        return next(iter(actions.values()))
    raise KeyError(f"No JointPositionAction in deploy.yaml actions: {list(actions)}")


def run_command(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return proc.returncode, proc.stdout


def scalar_max(array: np.ndarray) -> float:
    return float(np.nanmax(np.abs(array))) if array.size else float("nan")


def scalar_min(array: np.ndarray) -> float:
    return float(np.nanmin(array)) if array.size else float("nan")


def scalar_mean(array: np.ndarray) -> float:
    return float(np.nanmean(array)) if array.size else float("nan")


def ratio(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask) / mask.size) if mask.size else float("nan")


def rollout_metrics(rollout: Path, deploy_cfg: dict[str, Any]) -> dict[str, Any]:
    joint_names = list(deploy_cfg["joint_names"])
    action_dim = len(joint_names)
    effort_limit = as_vector(deploy_cfg["effort_limit_sim"], action_dim)
    velocity_limit = as_vector(deploy_cfg["velocity_limit_sim"], action_dim)
    data = np.load(rollout, allow_pickle=True)

    policy_action = np.asarray(data["policy_action"], dtype=np.float64)
    joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)
    pd_tau_cmd = np.asarray(data["pd_tau_cmd"], dtype=np.float64)
    pd_tau_cmd_clipped = np.asarray(data["pd_tau_cmd_clipped"], dtype=np.float64)
    root_height = np.asarray(data["root_height"], dtype=np.float64)
    contact_count = np.asarray(data["contact_count"], dtype=np.float64)
    foot_force = np.asarray(data["foot_contact_force_norm"], dtype=np.float64)
    target_joint_pos = np.asarray(data["target_joint_pos"], dtype=np.float64)
    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    base_ang_vel = np.asarray(data["base_ang_vel"], dtype=np.float64)
    root_quat = np.asarray(data["root_quat"], dtype=np.float64)
    projected_gravity = np.asarray(data["projected_gravity"], dtype=np.float64)
    velocity_exceeded = np.any(np.abs(joint_vel) > velocity_limit[None, :] + 1e-9, axis=0)
    torque_saturation = np.abs(pd_tau_cmd) > effort_limit[None, :] + 1e-9
    command = np.asarray(data["command"][0], dtype=np.float64).tolist()
    all_finite = all(
        bool(np.all(np.isfinite(np.asarray(data[key]))))
        for key in [
            "obs",
            "policy_action",
            "target_joint_pos",
            "pd_tau_cmd",
            "pd_tau_cmd_clipped",
            "joint_pos",
            "joint_vel",
            "root_height",
            "root_quat",
            "base_ang_vel",
            "projected_gravity",
            "contact_count",
            "foot_contact_force_norm",
        ]
    )
    return {
        "command": command,
        "control_mode": str(np.asarray(data["control_mode"]).item()) if "control_mode" in data else "unknown",
        "steps": int(data["obs"].shape[0]),
        "action_max_abs": scalar_max(policy_action),
        "action_saturation_ratio": ratio(np.abs(policy_action) >= 1.0 - 1e-6),
        "target_joint_pos_min": scalar_min(target_joint_pos),
        "target_joint_pos_max": float(np.nanmax(target_joint_pos)),
        "joint_pos_min": scalar_min(joint_pos),
        "joint_pos_max": float(np.nanmax(joint_pos)),
        "joint_vel_max_abs": scalar_max(joint_vel),
        "velocity_exceeded_count": int(np.count_nonzero(velocity_exceeded)),
        "velocity_exceeded_joints": [name for name, exceeded in zip(joint_names, velocity_exceeded, strict=True) if exceeded],
        "pd_tau_cmd_max_abs": scalar_max(pd_tau_cmd),
        "pd_tau_cmd_clipped_max_abs": scalar_max(pd_tau_cmd_clipped),
        "torque_saturation_ratio": ratio(torque_saturation),
        "root_height_min": scalar_min(root_height),
        "root_height_max": float(np.nanmax(root_height)),
        "root_height_final": float(root_height[-1]) if root_height.size else float("nan"),
        "root_quat_finite": bool(np.all(np.isfinite(root_quat))),
        "base_ang_vel_max_abs": scalar_max(base_ang_vel),
        "projected_gravity_finite": bool(np.all(np.isfinite(projected_gravity))),
        "contact_count_min": scalar_min(contact_count),
        "contact_count_mean": scalar_mean(contact_count),
        "contact_count_max": float(np.nanmax(contact_count)),
        "foot_force_min": scalar_min(foot_force),
        "foot_force_mean": scalar_mean(foot_force),
        "foot_force_max": float(np.nanmax(foot_force)),
        "nan_or_inf": not all_finite,
        "early_fall": bool(np.nanmin(root_height) < 0.12),
    }


def stability_score(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(metrics["root_height_min"]),
        -float(metrics["joint_vel_max_abs"]),
        -float(metrics["torque_saturation_ratio"]),
        -float(metrics["foot_force_max"]),
    )


def write_report(path: Path, records: list[dict[str, Any]], args: argparse.Namespace) -> None:
    ok_records = [record for record in records if record["run_ok"] and record["check_ok"]]
    stable = max(ok_records, key=lambda r: stability_score(r["metrics"])) if ok_records else None
    unstable = min(ok_records, key=lambda r: stability_score(r["metrics"])) if ok_records else None
    highest_action = max(ok_records, key=lambda r: r["metrics"]["action_saturation_ratio"]) if ok_records else None
    highest_torque = max(ok_records, key=lambda r: r["metrics"]["torque_saturation_ratio"]) if ok_records else None
    fall_cases = [record for record in ok_records if record["metrics"]["early_fall"]]
    velocity_cases = [record for record in ok_records if record["metrics"]["velocity_exceeded_count"] > 0]
    policy_records = [record for record in ok_records if record["case"].action_kind == "policy"]
    zero_records = [record for record in ok_records if record["case"].action_kind == "zero_action"]
    policy_root_ok = all(record["metrics"]["root_height_min"] >= 0.16 for record in policy_records)
    policy_torque_ok = all(record["metrics"]["torque_saturation_ratio"] <= 0.05 for record in policy_records)
    policy_action_ok = all(record["metrics"]["action_saturation_ratio"] <= 0.30 for record in policy_records)
    policy_velocity_ok = all(record["metrics"]["velocity_exceeded_count"] == 0 for record in policy_records)
    finite_ok = all(not record["metrics"].get("nan_or_inf", True) for record in ok_records)
    smoke_pass = (
        len(policy_records) == 8
        and int(args.steps) >= 500
        and policy_root_ok
        and policy_torque_ok
        and policy_action_ok
        and policy_velocity_ok
        and finite_ok
    )

    lines = [
        "# G0 Deploy Sim2sim Validation Matrix Report",
        "",
        "## Scope",
        "",
        f"- model: `{args.model}`",
        f"- deploy_cfg: `{args.deploy_cfg}`",
        f"- policy: `{args.policy}`",
        f"- steps per case: `{args.steps}`",
        f"- output_dir: `{args.output_dir}`",
        "",
        "This report validates the deploy-style MuJoCo runtime. It does not tune policy quality or physics parameters.",
        "",
    ]
    if args.policy is not None:
        identity = policy_metadata(args.policy, task="G0-Velocity-v0", command=(0.0, 0.0, 0.0), steps=args.steps)
        lines.extend(
            [
                "## Policy Identity",
                "",
                f"- policy_path: `{str(identity['policy_path'].item())}`",
                f"- policy_filename: `{str(identity['policy_filename'].item())}`",
                f"- policy_sha256: `{str(identity['policy_sha256'].item())}`",
                f"- checkpoint_run_folder: `{str(identity['checkpoint_run_folder'].item())}`",
                f"- task: `{str(identity['task'].item())}`",
                f"- steps: `{int(identity['steps'].item())}`",
                f"- action_dim: `{int(identity['action_dim'].item())}`",
                f"- obs_dim: `{int(identity['obs_dim'].item())}`",
                f"- action_scale: `{float(identity['action_scale'].item())}`",
                "",
            ]
        )
    lines.extend(
        [
        "## Case Matrix",
        "",
        "| case | run | check | action max | action sat | torque max | torque sat | vel exceeded | root h min/final | contacts mean/max | foot force mean/max | early fall |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for record in records:
        m = record.get("metrics", {})
        if not m:
            lines.append(
                f"| `{record['case'].name}` | `{record['run_ok']}` | `{record['check_ok']}` | "
                "| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
            )
            continue
        lines.append(
            f"| `{record['case'].name}` | `{record['run_ok']}` | `{record['check_ok']}` | "
            f"{m['action_max_abs']:.4g} | {m['action_saturation_ratio']:.4g} | "
            f"{m['pd_tau_cmd_max_abs']:.4g} | {m['torque_saturation_ratio']:.4g} | "
            f"{m['velocity_exceeded_count']} | {m['root_height_min']:.4g}/{m['root_height_final']:.4g} | "
            f"{m['contact_count_mean']:.4g}/{m['contact_count_max']:.4g} | "
            f"{m['foot_force_mean']:.4g}/{m['foot_force_max']:.4g} | `{m['early_fall']}` |"
        )

    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- cases run: `{len(records)}`",
            f"- cases with runner success: `{sum(1 for r in records if r['run_ok'])}/{len(records)}`",
            f"- cases with checker success: `{sum(1 for r in records if r['check_ok'])}/{len(records)}`",
            f"- most stable case by root height / velocity / torque / contact score: `{stable['case'].name if stable else 'n/a'}`",
            f"- least stable case by root height / velocity / torque / contact score: `{unstable['case'].name if unstable else 'n/a'}`",
            f"- highest action saturation case: `{highest_action['case'].name if highest_action else 'n/a'}`",
            f"- highest torque saturation case: `{highest_torque['case'].name if highest_torque else 'n/a'}`",
            f"- early fall cases: `{[record['case'].name for record in fall_cases]}`",
            f"- velocity-limit exceeded cases: `{[record['case'].name for record in velocity_cases]}`",
            "",
            "## Credibility Assessment",
            "",
            "- sim2sim started: `True`",
            f"- sim2sim smoke pass by this matrix: `{smoke_pass}`",
            "- sim2sim credible pass: `False`",
            "",
            "Smoke-pass criteria used here:",
            "",
            "- policy position and policy pd_torque both run at least 500 steps for all matrix commands",
            "- policy root height stays at or above `0.16 m`",
            "- velocity-limit exceeded joints are `0/22`",
            "- torque saturation ratio stays at or below `5%`",
            "- action saturation ratio stays at or below `30%`",
            "- no NaN/Inf is present",
            "",
            f"- policy root-height criterion: `{policy_root_ok}`",
            f"- policy torque saturation criterion: `{policy_torque_ok}`",
            f"- policy action saturation criterion: `{policy_action_ok}`",
            f"- policy velocity-limit criterion: `{policy_velocity_ok}`",
            f"- finite-data criterion: `{finite_ok}`",
            f"- zero-action early-fall cases: `{[record['case'].name for record in zero_records if record['metrics']['early_fall']]}`",
            "",
            "Current interpretation:",
            "",
            "- Passing policy cases support active-control matrix execution, but smoke pass remains false when any policy criterion fails.",
            "- Zero-action collapse does not by itself imply policy failure; it says the default target pose is not a static MuJoCo equilibrium under the current dynamics/contact model.",
            "- Credible pass is still false because it requires longer 1000-step runs, explainable position/pd_torque differences, and deeper actuator/contact behavior reports.",
            "",
            "## Position Vs PD Torque",
            "",
        ]
    )
    for action_kind in ["zero_action", "policy"]:
        for command in COMMANDS:
            subset = [r for r in ok_records if r["case"].action_kind == action_kind and r["case"].command == command]
            if len(subset) != 2:
                continue
            by_mode = {r["case"].control_mode: r for r in subset}
            if "position" not in by_mode or "pd_torque" not in by_mode:
                continue
            pm = by_mode["position"]["metrics"]
            tm = by_mode["pd_torque"]["metrics"]
            lines.append(
                f"- `{action_kind}` command `{list(command)}`: "
                f"root_min position/pd_torque `{pm['root_height_min']:.4g}` / `{tm['root_height_min']:.4g}`, "
                f"torque_sat `{pm['torque_saturation_ratio']:.4g}` / `{tm['torque_saturation_ratio']:.4g}`, "
                f"foot_force_max `{pm['foot_force_max']:.4g}` / `{tm['foot_force_max']:.4g}`"
            )

    lines.extend(
        [
            "",
            "## Zero-Action Vs Policy",
            "",
        ]
    )
    for control_mode in ["position", "pd_torque"]:
        for command in COMMANDS:
            subset = [r for r in ok_records if r["case"].control_mode == control_mode and r["case"].command == command]
            if len(subset) != 2:
                continue
            by_action = {r["case"].action_kind: r for r in subset}
            if "zero_action" not in by_action or "policy" not in by_action:
                continue
            zm = by_action["zero_action"]["metrics"]
            pm = by_action["policy"]["metrics"]
            lines.append(
                f"- `{control_mode}` command `{list(command)}`: "
                f"action_sat zero/policy `{zm['action_saturation_ratio']:.4g}` / `{pm['action_saturation_ratio']:.4g}`, "
                f"root_min `{zm['root_height_min']:.4g}` / `{pm['root_height_min']:.4g}`, "
                f"torque_sat `{zm['torque_saturation_ratio']:.4g}` / `{pm['torque_saturation_ratio']:.4g}`"
            )

    lines.extend(
        [
            "",
            "## Likely Failure Causes Ranked",
            "",
        ]
    )
    if fall_cases:
        lines.append("1. Root height drops below the fall heuristic in multiple zero-action cases, so passive/default-pose settling and contact dynamics need inspection before interpreting policy quality.")
    else:
        lines.append("1. No case tripped the root-height fall heuristic; root stability is not the top matrix signal.")
    if highest_action and highest_action["metrics"]["action_saturation_ratio"] > 0.0:
        lines.append("2. Policy action saturation is present; check policy observation conventions, action scale, and exported action processing.")
    else:
        lines.append("2. Policy action saturation ratio is zero in this matrix; action clipping is not the dominant signal.")
    if highest_torque and highest_torque["metrics"]["torque_saturation_ratio"] > 0.0:
        lines.append("3. Raw PD torque exceeds effort limits in some samples; inspect actuator limits and PD implementation before changing gains.")
    else:
        lines.append("3. Raw PD torque does not saturate in this matrix.")
    if velocity_cases:
        lines.append("4. Some joints exceeded `velocity_limit_sim`; inspect velocity-limit semantics and actuator behavior.")
    else:
        lines.append("4. No joint exceeded `velocity_limit_sim` in this matrix.")
    lines.append("5. Contact-force variation remains a physics fidelity signal; inspect contact/friction/solver only with controlled comparisons, not policy tuning.")

    lines.extend(
        [
            "",
            "## Next Steps",
            "",
            "- If action or torque saturation grows, inspect policy/action scale/exported actuator limits before touching gains.",
            "- If root-frame signals look inconsistent, run controlled frame diagnostics for `base_ang_vel` and `projected_gravity`.",
            "- If contact force is abnormal, compare contact geometry, friction, and solver behavior without changing formal foot mesh.",
            "- If `pd_torque` and `position` diverge strongly, isolate actuator implementation and timing differences.",
            "- Regenerate this matrix after each narrow sim2sim fix and compare only one variable at a time.",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.steps <= 0:
        print("ERROR: --steps must be positive.", file=sys.stderr)
        return 2
    if args.policy is None:
        print("ERROR: --policy is required and must be an absolute path for policy matrix cases.", file=sys.stderr)
        return 2
    try:
        args.policy = str(require_absolute_path(args.policy, "--policy"))
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    deploy_cfg = load_deploy_cfg(args.deploy_cfg)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        Case(action_kind=action_kind, control_mode=control_mode, command=command)
        for action_kind in ["zero_action", "policy"]
        for control_mode in ["position", "pd_torque"]
        for command in COMMANDS
    ]
    records: list[dict[str, Any]] = []
    for case in cases:
        rollout = output_dir / f"{case.name}.npz"
        check = output_dir / f"{case.name}_check.md"
        run_cmd = [
            sys.executable,
            "scripts/sim2sim/run_g0_mujoco_deploy.py",
            "--model",
            args.model,
            "--deploy-cfg",
            args.deploy_cfg,
            "--steps",
            str(args.steps),
            "--command",
            *(str(value) for value in case.command),
            "--control-mode",
            case.control_mode,
            "--record-rollout",
            str(rollout),
        ]
        if case.action_kind == "zero_action":
            run_cmd.append("--zero-action")
        else:
            run_cmd.extend(["--policy", args.policy, "--device", "cpu"])
        print(f"[matrix] running {case.name}")
        run_code, run_output = run_command(run_cmd)
        (output_dir / f"{case.name}_run.log").write_text(run_output, encoding="utf-8")
        run_ok = run_code == 0

        check_ok = False
        check_output = ""
        metrics: dict[str, Any] = {}
        if run_ok:
            check_cmd = [
                sys.executable,
                "scripts/sim2sim/check_g0_deploy_rollout.py",
                "--rollout",
                str(rollout),
                "--deploy-cfg",
                args.deploy_cfg,
                "--output",
                str(check),
            ]
            check_code, check_output = run_command(check_cmd)
            check_ok = check_code == 0
            (output_dir / f"{case.name}_check.log").write_text(check_output, encoding="utf-8")
            if rollout.exists():
                metrics = rollout_metrics(rollout, deploy_cfg)
        records.append(
            {
                "case": case,
                "rollout": rollout,
                "check": check,
                "run_ok": run_ok,
                "check_ok": check_ok,
                "run_output_tail": run_output[-2000:],
                "check_output_tail": check_output[-2000:],
                "metrics": metrics,
            }
        )
        print(f"[matrix] {case.name}: run_ok={run_ok} check_ok={check_ok}")

    report = Path(args.report)
    write_report(report, records, args)
    write_report(output_dir / "validation_matrix_summary.md", records, args)
    all_ok = all(record["run_ok"] and record["check_ok"] for record in records)
    print(f"Saved validation matrix report: {report}")
    print(f"Saved output-dir summary: {output_dir / 'validation_matrix_summary.md'}")
    print(f"Validation matrix {'OK' if all_ok else 'FAILED'}: {sum(1 for r in records if r['run_ok'] and r['check_ok'])}/{len(records)} cases OK")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
