#!/usr/bin/env python3
"""Export an Isaac-vs-MuJoCo actuator alignment table for G0.

The script reads Isaac actuator parameters from the source ``G0_CFG`` through
``g0_sim2sim_config`` and reads the current MuJoCo parameters from ``g0.xml``.
It does not import Isaac Sim, so it can run in a normal Python process.
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from scripts.sim2sim import g0_sim2sim_config as cfg
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.sim2sim import g0_sim2sim_config as cfg


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "mujoco" / "g0.xml"
DEFAULT_OUTPUT = REPO_ROOT / "logs" / "sim2sim" / "g0_actuator_alignment_table.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="MuJoCo XML path to inspect.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Markdown table output path.")
    return parser.parse_args()


def _float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _float_pair(value: str | None) -> tuple[float, float] | None:
    if value is None:
        return None
    parts = value.split()
    if len(parts) != 2:
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        return None


def _fmt(value: float | str | None) -> str:
    if value is None:
        return "missing"
    if isinstance(value, str):
        return value
    return f"{value:.10g}"


def _close(left: float | None, right: float, *, atol: float = 1e-9) -> bool:
    return left is not None and math.isclose(left, right, rel_tol=0.0, abs_tol=atol)


def _status(
    spec: cfg.IsaacActuatorSpec,
    actuator: ET.Element | None,
    joint: ET.Element | None,
) -> str:
    problems: list[str] = []
    if actuator is None:
        problems.append("missing_actuator")
    else:
        if not _close(_float(actuator.attrib.get("kp")), spec.stiffness):
            problems.append("kp")
        force_range = _float_pair(actuator.attrib.get("forcerange"))
        if force_range is None:
            problems.append("forcerange")
        elif not (
            math.isclose(force_range[0], -spec.effort_limit_sim, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(force_range[1], spec.effort_limit_sim, rel_tol=0.0, abs_tol=1e-9)
        ):
            problems.append("forcerange")
    if joint is None:
        problems.append("missing_joint")
    else:
        if not _close(_float(joint.attrib.get("damping")), spec.damping):
            problems.append("joint_damping")
        if not _close(_float(joint.attrib.get("armature")), spec.armature):
            problems.append("joint_armature")
    return "aligned" if not problems else "check:" + ",".join(problems)


def build_report(model_path: Path) -> str:
    root = ET.parse(model_path).getroot()
    actuators = {actuator.attrib.get("name"): actuator for actuator in root.findall(".//actuator/position")}
    joints = {joint.attrib.get("name"): joint for joint in root.findall(".//joint")}
    specs = cfg.get_isaac_actuator_specs()

    headers = [
        "joint_name",
        "servo_type",
        "Isaac stiffness",
        "Isaac damping",
        "Isaac effort_limit_sim",
        "Isaac velocity_limit_sim",
        "Isaac armature",
        "MuJoCo kp",
        "MuJoCo forcerange",
        "MuJoCo joint damping",
        "MuJoCo joint armature",
        "MuJoCo ctrlrange",
        "status",
    ]
    rows = ["# G0 Actuator Alignment Table", "", f"- MuJoCo model: `{model_path}`", ""]
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    aligned = 0
    for joint_name, spec in specs.items():
        actuator = actuators.get(joint_name)
        joint = joints.get(joint_name)
        status = _status(spec, actuator, joint)
        aligned += int(status == "aligned")
        rows.append(
            "| "
            + " | ".join(
                [
                    joint_name,
                    spec.servo_type,
                    _fmt(spec.stiffness),
                    _fmt(spec.damping),
                    _fmt(spec.effort_limit_sim),
                    _fmt(spec.velocity_limit_sim),
                    _fmt(spec.armature),
                    _fmt(_float(actuator.attrib.get("kp")) if actuator is not None else None),
                    _fmt(actuator.attrib.get("forcerange") if actuator is not None else None),
                    _fmt(_float(joint.attrib.get("damping")) if joint is not None else None),
                    _fmt(_float(joint.attrib.get("armature")) if joint is not None else None),
                    _fmt(actuator.attrib.get("ctrlrange") if actuator is not None else None),
                    status,
                ]
            )
            + " |"
        )
    rows.extend(
        [
            "",
            "## Summary",
            "",
            f"- aligned rows: `{aligned}/{len(specs)}`",
            "- `velocity_limit_sim` is recorded for tracking, but the current MJCF position actuator mapping does not enforce an exactly equivalent PhysX velocity limit.",
            "- `damping` is mapped to MJCF joint damping as a first-pass approximation of Isaac implicit actuator damping.",
        ]
    )
    return "\n".join(rows) + "\n"


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    output_path = Path(args.output)
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model does not exist: {model_path}")
    report = build_report(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote actuator alignment table: {output_path}")
    print(report.split("## Summary", maxsplit=1)[-1].strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
