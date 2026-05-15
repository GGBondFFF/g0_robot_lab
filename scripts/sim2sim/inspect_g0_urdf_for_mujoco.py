#!/usr/bin/env python3
"""Inspect the G0 URDF for MuJoCo migration readiness.

The script is intentionally read-only. It checks joint/link structure, mesh
paths, inertial tags, and whether the URDF movable joints cover the policy
joint order used by the Isaac Lab task.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts.sim2sim import g0_sim2sim_config as cfg
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.sim2sim import g0_sim2sim_config as cfg


@dataclass
class JointInfo:
    name: str
    joint_type: str
    parent: str
    child: str
    axis: str | None
    has_limit: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--urdf",
        default="source/g0_robot_lab/g0_robot_lab/assets/robots/g0/urdf/g0.urdf",
        help="Path to the G0 URDF.",
    )
    return parser.parse_args()


def _child_text(element: ET.Element, tag: str, attr: str | None = None) -> str | None:
    child = element.find(tag)
    if child is None:
        return None
    if attr is None:
        return child.text
    return child.attrib.get(attr)


def inspect_urdf(urdf_path: Path) -> dict[str, object]:
    if not urdf_path.exists():
        raise FileNotFoundError(f"URDF does not exist: {urdf_path}")
    root = ET.parse(urdf_path).getroot()
    urdf_dir = urdf_path.parent
    links = root.findall("link")
    joints = root.findall("joint")
    movable_types = {"revolute", "continuous", "prismatic"}

    joint_infos: list[JointInfo] = []
    for joint in joints:
        parent = _child_text(joint, "parent", "link") or ""
        child = _child_text(joint, "child", "link") or ""
        axis = _child_text(joint, "axis", "xyz")
        joint_infos.append(
            JointInfo(
                name=joint.attrib.get("name", ""),
                joint_type=joint.attrib.get("type", ""),
                parent=parent,
                child=child,
                axis=axis,
                has_limit=joint.find("limit") is not None,
            )
        )

    movable_joints = [joint for joint in joint_infos if joint.joint_type in movable_types]
    movable_names = [joint.name for joint in movable_joints]
    policy_names = cfg.get_joint_names()
    link_names = [link.attrib.get("name", "") for link in links]
    foot_links = [name for name in link_names if "foot" in name]

    mesh_paths: list[Path] = []
    missing_meshes: list[str] = []
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        path = (urdf_dir / filename).resolve()
        mesh_paths.append(path)
        if not path.exists():
            missing_meshes.append(filename)

    links_missing_inertial = [link.attrib.get("name", "") for link in links if link.find("inertial") is None]
    inertials_missing_mass = []
    inertials_missing_inertia = []
    for link in links:
        inertial = link.find("inertial")
        if inertial is None:
            continue
        name = link.attrib.get("name", "")
        if inertial.find("mass") is None:
            inertials_missing_mass.append(name)
        if inertial.find("inertia") is None:
            inertials_missing_inertia.append(name)

    movable_without_limit = [joint.name for joint in movable_joints if not joint.has_limit]
    missing_policy_joints = [name for name in policy_names if name not in movable_names]
    extra_movable_joints = [name for name in movable_names if name not in policy_names]
    child_links = {joint.child for joint in joint_infos}
    root_links = [name for name in link_names if name not in child_links]

    return {
        "link_count": len(links),
        "joint_count": len(joints),
        "movable_joint_count": len(movable_joints),
        "movable_joint_names": movable_names,
        "policy_joint_names": policy_names,
        "missing_policy_joints": missing_policy_joints,
        "extra_movable_joints": extra_movable_joints,
        "foot_links": foot_links,
        "root_links": root_links,
        "missing_meshes": missing_meshes,
        "mesh_count": len(mesh_paths),
        "links_missing_inertial": links_missing_inertial,
        "inertials_missing_mass": inertials_missing_mass,
        "inertials_missing_inertia": inertials_missing_inertia,
        "movable_without_limit": movable_without_limit,
        "joints": joint_infos,
    }


def print_report(report: dict[str, object]) -> None:
    print("G0 URDF MuJoCo migration inspection")
    print(f"link_count: {report['link_count']}")
    print(f"joint_count: {report['joint_count']}")
    print(f"movable_joint_count: {report['movable_joint_count']}")
    print(f"mesh_count: {report['mesh_count']}")
    print(f"root_links: {report['root_links']}")
    print(f"foot_links: {report['foot_links']}")
    print("")
    print("movable_joint_names:")
    for name in report["movable_joint_names"]:
        print(f"  - {name}")
    print("")
    print(f"missing_policy_joints: {report['missing_policy_joints']}")
    print(f"extra_movable_joints: {report['extra_movable_joints']}")
    print(f"missing_meshes: {report['missing_meshes']}")
    print(f"links_missing_inertial: {report['links_missing_inertial']}")
    print(f"inertials_missing_mass: {report['inertials_missing_mass']}")
    print(f"inertials_missing_inertia: {report['inertials_missing_inertia']}")
    print(f"movable_without_limit: {report['movable_without_limit']}")
    print("")
    print("joint_structure:")
    for joint in report["joints"]:
        print(
            f"  - {joint.name} [{joint.joint_type}] "
            f"{joint.parent} -> {joint.child}, axis={joint.axis}, has_limit={joint.has_limit}"
        )
    print("")
    print("suggested MuJoCo conversion warnings:")
    print("  - URDF mesh collisions may need simplified foot contact geoms in MJCF.")
    print("  - Actuator PD, torque limits, velocity limits, damping, and armature are not represented by URDF alone.")
    print("  - Confirm quaternion and base-frame conventions before comparing projected gravity or base angular velocity.")
    print("  - Verify mass/inertia values against the expected hardware mass before trusting dynamics.")


def main() -> int:
    args = parse_args()
    report = inspect_urdf(Path(args.urdf))
    print_report(report)
    if report["missing_policy_joints"] or report["missing_meshes"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

