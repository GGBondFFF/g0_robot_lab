#!/usr/bin/env python3
"""Inspect the G0 URDF for MuJoCo migration readiness.

The script is intentionally read-only. It checks joint/link structure, mesh
paths, inertial tags, and whether the URDF movable joints cover the policy
joint order used by the Isaac Lab task.
"""

from __future__ import annotations

import argparse
import struct
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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


@dataclass
class CollisionInfo:
    link_name: str
    kind: str
    origin_xyz: str
    origin_rpy: str
    mesh_filename: str | None
    mesh_path: Path | None
    mesh_exists: bool
    visual_mesh_same: bool | None
    bbox_min: tuple[float, float, float] | None
    bbox_max: tuple[float, float, float] | None
    vertex_count: int | None
    face_count: int | None


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


def _read_stl_mesh_stats(path: Path) -> tuple[tuple[float, float, float], tuple[float, float, float], int, int] | None:
    """Read basic STL bbox and counts for binary or ASCII STL."""

    if not path.exists():
        return None
    data = path.read_bytes()
    vertices: list[tuple[float, float, float]] = []
    if len(data) >= 84:
        tri_count = struct.unpack("<I", data[80:84])[0]
        expected_size = 84 + tri_count * 50
        if expected_size == len(data):
            offset = 84
            for _ in range(tri_count):
                offset += 12
                for _vertex in range(3):
                    vertices.append(struct.unpack("<fff", data[offset : offset + 12]))
                    offset += 12
                offset += 2
            arr = np.asarray(vertices, dtype=np.float64)
            return tuple(arr.min(axis=0)), tuple(arr.max(axis=0)), int(arr.shape[0]), int(tri_count)

    text = data.decode("utf-8", errors="ignore")
    face_count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("facet normal"):
            face_count += 1
        if stripped.startswith("vertex "):
            parts = stripped.split()
            if len(parts) == 4:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not vertices:
        return None
    arr = np.asarray(vertices, dtype=np.float64)
    return tuple(arr.min(axis=0)), tuple(arr.max(axis=0)), int(arr.shape[0]), int(face_count)


def _mesh_filename_from_geometry(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    mesh = element.find("./geometry/mesh")
    if mesh is not None:
        return mesh.attrib.get("filename")
    return None


def inspect_link_collisions(link: ET.Element, urdf_dir: Path) -> list[CollisionInfo]:
    link_name = link.attrib.get("name", "")
    visual_meshes = [_mesh_filename_from_geometry(visual) for visual in link.findall("visual")]
    collisions: list[CollisionInfo] = []
    for collision in link.findall("collision"):
        origin = collision.find("origin")
        geometry = collision.find("geometry")
        mesh_filename = _mesh_filename_from_geometry(collision)
        kind = "unknown"
        if geometry is not None and len(list(geometry)):
            kind = list(geometry)[0].tag
        mesh_path = (urdf_dir / mesh_filename).resolve() if mesh_filename else None
        stats = _read_stl_mesh_stats(mesh_path) if mesh_path else None
        collisions.append(
            CollisionInfo(
                link_name=link_name,
                kind=kind,
                origin_xyz="0 0 0" if origin is None else origin.attrib.get("xyz", "0 0 0"),
                origin_rpy="0 0 0" if origin is None else origin.attrib.get("rpy", "0 0 0"),
                mesh_filename=mesh_filename,
                mesh_path=mesh_path,
                mesh_exists=bool(mesh_path and mesh_path.exists()),
                visual_mesh_same=(mesh_filename in visual_meshes) if mesh_filename else None,
                bbox_min=None if stats is None else stats[0],
                bbox_max=None if stats is None else stats[1],
                vertex_count=None if stats is None else stats[2],
                face_count=None if stats is None else stats[3],
            )
        )
    return collisions


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
    foot_collision_info: dict[str, list[CollisionInfo]] = {}

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
    for link in links:
        name = link.attrib.get("name", "")
        if name in {"l_foot_link", "r_foot_link"}:
            foot_collision_info[name] = inspect_link_collisions(link, urdf_dir)

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
        "foot_collision_info": foot_collision_info,
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
    print("foot_collision_summary:")
    for link_name, collisions in report["foot_collision_info"].items():
        print(f"  {link_name}: collision_count={len(collisions)}")
        for index, collision in enumerate(collisions):
            print(f"    collision[{index}].type: {collision.kind}")
            print(f"    collision[{index}].origin_xyz: {collision.origin_xyz}")
            print(f"    collision[{index}].origin_rpy: {collision.origin_rpy}")
            print(f"    collision[{index}].mesh_filename: {collision.mesh_filename}")
            print(f"    collision[{index}].mesh_path: {collision.mesh_path}")
            print(f"    collision[{index}].mesh_exists: {collision.mesh_exists}")
            print(f"    collision[{index}].visual_mesh_same: {collision.visual_mesh_same}")
            print(f"    collision[{index}].bbox_min: {collision.bbox_min}")
            print(f"    collision[{index}].bbox_max: {collision.bbox_max}")
            print(f"    collision[{index}].vertex_count: {collision.vertex_count}")
            print(f"    collision[{index}].face_count: {collision.face_count}")
    print("")
    print("joint_structure:")
    for joint in report["joints"]:
        print(
            f"  - {joint.name} [{joint.joint_type}] "
            f"{joint.parent} -> {joint.child}, axis={joint.axis}, has_limit={joint.has_limit}"
        )
    print("")
    print("suggested MuJoCo conversion warnings:")
    print("  - URDF foot collisions are mesh-based; do not replace them in the formal MJCF without evidence.")
    print("  - If simplified foot geoms are needed for diagnosis, keep them in a separate debug-only model.")
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
