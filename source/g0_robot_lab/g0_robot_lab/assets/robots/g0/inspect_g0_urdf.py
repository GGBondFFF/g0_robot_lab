#!/usr/bin/env python3
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


ASSET_ROOT = Path(__file__).resolve().parent
URDF_PATH = ASSET_ROOT / "urdf" / "g0.urdf"


def has_cjk(text: str) -> bool:
    return any(
        "\u3400" <= char <= "\u4dbf"
        or "\u4e00" <= char <= "\u9fff"
        or "\uf900" <= char <= "\ufaff"
        for char in text
    )


def duplicates(values: list[str]) -> list[str]:
    return sorted(name for name, count in Counter(values).items() if count > 1)


def main() -> int:
    failures: list[str] = []

    if not URDF_PATH.is_file():
        print(f"Missing URDF: {URDF_PATH}")
        return 1

    urdf_text = URDF_PATH.read_text(encoding="utf-8")

    try:
        root = ET.fromstring(urdf_text)
    except ET.ParseError as exc:
        print(f"XML parse failed: {exc}")
        return 1

    robot_name = root.attrib.get("name", "")
    links = root.findall("link")
    joints = root.findall("joint")
    link_names = [link.attrib.get("name", "") for link in links]
    joint_names = [joint.attrib.get("name", "") for joint in joints]
    mesh_filenames = [
        mesh.attrib.get("filename", "")
        for mesh in root.findall(".//mesh")
        if mesh.attrib.get("filename")
    ]

    print(f"robot name: {robot_name}")
    print(f"link count: {len(links)}")
    print(f"joint count: {len(joints)}")

    print("links:")
    for name in link_names:
        print(f"  - {name}")

    print("joints:")
    for name in joint_names:
        print(f"  - {name}")

    print("joint details:")
    for joint in joints:
        name = joint.attrib.get("name", "")
        joint_type = joint.attrib.get("type", "")
        parent = joint.find("parent")
        child = joint.find("child")
        axis = joint.find("axis")
        limit = joint.find("limit")
        parent_link = parent.attrib.get("link", "") if parent is not None else ""
        child_link = child.attrib.get("link", "") if child is not None else ""
        axis_xyz = axis.attrib.get("xyz", "") if axis is not None else ""
        limit_attrs = dict(limit.attrib) if limit is not None else {}
        print(
            f"  - {name}: type={joint_type}, parent={parent_link}, "
            f"child={child_link}, axis={axis_xyz}, limit={limit_attrs}"
        )

    print("mesh filenames:")
    for filename in mesh_filenames:
        print(f"  - {filename}")

    if root.tag != "robot":
        failures.append(f"root tag is {root.tag!r}, expected 'robot'")
    if robot_name != "g0":
        failures.append(f"robot name is {robot_name!r}, expected 'g0'")
    if "package://" in urdf_text:
        failures.append("URDF still contains package://")
    if has_cjk(urdf_text):
        failures.append("URDF still contains Chinese/CJK characters")

    for bad_name in duplicates(link_names):
        failures.append(f"duplicate link name: {bad_name}")
    for bad_name in duplicates(joint_names):
        failures.append(f"duplicate joint name: {bad_name}")

    for filename in mesh_filenames:
        if "package://" in filename:
            failures.append(f"mesh path still uses package://: {filename}")
            continue
        mesh_path = (URDF_PATH.parent / filename).resolve()
        if not mesh_path.is_file():
            failures.append(f"missing mesh file: {filename} -> {mesh_path}")

    for path in ASSET_ROOT.rglob("*"):
        if "__MACOSX" in path.parts:
            failures.append(f"unexpected __MACOSX path: {path}")
        if path.name.startswith("._"):
            failures.append(f"unexpected macOS resource fork file: {path}")
        if path.name == ".DS_Store":
            failures.append(f"unexpected .DS_Store file: {path}")

    if failures:
        print("G0 URDF inspection failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("G0 URDF inspection passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
