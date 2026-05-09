#!/usr/bin/env python3

from pathlib import Path
import xml.etree.ElementTree as ET

URDF_PATH = Path(
    "source/g0_robot_lab/g0_robot_lab/assets/robots/g0/urdf/g0.urdf"
)

tree = ET.parse(URDF_PATH)
root = tree.getroot()

print(f"URDF: {URDF_PATH}")
print("=" * 80)

movable_joints = []

for joint in root.findall("joint"):
    name = joint.attrib.get("name", "")
    joint_type = joint.attrib.get("type", "")

    if joint_type == "fixed":
        continue

    parent = joint.find("parent")
    child = joint.find("child")
    origin = joint.find("origin")
    axis = joint.find("axis")
    limit = joint.find("limit")

    parent_link = parent.attrib.get("link", "") if parent is not None else ""
    child_link = child.attrib.get("link", "") if child is not None else ""
    xyz = origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0"
    rpy = origin.attrib.get("rpy", "0 0 0") if origin is not None else "0 0 0"
    axis_xyz = axis.attrib.get("xyz", "") if axis is not None else ""

    lower = limit.attrib.get("lower", "") if limit is not None else ""
    upper = limit.attrib.get("upper", "") if limit is not None else ""
    effort = limit.attrib.get("effort", "") if limit is not None else ""
    velocity = limit.attrib.get("velocity", "") if limit is not None else ""

    movable_joints.append(name)

    print(f"[{len(movable_joints) - 1:02d}] {name}")
    print(f"     type   : {joint_type}")
    print(f"     parent : {parent_link}")
    print(f"     child  : {child_link}")
    print(f"     origin : xyz={xyz}, rpy={rpy}")
    print(f"     axis   : {axis_xyz}")
    print(f"     limit  : lower={lower}, upper={upper}, effort={effort}, velocity={velocity}")
    print()

print("=" * 80)
print(f"Movable joint count: {len(movable_joints)}")
print()
print("Python list:")
print("[")
for name in movable_joints:
    print(f'    "{name}",')
print("]")
