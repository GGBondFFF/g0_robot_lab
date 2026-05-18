#!/usr/bin/env python3
"""Generate a Unitree-mujoco-style G0 MJCF from the current URDF-derived model."""

from __future__ import annotations

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sim2sim import g0_sim2sim_config as cfg  # noqa: E402


DEFAULT_SOURCE = REPO_ROOT / "mujoco" / "g0.xml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "mujoco" / "unitree_mujoco_g0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Source URDF-derived G0 MJCF.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for scene.xml and g0.xml.")
    return parser.parse_args()


def _find_body(root: ET.Element, name: str) -> ET.Element | None:
    for body in root.findall(".//body"):
        if body.attrib.get("name") == name:
            return body
    return None


def _ensure_imu_site(root: ET.Element) -> None:
    base = _find_body(root, "base_link")
    if base is None:
        raise RuntimeError("Could not find base_link body for IMU site")
    for site in base.findall("site"):
        if site.attrib.get("name") == "imu":
            return
    ET.SubElement(
        base,
        "site",
        {
            "name": "imu",
            "pos": "0 0 0",
            "quat": "1 0 0 0",
            "size": "0.01",
            "rgba": "0 1 0 0.35",
        },
    )


def _rewrite_mesh_paths(root: ET.Element, source_dir: Path, output_dir: Path) -> None:
    for mesh in root.findall(".//mesh"):
        file_name = mesh.attrib.get("file")
        if file_name is None:
            continue
        mesh_path = (source_dir / file_name).resolve()
        mesh.attrib["file"] = os.path.relpath(mesh_path, output_dir.resolve())


def _replace_actuators(root: ET.Element) -> None:
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    actuator.clear()
    actuator.append(ET.Comment("Unitree-style torque motors in G0_JOINT_SDK_NAMES order."))
    specs = cfg.get_isaac_actuator_specs()
    for name in cfg.get_joint_names():
        limit = specs[name].effort_limit_sim
        ET.SubElement(
            actuator,
            "motor",
            {
                "name": name,
                "joint": name,
                "gear": "1",
                "forcelimited": "true",
                "forcerange": f"{-limit:.12g} {limit:.12g}",
            },
        )


def _replace_sensors(root: ET.Element) -> None:
    for old in list(root.findall("sensor")):
        root.remove(old)
    sensor = ET.SubElement(root, "sensor")
    sensor.append(ET.Comment("Sensor order mirrors unitree_mujoco: q, dq, tau_est for all motors, then IMU/frame."))
    for name in cfg.get_joint_names():
        ET.SubElement(sensor, "jointpos", {"name": f"{name}_pos", "joint": name})
    for name in cfg.get_joint_names():
        ET.SubElement(sensor, "jointvel", {"name": f"{name}_vel", "joint": name})
    for name in cfg.get_joint_names():
        ET.SubElement(sensor, "jointactuatorfrc", {"name": f"{name}_torque", "joint": name})
    ET.SubElement(sensor, "framequat", {"name": "imu_quat", "objtype": "site", "objname": "imu"})
    ET.SubElement(sensor, "gyro", {"name": "imu_gyro", "site": "imu"})
    ET.SubElement(sensor, "accelerometer", {"name": "imu_acc", "site": "imu"})
    ET.SubElement(sensor, "framepos", {"name": "frame_pos", "objtype": "site", "objname": "imu"})
    ET.SubElement(sensor, "framelinvel", {"name": "frame_vel", "objtype": "site", "objname": "imu"})


def _write_scene(output_dir: Path) -> None:
    scene = ET.Element("mujoco", {"model": "g0 unitree-style scene"})
    ET.SubElement(scene, "include", {"file": "g0.xml"})
    ET.SubElement(scene, "statistic", {"center": "0 0 0.12", "extent": "0.7"})
    visual = ET.SubElement(scene, "visual")
    ET.SubElement(visual, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"})
    ET.SubElement(visual, "global", {"azimuth": "-130", "elevation": "-20"})
    tree = ET.ElementTree(scene)
    ET.indent(tree, space="  ")
    tree.write(output_dir / "scene.xml", encoding="utf-8", xml_declaration=False)


def generate(source: Path, output_dir: Path) -> tuple[Path, Path]:
    if not source.exists():
        raise FileNotFoundError(f"Source MJCF does not exist: {source}")
    output_dir.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(source)
    root = tree.getroot()
    root.attrib["model"] = "g0_unitree_mujoco_style"
    _rewrite_mesh_paths(root, source.parent.resolve(), output_dir.resolve())
    _ensure_imu_site(root)
    _replace_actuators(root)
    _replace_sensors(root)
    ET.indent(tree, space="  ")
    model_path = output_dir / "g0.xml"
    tree.write(model_path, encoding="utf-8", xml_declaration=True)
    _write_scene(output_dir)
    return model_path, output_dir / "scene.xml"


def main() -> int:
    args = parse_args()
    model_path, scene_path = generate(Path(args.source), Path(args.output_dir))
    print(f"Wrote Unitree-style G0 model: {model_path}")
    print(f"Wrote Unitree-style G0 scene: {scene_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
