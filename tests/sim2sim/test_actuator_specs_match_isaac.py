from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.sim2sim import g0_sim2sim_config as cfg


def _pair(value: str) -> tuple[float, float]:
    left, right = value.split()
    return float(left), float(right)


def test_mujoco_actuator_specs_match_isaac_first_pass_mapping() -> None:
    root = ET.parse(Path("mujoco/g0.xml")).getroot()
    actuators = {element.attrib["name"]: element for element in root.findall(".//actuator/position")}
    joints = {element.attrib["name"]: element for element in root.findall(".//joint")}

    for joint_name, spec in cfg.get_isaac_actuator_specs().items():
        actuator = actuators[joint_name]
        joint = joints[joint_name]
        force_min, force_max = _pair(actuator.attrib["forcerange"])
        assert math.isclose(float(actuator.attrib["kp"]), spec.stiffness, abs_tol=1e-12)
        assert math.isclose(force_min, -spec.effort_limit_sim, abs_tol=1e-9)
        assert math.isclose(force_max, spec.effort_limit_sim, abs_tol=1e-9)
        assert math.isclose(float(joint.attrib["damping"]), spec.damping, abs_tol=1e-12)
        assert math.isclose(float(joint.attrib["armature"]), spec.armature, abs_tol=1e-12)
