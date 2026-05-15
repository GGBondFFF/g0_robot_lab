#!/usr/bin/env python3
"""Generate a first URDF-derived MuJoCo G0 working model.

The generated model keeps the URDF-derived body tree, joint axes, limits,
inertials, and mesh geoms emitted by MuJoCo's URDF compiler. It then adds the
sim2sim-specific pieces needed for policy validation:

- a floating root body around the URDF tree
- a ground plane
- 22 position actuators named exactly like ``G0_JOINT_SDK_NAMES``
- actuator stiffness, effort limits, joint damping, and armature resolved from
  the Isaac Lab ``G0_CFG`` source
- mesh paths rewritten relative to ``mujoco/g0.xml``
- collision filtering that preserves robot-ground contact while disabling
  robot internal self-collision, matching Isaac Lab ``enabled_self_collisions=False``

This is still not a final dynamics model. Velocity limits, contact/friction,
solver settings, and mass/inertia sanity checks must be aligned against the
Isaac Lab baseline in later passes.
"""

from __future__ import annotations

import argparse
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
DEFAULT_URDF = REPO_ROOT / "source" / "g0_robot_lab" / "g0_robot_lab" / "assets" / "robots" / "g0" / "urdf" / "g0.urdf"
DEFAULT_OUTPUT = REPO_ROOT / "mujoco" / "g0.xml"
DEFAULT_INTERMEDIATE = REPO_ROOT / "mujoco" / "generated" / "g0_from_urdf_compiled.xml"
MESH_PREFIX = "../source/g0_robot_lab/g0_robot_lab/assets/robots/g0/meshes"


def _fmt(value: float) -> str:
    return f"{float(value):.10g}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", default=str(DEFAULT_URDF), help="Input G0 URDF path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output MJCF path.")
    parser.add_argument("--intermediate", default=str(DEFAULT_INTERMEDIATE), help="Generated intermediate MJCF path.")
    return parser.parse_args()


def indent(element: ET.Element, level: int = 0) -> None:
    """Pretty-print an ElementTree in-place."""

    spacer = "\n" + level * "  "
    if len(element):
        if not element.text or not element.text.strip():
            element.text = spacer + "  "
        for child in element:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = spacer
    if level and (not element.tail or not element.tail.strip()):
        element.tail = spacer


def compile_urdf_to_mjcf(urdf_path: Path, intermediate_path: Path) -> ET.Element:
    """Use MuJoCo's URDF compiler and return the emitted MJCF root."""

    try:
        import mujoco
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("DeepMind MuJoCo Python package is required to generate g0.xml.") from exc

    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    intermediate_path.parent.mkdir(parents=True, exist_ok=True)
    mujoco.mj_saveLastXML(str(intermediate_path), model)
    return ET.parse(intermediate_path).getroot()


def rewrite_mesh_paths(root: ET.Element) -> None:
    """Rewrite generated mesh paths so they resolve from ``mujoco/g0.xml``."""

    for mesh in root.findall(".//mesh"):
        file_name = Path(mesh.attrib["file"]).name
        mesh.attrib["file"] = f"{MESH_PREFIX}/{file_name}"


def add_sim_options(root: ET.Element) -> None:
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.attrib["timestep"] = str(cfg.ISAAC_SIM_DT)
    option.attrib["gravity"] = "0 0 -9.81"


def wrap_with_floating_root(root: ET.Element) -> None:
    """Wrap URDF-generated world bodies in a floating root body."""

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Generated MJCF does not contain worldbody.")

    original_children = list(worldbody)
    worldbody.clear()
    ET.SubElement(
        worldbody,
        "geom",
        name="ground",
        type="plane",
        size="2 2 0.05",
        rgba="0.3 0.35 0.35 1",
        contype="1",
        conaffinity="2",
    )
    base_body = ET.SubElement(worldbody, "body", name="base_link", pos="0 0 0.23")
    ET.SubElement(base_body, "freejoint", name="root")
    base_body.append(
        ET.Comment(
            "URDF-derived tree follows. The free root and ground are sim2sim additions, "
            "not original URDF tags."
        )
    )
    for child in original_children:
        base_body.append(child)


def apply_isaac_style_collision_filter(root: ET.Element) -> None:
    """Disable robot-robot self-collision while preserving robot-ground contact.

    MuJoCo generates contacts when
    ``(contype1 & conaffinity2) || (contype2 & conaffinity1)`` is non-zero.

    We set:

    - ground: ``contype=1``, ``conaffinity=2``
    - robot geoms: ``contype=2``, ``conaffinity=1``

    This keeps ground-robot contacts enabled:

    ``(1 & 1) || (2 & 2) != 0``

    and disables robot-robot contacts:

    ``(2 & 1) || (2 & 1) == 0``

    Foot mesh geoms remain mesh geoms. This function does not add boxes,
    capsules, or other debug-only contact shapes.
    """

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("MJCF does not contain worldbody.")
    for geom in worldbody.iter("geom"):
        if geom.attrib.get("name") == "ground":
            geom.attrib["contype"] = "1"
            geom.attrib["conaffinity"] = "2"
        else:
            geom.attrib["contype"] = "2"
            geom.attrib["conaffinity"] = "1"


def apply_isaac_actuator_joint_params(root: ET.Element) -> None:
    """Apply Isaac G0 actuator damping and armature to MuJoCo joints.

    Isaac's ``ImplicitActuatorCfg`` damping is an implicit drive damping term.
    The first MuJoCo alignment maps it to MJCF joint ``damping`` as a
    conservative approximation so the parameter is present in the dynamics
    model and visible in reports. This is not yet a proof of full PhysX/MuJoCo
    drive equivalence.
    """

    specs = cfg.get_isaac_actuator_specs()
    joint_elements = {joint.attrib.get("name"): joint for joint in root.findall(".//joint")}
    missing = [name for name in cfg.get_joint_names() if name not in joint_elements]
    if missing:
        raise RuntimeError(f"Generated MJCF is missing policy joints: {missing}")
    for joint_name, spec in specs.items():
        joint = joint_elements[joint_name]
        joint.attrib["damping"] = _fmt(spec.damping)
        joint.attrib["armature"] = _fmt(spec.armature)


def _joint_ctrlrange(root: ET.Element, joint_name: str) -> str:
    joint = root.find(f".//joint[@name='{joint_name}']")
    if joint is not None and "range" in joint.attrib:
        return joint.attrib["range"]
    return "-3.14 3.14"


def add_actuators(root: ET.Element) -> None:
    old_actuator = root.find("actuator")
    if old_actuator is not None:
        root.remove(old_actuator)
    actuator = ET.SubElement(root, "actuator")
    actuator.append(
        ET.Comment(
            "Position actuators aligned to Isaac G0_CFG stiffness and effort_limit_sim. "
            "Velocity limits are documented separately; MJCF position actuators do not "
            "make this a fully equivalent PhysX implicit drive model by themselves."
        )
    )
    for name, spec in cfg.get_isaac_actuator_specs().items():
        ET.SubElement(
            actuator,
            "position",
            name=name,
            joint=name,
            kp=_fmt(spec.stiffness),
            ctrllimited="true",
            ctrlrange=_joint_ctrlrange(root, name),
            forcelimited="true",
            forcerange=f"{_fmt(-spec.effort_limit_sim)} {_fmt(spec.effort_limit_sim)}",
        )


def add_header(root: ET.Element) -> None:
    root.attrib["model"] = "g0_urdf_derived_working"
    root.insert(
        0,
        ET.Comment(
            "URDF-derived working MJCF for sim2sim validation. Not final dynamics: "
            "velocity limits, contact/friction, solver settings, and inertias need verification."
        ),
    )


def generate(urdf_path: Path, output_path: Path, intermediate_path: Path) -> None:
    root = compile_urdf_to_mjcf(urdf_path, intermediate_path)
    add_header(root)
    add_sim_options(root)
    rewrite_mesh_paths(root)
    wrap_with_floating_root(root)
    apply_isaac_style_collision_filter(root)
    apply_isaac_actuator_joint_params(root)
    add_actuators(root)
    indent(root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=False)


def main() -> int:
    args = parse_args()
    generate(Path(args.urdf), Path(args.output), Path(args.intermediate))
    print(f"Wrote MuJoCo working model: {args.output}")
    print(f"Wrote intermediate compiler output: {args.intermediate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
