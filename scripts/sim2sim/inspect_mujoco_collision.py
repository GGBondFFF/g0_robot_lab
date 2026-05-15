#!/usr/bin/env python3
"""Inspect MuJoCo foot collision geoms and initial contacts for G0."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

try:
    from scripts.sim2sim import g0_sim2sim_config as cfg
    from scripts.sim2sim.g0_mujoco_interface import G0MuJoCoInterface, import_mujoco
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from scripts.sim2sim import g0_sim2sim_config as cfg
    from scripts.sim2sim.g0_mujoco_interface import G0MuJoCoInterface, import_mujoco


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mujoco/g0.xml", help="MuJoCo XML path.")
    parser.add_argument("--steps", type=int, default=1, help="Number of diagnostic simulation steps after reset.")
    return parser.parse_args()


def _xml_body_geoms(model_path: Path) -> dict[str, list[dict[str, str | None]]]:
    root = ET.parse(model_path).getroot()
    result: dict[str, list[dict[str, str | None]]] = {"l_foot_link": [], "r_foot_link": []}

    def visit(body: ET.Element, stack: list[str]) -> None:
        name = body.attrib.get("name", "")
        new_stack = [*stack, name]
        for foot in result:
            if foot in new_stack:
                for geom in body.findall("geom"):
                    result[foot].append(
                        {
                            "body": name,
                            "name": geom.attrib.get("name"),
                            "type": geom.attrib.get("type"),
                            "mesh": geom.attrib.get("mesh"),
                            "friction": geom.attrib.get("friction"),
                            "contype": geom.attrib.get("contype"),
                            "conaffinity": geom.attrib.get("conaffinity"),
                            "solref": geom.attrib.get("solref"),
                            "solimp": geom.attrib.get("solimp"),
                        }
                    )
        for child in body.findall("body"):
            visit(child, new_stack)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for body in worldbody.findall("body"):
            visit(body, [])
    return result


def _name(mujoco, model, obj_type, obj_id: int) -> str:
    if obj_id < 0:
        return "<none>"
    value = mujoco.mj_id2name(model, obj_type, int(obj_id))
    return "<unnamed>" if value is None else value


def _body_ancestors(mujoco, model, body_id: int) -> list[str]:
    names: list[str] = []
    current = int(body_id)
    while current > 0:
        names.append(_name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, current))
        current = int(model.body_parentid[current])
    return names


def _is_foot_geom(mujoco, model, geom_id: int) -> bool:
    ancestors = _body_ancestors(mujoco, model, int(model.geom_bodyid[geom_id]))
    return "l_foot_link" in ancestors or "r_foot_link" in ancestors


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model does not exist: {model_path}")

    mujoco = import_mujoco()
    interface = G0MuJoCoInterface(model_path)
    model = interface.model
    data = interface.data

    print("MuJoCo collision inspection")
    print(f"model: {model_path}")
    print("")

    print("xml_foot_geoms:")
    for foot, geoms in _xml_body_geoms(model_path).items():
        print(f"  {foot}: geom_count={len(geoms)}")
        for geom in geoms:
            print(f"    - {geom}")
    print("")

    print("compiled_foot_geoms:")
    for geom_id in range(model.ngeom):
        if not _is_foot_geom(mujoco, model, geom_id):
            continue
        geom_name = _name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        body_name = _name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id]))
        geom_type = int(model.geom_type[geom_id])
        mesh_id = int(model.geom_dataid[geom_id])
        mesh_name = _name(mujoco, model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) if mesh_id >= 0 else "<none>"
        print(f"  geom_id={geom_id} name={geom_name} body={body_name}")
        print(f"    type_id: {geom_type}")
        print(f"    mesh: {mesh_name}")
        print(f"    friction: {model.geom_friction[geom_id].tolist()}")
        print(f"    contype: {int(model.geom_contype[geom_id])}")
        print(f"    conaffinity: {int(model.geom_conaffinity[geom_id])}")
        print(f"    solref: {model.geom_solref[geom_id].tolist()}")
        print(f"    solimp: {model.geom_solimp[geom_id].tolist()}")
    print("")

    default_action = np.zeros(cfg.get_action_dim(), dtype=np.float64)
    interface.apply_policy_action(default_action)
    mujoco.mj_forward(model, data)
    for _ in range(max(0, args.steps)):
        mujoco.mj_step(model, data)

    print(f"initial_or_step_contacts: ncon={data.ncon}")
    foot_ground = 0
    self_contacts = 0
    negative_dist = 0
    max_force = 0.0
    base_torso_contact = False
    for index in range(data.ncon):
        contact = data.contact[index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        name1 = _name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom1)
        name2 = _name(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, geom2)
        body1 = _name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom1]))
        body2 = _name(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom2]))
        is_ground = name1 == "ground" or name2 == "ground"
        is_foot = _is_foot_geom(mujoco, model, geom1) or _is_foot_geom(mujoco, model, geom2)
        if is_ground and is_foot:
            foot_ground += 1
        if not is_ground:
            self_contacts += 1
        if {body1, body2} == {"base_link", "torso_link"}:
            base_torso_contact = True
        if float(contact.dist) < 0.0:
            negative_dist += 1
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, index, force)
        max_force = max(max_force, float(np.linalg.norm(force[:3])))
        print(
            f"  contact[{index}]: {name1}({body1}) <-> {name2}({body2}), "
            f"dist={float(contact.dist):.6g}, foot={is_foot}, ground={is_ground}, force_norm={float(np.linalg.norm(force[:3])):.6g}"
        )
    print("")
    print(f"foot_ground_contacts: {foot_ground}")
    print(f"non_ground_self_contacts: {self_contacts}")
    print(f"negative_distance_contacts: {negative_dist}")
    print(f"max_contact_force_norm: {max_force:.6g}")
    print(f"base_link_torso_link_contact: {base_torso_contact}")
    print(f"isaac_style_self_collision_disabled: {self_contacts == 0}")
    print("")
    print("self_collision_note:")
    print("  Isaac G0_CFG has enabled_self_collisions=False.")
    print("  MuJoCo geoms with contype/conaffinity enabled may self-collide unless excluded.")
    print("  If non-ground self contacts appear here, they are a likely mismatch with Isaac Lab semantics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
