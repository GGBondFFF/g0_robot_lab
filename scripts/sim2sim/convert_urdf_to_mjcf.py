import argparse
import os
import tempfile
import xml.etree.ElementTree as ET

import mujoco


def ensure_mujoco_fusestatic_disabled(root):
    mujoco_ext = root.find("mujoco")
    if mujoco_ext is None:
        mujoco_ext = ET.SubElement(root, "mujoco")

    compiler = mujoco_ext.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mujoco_ext, "compiler")

    # Keep base_link as a real body. Otherwise MuJoCo fuses the fixed root into
    # world and lifts waist/hip branches, which breaks the sim2sim root model.
    compiler.set("fusestatic", "false")

    # MuJoCo's URDF compiler defaults are boundmass=1e-3 and boundinertia=1e-3,
    # which silently clamp all link inertias up. For G0 (smallest link inertia
    # ~2e-5 kg*m^2) this inflates rotational inertia by ~40x and destroys
    # sim2sim parity with Isaac (which uses the URDF inertials directly via
    # PhysX). Set both bounds to effectively zero so the URDF inertials pass
    # through untouched.
    compiler.set("boundmass", "1e-12")
    compiler.set("boundinertia", "1e-12")


def write_preserve_base_urdf(urdf_path):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    if root.tag != "robot":
        raise RuntimeError(f"Expected root <robot>, got <{root.tag}>")

    ensure_mujoco_fusestatic_disabled(root)
    ET.indent(tree, space="  ")

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".urdf",
        prefix="g0_preserve_base_",
        dir=os.path.dirname(os.path.abspath(urdf_path)),
        delete=False,
        encoding="utf-8",
    )
    try:
        tree.write(tmp, encoding="unicode", xml_declaration=True)
        return tmp.name
    finally:
        tmp.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    patched_urdf = write_preserve_base_urdf(args.urdf)
    try:
        model = mujoco.MjModel.from_xml_path(patched_urdf)
    finally:
        os.remove(patched_urdf)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    mujoco.mj_saveLastXML(args.out, model)

    print("Saved MJCF XML:")
    print(args.out)


if __name__ == "__main__":
    main()
