import argparse
import os
import xml.etree.ElementTree as ET


# This is the current URDF / MuJoCo joint order printed by check_mujoco_model.py.
# It matches G0_JOINT_NAMES in g0.py.
MUJOCO_JOINT_ORDER = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "l_shoulder_pitch_joint",
    "l_shoulder_roll_joint",
    "l_shoulder_yaw_joint",
    "l_elbow_pitch_joint",
    "r_shoulder_pitch_joint",
    "r_shoulder_roll_joint",
    "r_shoulder_yaw_joint",
    "r_elbow_pitch_joint",
    "l_hip_pitch_joint",
    "l_hip_roll_joint",
    "l_hip_yaw_joint",
    "l_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "l_ankle_roll_joint",
    "r_hip_pitch_joint",
    "r_hip_roll_joint",
    "r_hip_yaw_joint",
    "r_knee_pitch_joint",
    "r_ankle_pitch_joint",
    "r_ankle_roll_joint",
]

RIGHT_ANGLE_JOINTS = {
    "l_elbow_pitch_joint",
    "r_elbow_pitch_joint",
    "l_knee_pitch_joint",
    "r_knee_pitch_joint",
    "l_ankle_pitch_joint",
    "r_ankle_pitch_joint",
}

# From g0_actuators.py
STANDARD_SERVO_RATED_TORQUE = 0.5
RIGHT_ANGLE_SERVO_RATED_TORQUE = 0.5 * 7.0 / 6.0

STANDARD_SERVO_ARMATURE = 0.0015
RIGHT_ANGLE_SERVO_ARMATURE = STANDARD_SERVO_ARMATURE * (7.0 / 6.0) ** 2

# Default pose from G0_DEFAULT_JOINT_POS, reordered into current MuJoCo order.
DEFAULT_Q = {
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,

    "l_shoulder_pitch_joint": 0.30,
    "l_shoulder_roll_joint": 0.25,
    "l_shoulder_yaw_joint": 0.0,
    "l_elbow_pitch_joint": -0.97,

    "r_shoulder_pitch_joint": 0.30,
    "r_shoulder_roll_joint": -0.25,
    "r_shoulder_yaw_joint": 0.0,
    "r_elbow_pitch_joint": -0.97,

    "l_hip_pitch_joint": -0.20,
    "l_hip_roll_joint": 0.0,
    "l_hip_yaw_joint": 0.0,
    "l_knee_pitch_joint": 0.34,
    "l_ankle_pitch_joint": -0.14,
    "l_ankle_roll_joint": 0.0,

    "r_hip_pitch_joint": -0.20,
    "r_hip_roll_joint": 0.0,
    "r_hip_yaw_joint": 0.0,
    "r_knee_pitch_joint": 0.34,
    "r_ankle_pitch_joint": -0.14,
    "r_ankle_roll_joint": 0.0,
}


def ensure_option(root):
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)

    # Unitree config.py uses SIMULATE_DT=0.005 for G1, but G1 link inertias
    # are 1e-3 kg*m^2 scale. G0 inertias are 1e-5 ~ 1e-6 scale (100-1000x
    # lighter rotationally), so 0.005s is too coarse for accurate constraint
    # integration here — measured Δpitch divergence is ~1.5x worse at dt=0.005
    # vs dt=0.002 under identical Isaac action playback. Keep 0.002 for G0.
    option.set("timestep", "0.002")
    option.set("gravity", "0 0 -9.81")


def ensure_ground(worldbody):
    for geom in worldbody.findall("geom"):
        if geom.attrib.get("name") == "ground":
            return

    ground = ET.Element(
        "geom",
        {
            "name": "ground",
            "type": "plane",
            "pos": "0 0 0",
            "size": "5 5 0.05",
            "rgba": "0.8 0.8 0.8 1",
            "friction": "1.0 0.005 0.0001",
        },
    )
    worldbody.insert(0, ground)


def ensure_freejoint(worldbody):
    # Reconstruct the original URDF root hierarchy:
    #
    #   base_link
    #     ├── waist_yaw_link
    #     ├── l_hip_pitch_link
    #     └── r_hip_pitch_link
    #
    # MuJoCo's URDF compiler converted base_link into a top-level world geom
    # and lifted its three movable child branches into top-level bodies.
    # For floating-base sim2sim, we must rebuild base_link as the movable root body:
    #
    # <worldbody>
    #   <geom name="ground" .../>
    #   <body name="base_link" pos="0 0 0">
    #     <inertial .../>
    #     <freejoint name="root"/>
    #     <geom mesh="base_link"/>
    #     <body name="waist_yaw_link">...</body>
    #     <body name="l_hip_pitch_link">...</body>
    #     <body name="r_hip_pitch_link">...</body>
    #   </body>
    # </worldbody>

    # Already patched.
    for body in worldbody.findall("body"):
        if body.attrib.get("name") == "base_link":
            has_freejoint = any(child.tag == "freejoint" for child in body)
            if not has_freejoint:
                body.insert(1, ET.Element("freejoint", {"name": "root"}))
            return

    children = list(worldbody)

    # MuJoCo-generated top-level robot bodies:
    # waist_yaw_link, l_hip_pitch_link, r_hip_pitch_link.
    robot_roots = [
        child for child in children
        if child.tag == "body"
    ]

    if not robot_roots:
        raise RuntimeError("No robot root body found under <worldbody>.")

    # MuJoCo-generated top-level robot geoms.
    # Keep ground in worldbody; move base_link mesh under base_link body.
    base_geoms = [
        child for child in children
        if child.tag == "geom"
        and child.attrib.get("name") != "ground"
    ]

    # Insert base_link where the first robot body originally appeared.
    insert_candidates = robot_roots + base_geoms
    insert_index = min(children.index(obj) for obj in insert_candidates)

    base_body = ET.Element("body", {"name": "base_link", "pos": "0 0 0"})

    # Real base_link inertial from URDF.
    # URDF:
    #   origin xyz="-1.2518E-05 0.00025287 -0.0020895"
    #   mass value="0.033536"
    #   inertia:
    #     ixx="5.9104E-06"
    #     ixy="3.3651E-11"
    #     ixz="-8.873E-09"
    #     iyy="1.1303E-05"
    #     iyz="2.1244E-09"
    #     izz="6.7549E-06"
    #
    # MJCF fullinertia order:
    #   ixx iyy izz ixy ixz iyz
    base_body.append(
        ET.Element(
            "inertial",
            {
                "pos": "-1.2518e-05 0.00025287 -0.0020895",
                "mass": "0.033536",
                "fullinertia": "5.9104e-06 1.1303e-05 6.7549e-06 3.3651e-11 -8.873e-09 2.1244e-09",
            },
        )
    )

    base_body.append(ET.Element("freejoint", {"name": "root"}))

    # Move base_link geom(s) into base_link body.
    for geom in base_geoms:
        worldbody.remove(geom)
        base_body.append(geom)

    # Move all robot branches into base_link body.
    for body in robot_roots:
        worldbody.remove(body)
        base_body.append(body)

    worldbody.insert(insert_index, base_body)

    body_names = [body.attrib.get("name", "<unnamed>") for body in robot_roots]
    geom_names = [
        geom.attrib.get("name", geom.attrib.get("mesh", "<unnamed>"))
        for geom in base_geoms
    ]

    print(f"Reconstructed movable base_link body.")
    print(f"Moved base geoms under base_link: {geom_names}")
    print(f"Moved robot branches under base_link: {body_names}")


def patch_joint_armature(root):
    # Per-joint armature (matches Isaac STANDARD/RIGHT_ANGLE servo armature),
    # plus small implicit damping + frictionloss following the unitree_mujoco
    # G1 convention (unitree_robots/g1/g1_29dof.xml). G1 uses
    # damping=0.05 / frictionloss=0.2 at ~25-139 Nm torque range; G0's
    # ctrlrange peaks at 0.583 Nm, so we scale by torque ratio ~1/240:
    #   damping      ~ 2e-4 N*m*s/rad
    #   frictionloss ~ 1e-3 N*m
    # The script-side PD still handles the dominant kp*(q_des-q) - kd*qd;
    # these small implicit terms eliminate per-step numerical jitter in the
    # joint constraint solver without overriding the MIT control law.
    UNITREE_DAMPING = 2e-4
    UNITREE_FRICTIONLOSS = 1e-3
    for joint in root.iter("joint"):
        name = joint.attrib.get("name")
        if name not in MUJOCO_JOINT_ORDER:
            continue

        if name in RIGHT_ANGLE_JOINTS:
            joint.set("armature", f"{RIGHT_ANGLE_SERVO_ARMATURE:.10g}")
        else:
            joint.set("armature", f"{STANDARD_SERVO_ARMATURE:.10g}")

        joint.set("damping", f"{UNITREE_DAMPING:.10g}")
        joint.set("frictionloss", f"{UNITREE_FRICTIONLOSS:.10g}")


def ensure_actuators(root):
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.Element("actuator")
        root.append(actuator)

    existing = {
        act.attrib.get("joint")
        for act in actuator
        if act.tag == "motor" and "joint" in act.attrib
    }

    for joint in MUJOCO_JOINT_ORDER:
        if joint in existing:
            continue

        if joint in RIGHT_ANGLE_JOINTS:
            limit = RIGHT_ANGLE_SERVO_RATED_TORQUE
        else:
            limit = STANDARD_SERVO_RATED_TORQUE

        motor = ET.Element(
            "motor",
            {
                "name": joint.replace("_joint", "_motor"),
                "joint": joint,
                "gear": "1",
                "ctrllimited": "true",
                "ctrlrange": f"{-limit:.10g} {limit:.10g}",
            },
        )
        actuator.append(motor)


def patch_foot_contacts(root):
    """Replace foot mesh-on-plane collision with 4 corner spheres per foot.

    Mesh-vs-plane contact resolution differs significantly between PhysX (Isaac)
    and MuJoCo: PhysX builds an internal contact set from the convex hull,
    MuJoCo from the mesh's collision representation. The two can produce
    different contact normals / counts for the same pose, which destroys
    sim2sim parity at the ankle.

    Following the Unitree G1 reference (unitree_robots/g1/g1_29dof.xml:124-127),
    we disable the foot mesh as a collider (keep it visual only) and add 4
    small spheres at the 4 bottom corners of the foot's body-local AABB.
    Sphere-on-plane contact is analytic in both simulators and matches
    near-perfectly.

    AABB measured at default_stand from the patched MJCF (body-local frame):
        x (heel->toe): [-0.0395, +0.0353]
        y (lateral):   ±0.0160 (effectively symmetric for both feet)
        z (bottom):    -0.0168
    The 4 sphere centers are placed at z = mesh_bottom + radius so the bottom
    of each sphere lines up with the mesh bottom and the stand height stays the
    same as before.
    """
    HEEL_X = -0.0395
    TOE_X = 0.0353
    HALF_Y = 0.0160
    MESH_BOT_Z = -0.0168
    R = 0.003
    SPHERE_CENTER_Z = MESH_BOT_Z + R  # = -0.0138

    corners = [
        (HEEL_X, +HALF_Y, SPHERE_CENTER_Z),
        (HEEL_X, -HALF_Y, SPHERE_CENTER_Z),
        (TOE_X,  +HALF_Y, SPHERE_CENTER_Z),
        (TOE_X,  -HALF_Y, SPHERE_CENTER_Z),
    ]

    foot_bodies = ("l_foot_link", "r_foot_link")
    patched = 0
    for body in root.iter("body"):
        if body.attrib.get("name") not in foot_bodies:
            continue
        # Disable contact on existing mesh geoms; keep them as visuals.
        for g in body.findall("geom"):
            if g.attrib.get("type", "mesh") == "mesh":
                g.set("contype", "0")
                g.set("conaffinity", "0")
                g.set("group", "1")
        # Remove any previously injected contact spheres (so re-patching is idempotent).
        for g in list(body.findall("geom")):
            if g.attrib.get("name", "").startswith(body.attrib["name"] + "_contact_"):
                body.remove(g)
        # Add 4 corner spheres.
        for i, (x, y, z) in enumerate(corners):
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"{body.attrib['name']}_contact_{i}",
                    "type": "sphere",
                    "size": f"{R:.6g}",
                    "pos": f"{x:.6g} {y:.6g} {z:.6g}",
                    "rgba": "0.9 0.2 0.2 1",
                    "contype": "1",
                    "conaffinity": "1",
                    "group": "3",
                    "friction": "1.0 0.005 0.0001",
                },
            )
        patched += 1
    if patched != 2:
        raise RuntimeError(
            f"patch_foot_contacts: expected to patch 2 foot bodies, patched {patched}"
        )


def ensure_keyframe(root):
    keyframe = root.find("keyframe")
    if keyframe is None:
        keyframe = ET.Element("keyframe")
        root.append(keyframe)

    # freejoint qpos = x y z qw qx qy qz
    root_qpos = [0.0, 0.0, 0.23, 1.0, 0.0, 0.0, 0.0]
    # suspended_stand: same joint pose as default_stand but raised so the
    # elastic band visibly holds the robot off the ground (Unitree staging SOP).
    # Opt-in via deploy_staging.yaml `keyframe: suspended_stand`.
    suspended_root_qpos = [0.0, 0.0, 0.50, 1.0, 0.0, 0.0, 0.0]
    default_joints = [DEFAULT_Q[name] for name in MUJOCO_JOINT_ORDER]
    keyframes = {
        "zero_pose": root_qpos + [0.0 for _ in MUJOCO_JOINT_ORDER],
        "default_stand": root_qpos + default_joints,
        "suspended_stand": suspended_root_qpos + default_joints,
    }

    for name, qpos in keyframes.items():
        qpos_str = " ".join(f"{x:.8g}" for x in qpos)
        for key in keyframe.findall("key"):
            if key.attrib.get("name") != name:
                continue
            key.set("qpos", qpos_str)
            break
        else:
            keyframe.append(ET.Element("key", {"name": name, "qpos": qpos_str}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_xml", required=True)
    parser.add_argument("--out_xml", required=True)
    args = parser.parse_args()

    tree = ET.parse(args.in_xml)
    root = tree.getroot()

    if root.tag != "mujoco":
        raise RuntimeError(f"Expected root <mujoco>, got <{root.tag}>")

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Missing <worldbody>")

    ensure_option(root)
    ensure_ground(worldbody)
    ensure_freejoint(worldbody)
    patch_joint_armature(root)
    patch_foot_contacts(root)
    ensure_actuators(root)
    ensure_keyframe(root)

    ET.indent(tree, space="  ")
    os.makedirs(os.path.dirname(args.out_xml), exist_ok=True)
    tree.write(args.out_xml, encoding="utf-8", xml_declaration=True)

    print("Patched MJCF written to:")
    print(args.out_xml)
    print()
    print("Expected model stats after loading:")
    print("  nq   = 29")
    print("  nv   = 28")
    print("  nu   = 22")
    print("  njnt = 23")
    print()
    print("Actuator mode:")
    print("  torque motors")
    print("  standard servo ctrlrange = ±0.5 Nm")
    print(f"  right-angle servo ctrlrange = ±{RIGHT_ANGLE_SERVO_RATED_TORQUE:.6f} Nm")


if __name__ == "__main__":
    main()
