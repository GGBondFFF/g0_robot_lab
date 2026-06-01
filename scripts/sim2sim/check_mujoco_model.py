import argparse
import mujoco


def obj_name(model, obj_type, obj_id):
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    return name if name is not None else "<unnamed>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to MuJoCo XML/MJCF/URDF file")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    print("=== MuJoCo model loaded successfully ===")
    print(f"model: {args.model}")
    print(f"nq:    {model.nq}")
    print(f"nv:    {model.nv}")
    print(f"nu:    {model.nu}")
    print(f"njnt:  {model.njnt}")
    print(f"nbody: {model.nbody}")
    print(f"ngeom: {model.ngeom}")

    print("\n=== Joints ===")
    for i in range(model.njnt):
        name = obj_name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jnt_type = int(model.jnt_type[i])
        qpos_addr = int(model.jnt_qposadr[i])
        dof_addr = int(model.jnt_dofadr[i])
        axis = model.jnt_axis[i]
        rng = model.jnt_range[i]
        print(
            f"{i:02d}  name={name:35s} "
            f"type={jnt_type} qposadr={qpos_addr:02d} dofadr={dof_addr:02d} "
            f"axis=[{axis[0]: .3f}, {axis[1]: .3f}, {axis[2]: .3f}] "
            f"range=[{rng[0]: .3f}, {rng[1]: .3f}]"
        )

    print("\n=== Actuators ===")
    if model.nu == 0:
        print("[WARN] model.nu == 0. This model has no actuators yet.")
        print("       URDF usually has joints but no MuJoCo actuators.")
        print("       Later we need to add 22 motor actuators in MJCF/XML.")
    else:
        for i in range(model.nu):
            name = obj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            trnid = model.actuator_trnid[i]
            ctrlrange = model.actuator_ctrlrange[i]
            forcerange = model.actuator_forcerange[i]
            print(
                f"{i:02d}  name={name:35s} "
                f"trnid={trnid.tolist()} "
                f"ctrlrange=[{ctrlrange[0]: .3f}, {ctrlrange[1]: .3f}] "
                f"forcerange=[{forcerange[0]: .3f}, {forcerange[1]: .3f}]"
            )

    print("\n=== Initial qpos head ===")
    print(data.qpos[: min(30, model.nq)])

    print("\n=== Summary ===")
    if model.nu == 0:
        print("Load OK, but actuator missing. Next step: convert to MJCF and add actuators.")
    else:
        print("Load OK, actuator exists. Next step: check zero-action/default standing.")


if __name__ == "__main__":
    main()
