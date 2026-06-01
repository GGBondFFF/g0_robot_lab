import argparse
import time

import mujoco
import numpy as np


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


# Same gains as current g0.py, reordered into MuJoCo joint order.
KP = {
    "waist_yaw_joint": 2.0,
    "waist_roll_joint": 2.0,

    "l_shoulder_pitch_joint": 1.5,
    "l_shoulder_roll_joint": 1.5,
    "l_shoulder_yaw_joint": 1.5,
    "l_elbow_pitch_joint": 2.0,

    "r_shoulder_pitch_joint": 1.5,
    "r_shoulder_roll_joint": 1.5,
    "r_shoulder_yaw_joint": 1.5,
    "r_elbow_pitch_joint": 2.0,

    "l_hip_pitch_joint": 4.0,
    "l_hip_roll_joint": 4.0,
    "l_hip_yaw_joint": 3.0,
    "l_knee_pitch_joint": 4.0,
    "l_ankle_pitch_joint": 4.0,
    "l_ankle_roll_joint": 4.5,

    "r_hip_pitch_joint": 4.0,
    "r_hip_roll_joint": 4.0,
    "r_hip_yaw_joint": 3.0,
    "r_knee_pitch_joint": 4.0,
    "r_ankle_pitch_joint": 4.0,
    "r_ankle_roll_joint": 4.5,
}


KD = {
    "waist_yaw_joint": 0.08,
    "waist_roll_joint": 0.08,

    "l_shoulder_pitch_joint": 0.06,
    "l_shoulder_roll_joint": 0.06,
    "l_shoulder_yaw_joint": 0.06,
    "l_elbow_pitch_joint": 0.08,

    "r_shoulder_pitch_joint": 0.06,
    "r_shoulder_roll_joint": 0.06,
    "r_shoulder_yaw_joint": 0.06,
    "r_elbow_pitch_joint": 0.08,

    "l_hip_pitch_joint": 0.18,
    "l_hip_roll_joint": 0.16,
    "l_hip_yaw_joint": 0.10,
    "l_knee_pitch_joint": 0.26,
    "l_ankle_pitch_joint": 0.22,
    "l_ankle_roll_joint": 0.15,

    "r_hip_pitch_joint": 0.18,
    "r_hip_roll_joint": 0.16,
    "r_hip_yaw_joint": 0.10,
    "r_knee_pitch_joint": 0.26,
    "r_ankle_pitch_joint": 0.22,
    "r_ankle_roll_joint": 0.15,
}


def get_joint_addresses(model):
    qadr = []
    dadr = []
    for name in MUJOCO_JOINT_ORDER:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(f"Missing joint: {name}")
        qadr.append(int(model.jnt_qposadr[jid]))
        dadr.append(int(model.jnt_dofadr[jid]))
    return np.array(qadr, dtype=np.int32), np.array(dadr, dtype=np.int32)


def get_target_q(target_pose):
    if target_pose == "zero_pose":
        return np.zeros(len(MUJOCO_JOINT_ORDER), dtype=np.float64)
    if target_pose == "default_stand":
        return np.array([DEFAULT_Q[name] for name in MUJOCO_JOINT_ORDER], dtype=np.float64)
    raise ValueError(f"Unknown target pose: {target_pose}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument(
        "--keyframe",
        default="default_stand",
        choices=("default_stand", "zero_pose"),
        help="Initial pose keyframe. Use zero_pose to compare against the static URDF/USD asset.",
    )
    parser.add_argument(
        "--target-pose",
        default=None,
        choices=("default_stand", "zero_pose"),
        help="PD hold target. Defaults to the same pose as --keyframe.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Open the viewer at the selected keyframe without stepping physics.",
    )
    parser.add_argument("--viewer", action="store_true")
    args = parser.parse_args()
    target_pose = args.target_pose or args.keyframe

    model = mujoco.MjModel.from_xml_path(args.model)
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe)
    if key_id < 0:
        raise RuntimeError(f"Missing keyframe: {args.keyframe}")
    mujoco.mj_resetDataKeyframe(model, data, key_id)

    qadr, dadr = get_joint_addresses(model)

    target_q = get_target_q(target_pose)
    kp = np.array([KP[name] for name in MUJOCO_JOINT_ORDER], dtype=np.float64)
    kd = np.array([KD[name] for name in MUJOCO_JOINT_ORDER], dtype=np.float64)

    print("=== Zero-action standing test ===")
    print("model:", args.model)
    print("duration:", args.duration)
    print("keyframe:", args.keyframe)
    print("target_pose:", target_pose)
    print("nq/nv/nu:", model.nq, model.nv, model.nu)
    print("initial root z:", data.qpos[2])
    print("initial joint q:", data.qpos[qadr])
    print("target joint q:", target_q)

    start = time.time()
    sim_steps = int(args.duration / model.opt.timestep)

    def step_once():
        q = data.qpos[qadr]
        dq = data.qvel[dadr]

        tau = kp * (target_q - q) - kd * dq

        # data.ctrl is already clamped by actuator ctrlrange in MuJoCo,
        # but clipping here makes diagnostics explicit.
        ctrl_min = model.actuator_ctrlrange[:, 0]
        ctrl_max = model.actuator_ctrlrange[:, 1]
        tau = np.clip(tau, ctrl_min, ctrl_max)

        data.ctrl[:] = tau
        mujoco.mj_step(model, data)

        return tau

    if args.viewer:
        from mujoco import viewer as mujoco_viewer

        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.sync()
            if args.inspect_only:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.05)
                return

            for step in range(sim_steps):
                tau = step_once()

                if step % 250 == 0:
                    print(
                        f"t={data.time:6.3f} "
                        f"root_z={data.qpos[2]: .4f} "
                        f"max|tau|={np.max(np.abs(tau)): .4f} "
                        f"max|dq|={np.max(np.abs(data.qvel[dadr])): .4f}"
                    )

                viewer.sync()
                time.sleep(model.opt.timestep)
    else:
        for step in range(sim_steps):
            tau = step_once()

            if step % 250 == 0:
                print(
                    f"t={data.time:6.3f} "
                    f"root_z={data.qpos[2]: .4f} "
                    f"max|tau|={np.max(np.abs(tau)): .4f} "
                    f"max|dq|={np.max(np.abs(data.qvel[dadr])): .4f}"
                )

    print("=== Final ===")
    print("root pos:", data.qpos[0:3])
    print("root quat:", data.qpos[3:7])
    print("joint q:", data.qpos[qadr])
    print("elapsed wall time:", time.time() - start)


if __name__ == "__main__":
    main()
