import xml.etree.ElementTree as ET

from patch_g0_mjcf_for_sim2sim import MUJOCO_JOINT_ORDER, ensure_keyframe


def test_ensure_keyframe_adds_zero_pose_and_default_stand():
    root = ET.fromstring("<mujoco />")

    ensure_keyframe(root)

    keyframe = root.find("keyframe")
    assert keyframe is not None

    keys = {key.attrib["name"]: key for key in keyframe.findall("key")}
    assert set(keys) == {"zero_pose", "default_stand"}

    zero_qpos = [float(value) for value in keys["zero_pose"].attrib["qpos"].split()]
    default_qpos = [float(value) for value in keys["default_stand"].attrib["qpos"].split()]

    assert zero_qpos[:7] == [0.0, 0.0, 0.23, 1.0, 0.0, 0.0, 0.0]
    assert zero_qpos[7:] == [0.0] * len(MUJOCO_JOINT_ORDER)
    assert default_qpos[:7] == zero_qpos[:7]
    assert default_qpos[7:] != zero_qpos[7:]
