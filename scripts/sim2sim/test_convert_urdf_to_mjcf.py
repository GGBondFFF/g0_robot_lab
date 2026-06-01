import xml.etree.ElementTree as ET
from pathlib import Path

from convert_urdf_to_mjcf import ensure_mujoco_fusestatic_disabled, write_preserve_base_urdf


def test_ensure_mujoco_fusestatic_disabled_adds_compiler_when_missing():
    root = ET.fromstring('<robot name="g0"><link name="base_link" /></robot>')

    ensure_mujoco_fusestatic_disabled(root)

    compiler = root.find("mujoco/compiler")
    assert compiler is not None
    assert compiler.attrib["fusestatic"] == "false"


def test_ensure_mujoco_fusestatic_disabled_preserves_existing_compiler_attrs():
    root = ET.fromstring(
        '<robot name="g0"><mujoco><compiler meshdir="../meshes" /></mujoco></robot>'
    )

    ensure_mujoco_fusestatic_disabled(root)

    compiler = root.find("mujoco/compiler")
    assert compiler is not None
    assert compiler.attrib["meshdir"] == "../meshes"
    assert compiler.attrib["fusestatic"] == "false"


def test_write_preserve_base_urdf_keeps_temp_file_next_to_source(tmp_path):
    urdf_path = tmp_path / "robot.urdf"
    urdf_path.write_text('<robot name="g0"><link name="base_link" /></robot>', encoding="utf-8")

    patched_path = Path(write_preserve_base_urdf(str(urdf_path)))

    try:
        assert patched_path.parent == tmp_path
    finally:
        patched_path.unlink()
