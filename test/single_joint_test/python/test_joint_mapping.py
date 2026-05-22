"""Phase 1 tests: mapping completeness + sign conversion + frame builder.

Run with::

    cd /home/lz/g0_robot_lab/g0_robot_lab
    pytest test/single_joint_test/python/test_joint_mapping.py -v
"""
from __future__ import annotations

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))

import joint_mapping as jm  # noqa: E402


# -- 1. Mapping completeness -------------------------------------------------

def test_motor_ids_are_1_to_22():
    ids = [e.motor_id for e in jm.JOINT_TABLE]
    assert ids == list(range(1, 23))


def test_joint_names_are_unique():
    names = [e.joint_name for e in jm.JOINT_TABLE]
    assert len(set(names)) == len(names) == 22


def test_sim_sign_is_only_plus_or_minus_one():
    for e in jm.JOINT_TABLE:
        assert e.sim_sign_observed in (+1, -1), e


def test_unknown_motor_id_raises():
    for bad in (0, 23, -1, 100):
        with pytest.raises(KeyError):
            jm.get_entry(bad)


# -- 2. Sign conversion ------------------------------------------------------

@pytest.mark.parametrize(
    "motor_id,desired,expected",
    [
        (7,  +0.1, +0.1),   # sim_sign +1
        (14, +0.1, -0.1),   # sim_sign -1
        (21, -0.1, -0.1),   # sim_sign +1
        (1,  -0.1, +0.1),   # sim_sign -1
        (10, +0.25, +0.25), # sim_sign +1
        (20, -0.3,  -0.3),  # sim_sign +1
        (3,  +0.2, -0.2),   # sim_sign -1
    ],
)
def test_apply_sign(motor_id, desired, expected):
    assert jm.apply_sign(motor_id, desired) == pytest.approx(expected)


# -- 3. Single-joint frame ---------------------------------------------------

def test_single_joint_frame_motor_14():
    frame = jm.build_single_joint_frame(14, +0.1)
    assert len(frame) == 22
    assert frame[13] == pytest.approx(-0.1)  # sim_sign(14) = -1
    for i, v in enumerate(frame):
        if i != 13:
            assert v == 0.0


@pytest.mark.parametrize("motor_id", list(range(1, 23)))
def test_single_joint_frame_isolates_target(motor_id):
    frame = jm.build_single_joint_frame(motor_id, +0.5)
    target_slot = motor_id - 1
    assert frame[target_slot] != 0.0
    for i, v in enumerate(frame):
        if i != target_slot:
            assert v == 0.0


# -- 4. Articulation validation ---------------------------------------------

def test_validate_against_articulation_ok():
    # Build a fake articulation joint_names list in a different order than
    # motor_id (shuffled) plus a few unrelated joints. All 22 mapped names
    # must still resolve.
    names = [e.joint_name for e in jm.JOINT_TABLE][::-1]
    names = ["root_dummy", *names, "free_floating"]
    mapping = jm.validate_against_articulation(names)
    assert set(mapping.keys()) == set(range(1, 23))
    for motor_id, idx in mapping.items():
        assert names[idx] == jm.motor_id_to_joint_name(motor_id)


def test_validate_against_articulation_missing():
    incomplete = [e.joint_name for e in jm.JOINT_TABLE if e.motor_id != 14]
    with pytest.raises(ValueError, match="l_knee_pitch_joint"):
        jm.validate_against_articulation(incomplete)


# -- 5. Python <-> C++ table cross-check ------------------------------------
#
# This parses the C++ header textually and asserts every row matches the
# Python table. Keeps the two sources of truth honest without needing to
# build/link the C++ side.

_CPP_ROW_RE = re.compile(
    r'\{\s*(\d+)\s*,\s*"([a-zA-Z_]+)"\s*,\s*([+-]?1)\s*\}'
)


def _load_cpp_table():
    here = os.path.dirname(__file__)
    hpp = os.path.normpath(os.path.join(here, "..", "cpp", "include", "joint_mapping.hpp"))
    with open(hpp, "r") as f:
        text = f.read()
    rows = []
    for m in _CPP_ROW_RE.finditer(text):
        rows.append((int(m.group(1)), m.group(2), int(m.group(3))))
    return rows


def test_cpp_table_matches_python():
    cpp_rows = _load_cpp_table()
    py_rows = [(e.motor_id, e.joint_name, e.sim_sign_observed) for e in jm.JOINT_TABLE]
    assert cpp_rows == py_rows
