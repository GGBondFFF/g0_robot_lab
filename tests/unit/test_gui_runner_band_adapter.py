"""Task 8 — DRY band in scripts/sim2sim/g0_mujoco_onnx_gui_runner.py.

Verifies the runner now drives the shared deploy.common.elastic_band.ElasticBand
through a thin MuJoCo adapter, that the produced base force is algebraically
identical to the old inline two-sided band math, and that the new
``--staging-sop`` preset enables the band.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RUNNER_PATH = os.path.join(
    REPO_ROOT, "scripts", "sim2sim", "g0_mujoco_onnx_gui_runner.py")


def _load_runner():
    spec = importlib.util.spec_from_file_location("g0_gui_runner", RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _FakeData:
    def __init__(self, base_id, n_bodies=5):
        # freejoint qpos = x y z qw qx qy qz ; qvel base = vx vy vz wx wy wz
        self.qpos = np.array([0.10, 0.20, 0.50, 1, 0, 0, 0], dtype=np.float64)
        self.qvel = np.array([0.0, 0.0, -0.30, 0, 0, 0], dtype=np.float64)
        self.xfrc_applied = np.zeros((n_bodies, 6), dtype=np.float64)


def test_adapter_read_apply_clear():
    gui = _load_runner()
    base_id = 2
    data = _FakeData(base_id)
    adapter = gui._MjBaseBandAdapter(data, base_id)

    s = adapter.read_state()
    np.testing.assert_allclose(s["base_pos_w"], data.qpos[0:3])
    np.testing.assert_allclose(s["base_lin_vel_w"], data.qvel[0:3])

    adapter.apply_external_force(np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(data.xfrc_applied[base_id, 0:3], [1.0, 2.0, 3.0])

    adapter.clear_external_force()
    np.testing.assert_allclose(data.xfrc_applied[base_id, 0:3], [0.0, 0.0, 0.0])


def test_band_force_matches_old_inline_math():
    from deploy.common.elastic_band import ElasticBand
    gui = _load_runner()
    base_id = 2
    data = _FakeData(base_id)
    adapter = gui._MjBaseBandAdapter(data, base_id)

    anchor = np.array([0.0, 0.0, 2.0])
    band_k, band_c, band_L = 10.0, 2.0, 0.0

    # Shared-band path (two-sided to match the legacy inline formula).
    band = ElasticBand(anchor=tuple(anchor), stiffness=band_k, damping=band_c,
                       rest_length=band_L, enabled=True, one_sided=False,
                       mode="rope")
    band.update(adapter)
    got = data.xfrc_applied[base_id, 0:3].copy()

    # Old inline math (g0_mujoco_onnx_gui_runner.py:495-501, pre-refactor).
    x_base = data.qpos[0:3]
    v_base = data.qvel[0:3]
    delta = anchor - x_base
    dist = float(np.linalg.norm(delta))
    dir_unit = delta / dist
    v_along = float(np.dot(v_base, dir_unit))
    scalar_f = band_k * (dist - band_L) - band_c * v_along
    expected = scalar_f * dir_unit

    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_staging_sop_flag_enables_band():
    gui = _load_runner()
    parser = gui.build_arg_parser()
    args = parser.parse_args(["--model", "m.xml", "--policy", "p.onnx",
                              "--staging-sop"])
    assert args.staging_sop is True
    assert gui.band_enabled_from_args(args) is True

    # Without --elastic-band/--staging-sop the band stays off.
    args2 = parser.parse_args(["--model", "m.xml", "--policy", "p.onnx"])
    assert gui.band_enabled_from_args(args2) is False
