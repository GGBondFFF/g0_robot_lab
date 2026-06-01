# tests/unit/test_main_band_wiring.py
"""Unit test for the band-action wiring in deploy.robots.g0.main.tick().

Uses lightweight fakes (no MuJoCo / ONNX) to prove that the production tick
loop consumes RemoteController band actions and applies them to the band /
staging context before band.update().
"""
import numpy as np
import pytest

from deploy.common.elastic_band import ElasticBand
from deploy.common.remote_controller import RemoteController
from deploy.robots.g0.main import tick


_KEYS = {
    "passive": "p", "fix_stand": "f", "rl_base": "r", "zero_cmd": "0",
    "vx_up": "w", "vx_down": "s", "vy_left": "a", "vy_right": "d",
    "wz_left": "q", "wz_right": "e",
    "band_toggle": "9", "band_tighten": "7", "band_loosen": "8",
    "confirm_ground": "g",
}


class _FakeBackend:
    def __init__(self, z=1.0):
        self._z = z
        self.last_f = None

    def read_state(self):
        return {
            "base_pos_w": np.array([0.0, 0.0, self._z]),
            "base_lin_vel_w": np.zeros(3),
        }

    def clear_external_force(self):
        pass

    def apply_external_force(self, f):
        self.last_f = np.asarray(f, dtype=np.float64)


class _FakeCmdMgr:
    def set(self, *a):
        self.last = a


class _FakeFsm:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


class _Staging:
    def __init__(self):
        self.feet_on_ground = False


def _ctx():
    return RemoteController(_KEYS), _FakeBackend(), _FakeCmdMgr(), _FakeFsm()


def test_tick_loosen_increments_rest_length_once():
    rc, backend, cmd, fsm = _ctx()
    band = ElasticBand(rest_length=0.0, enabled=True)
    rc.on_key("8")
    tick(cmd, rc, band, backend, fsm, length_step_m=0.1)
    assert band.L == pytest.approx(0.1)
    # Action is consumed; a second tick without a keypress must not move L.
    tick(cmd, rc, band, backend, fsm, length_step_m=0.1)
    assert band.L == pytest.approx(0.1)
    # FSM was stepped each tick, band.update still drives the loop.
    assert fsm.steps == 2


def test_tick_tighten_decrements_rest_length():
    rc, backend, cmd, fsm = _ctx()
    band = ElasticBand(rest_length=0.5, enabled=True)
    rc.on_key("7")
    tick(cmd, rc, band, backend, fsm, length_step_m=0.1)
    assert band.L == pytest.approx(0.4)


def test_tick_toggle_disables_band():
    rc, backend, cmd, fsm = _ctx()
    band = ElasticBand(rest_length=0.0, enabled=True)
    rc.on_key("9")
    tick(cmd, rc, band, backend, fsm, length_step_m=0.1)
    assert band.enabled is False


def test_tick_confirm_ground_sets_flag():
    rc, backend, cmd, fsm = _ctx()
    band = ElasticBand(rest_length=0.0, enabled=True)
    staging = _Staging()
    rc.on_key("g")
    tick(cmd, rc, band, backend, fsm, length_step_m=0.1, staging=staging)
    assert staging.feet_on_ground is True


def test_tick_backward_compatible_without_band_keys():
    # Production deploy.yaml call path: 5 positional args, no band keys pressed.
    rc = RemoteController({
        "passive": "p", "fix_stand": "f", "rl_base": "r", "zero_cmd": "0",
        "vx_up": "w", "vx_down": "s", "vy_left": "a", "vy_right": "d",
        "wz_left": "q", "wz_right": "e",
    })
    band = ElasticBand(rest_length=0.0, enabled=True)
    backend, cmd, fsm = _FakeBackend(), _FakeCmdMgr(), _FakeFsm()
    tick(cmd, rc, band, backend, fsm)
    assert band.L == pytest.approx(0.0)
    assert band.enabled is True
    assert fsm.steps == 1
