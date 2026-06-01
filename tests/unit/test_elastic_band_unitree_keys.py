# tests/unit/test_elastic_band_unitree_keys.py
import numpy as np
import pytest
from deploy.common.elastic_band import ElasticBand


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


def test_adjust_rest_length_matches_unitree_step():
    band = ElasticBand(
        anchor=(0.0, 0.0, 2.0),
        stiffness=50.0,
        damping=10.0,
        rest_length=0.0,
        enabled=True,
        one_sided=True,
    )
    band.adjust_rest_length(+0.1)
    assert band.L == pytest.approx(0.1)
    band.adjust_rest_length(-0.1)
    assert band.L == pytest.approx(0.0)


def test_toggle_enabled():
    band = ElasticBand(enabled=True)
    band.toggle_enabled()
    assert band.enabled is False
    band.toggle_enabled()
    assert band.enabled is True
