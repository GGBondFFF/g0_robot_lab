# tests/unit/test_remote_controller_band_keys.py
from deploy.common.remote_controller import RemoteController


def _rc():
    return RemoteController({
        "passive": "p", "fix_stand": "f", "rl_base": "r", "zero_cmd": "0",
        "vx_up": "w", "vx_down": "s", "vy_left": "a", "vy_right": "d",
        "wz_left": "q", "wz_right": "e",
        "band_toggle": "9", "band_tighten": "7", "band_loosen": "8",
        "confirm_ground": "g",
    })


def test_band_loosen_key():
    rc = _rc()
    rc.on_key("8")
    assert rc.consume_band_action() == "loosen"
    assert rc.consume_band_action() is None


def test_band_tighten_key():
    rc = _rc()
    rc.on_key("7")
    assert rc.consume_band_action() == "tighten"
    assert rc.consume_band_action() is None


def test_band_toggle_key():
    rc = _rc()
    rc.on_key("9")
    assert rc.consume_band_action() == "toggle"
    assert rc.consume_band_action() is None


def test_confirm_ground_key():
    rc = _rc()
    rc.on_key("g")
    assert rc.consume_band_action() == "confirm_ground"


def test_band_keys_absent_in_production_bindings():
    # deploy.yaml has no band keys; missing bindings must not raise.
    rc = RemoteController({
        "passive": "p", "fix_stand": "f", "rl_base": "r", "zero_cmd": "0",
        "vx_up": "w", "vx_down": "s", "vy_left": "a", "vy_right": "d",
        "wz_left": "q", "wz_right": "e",
    })
    rc.on_key("8")
    rc.on_key("g")
    assert rc.consume_band_action() is None
