"""Keyboard 'remote controller' — Python stand-in for Unitree's gamepad.

The viewer's key callback feeds key events here; FSM polls .pending_fsm and
.cmd. State transitions are *requested*, never forced: CtrlFSM gates them.
"""

import threading


class RemoteController:
    def __init__(self, key_bindings: dict):
        self.kb = key_bindings
        self._lock = threading.Lock()
        self.pending_fsm = None     # one of: "passive", "fix_stand", "rl_base"
        # one of: "loosen", "tighten", "toggle", "confirm_ground" (Unitree staging keys)
        self.pending_band_action = None
        self.cmd = [0.0, 0.0, 0.0]  # vx, vy, wz
        # nudge step sizes
        self.dv = 0.1
        self.dw = 0.2

    def on_key(self, key_char: str):
        if not key_char:
            return
        c = key_char.lower()
        with self._lock:
            if c == self.kb["passive"]:
                self.pending_fsm = "passive"
            elif c == self.kb["fix_stand"]:
                self.pending_fsm = "fix_stand"
            elif c == self.kb["rl_base"]:
                self.pending_fsm = "rl_base"
            elif c == self.kb["zero_cmd"]:
                self.cmd = [0.0, 0.0, 0.0]
            elif c == self.kb["vx_up"]:
                self.cmd[0] += self.dv
            elif c == self.kb["vx_down"]:
                self.cmd[0] -= self.dv
            elif c == self.kb["vy_left"]:
                self.cmd[1] += self.dv
            elif c == self.kb["vy_right"]:
                self.cmd[1] -= self.dv
            elif c == self.kb["wz_left"]:
                self.cmd[2] += self.dw
            elif c == self.kb["wz_right"]:
                self.cmd[2] -= self.dw
            # Unitree staging band keys. Use .get() so production deploy.yaml,
            # which has no band bindings, is unaffected (no key -> None != c).
            elif c == self.kb.get("band_loosen"):
                self.pending_band_action = "loosen"
            elif c == self.kb.get("band_tighten"):
                self.pending_band_action = "tighten"
            elif c == self.kb.get("band_toggle"):
                self.pending_band_action = "toggle"
            elif c == self.kb.get("confirm_ground"):
                self.pending_band_action = "confirm_ground"

    def consume_band_action(self):
        with self._lock:
            v, self.pending_band_action = self.pending_band_action, None
            return v

    def consume_fsm(self):
        with self._lock:
            v, self.pending_fsm = self.pending_fsm, None
            return v

    def get_cmd(self):
        with self._lock:
            return list(self.cmd)
