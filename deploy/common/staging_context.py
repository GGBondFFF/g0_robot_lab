"""Mutable staging context for the Unitree-aligned sim2sim SOP.

Carries the ``feet_on_ground`` gate that ``StateFixStand`` checks before it
allows the RLBase transition (mirrors the Unitree flow: lower the robot onto
the ground with the band, confirm contact, *then* hand off to the policy).

Two ways to set the flag:

* **Manual** — the operator presses ``g`` (``confirm_ground``); the main loop
  sets ``feet_on_ground = True`` directly.
* **Auto-detect** (opt-in via ``auto_ground_detect: true``) — once ``base_z`` and
  ``|roll|/|pitch|`` stay within thresholds for ``ground_hold_s`` seconds,
  ``maybe_auto_detect`` flips the flag.

The context is *present* only for the staging profile (``deploy_staging.yaml``
has a ``staging:`` block). Production ``deploy.yaml`` has none, so the gate is
inactive and behavior is unchanged.
"""

from .rotation import quat_to_rpy_wxyz


class StagingContext:
    def __init__(self, cfg: dict | None):
        self.cfg = cfg or {}
        self.feet_on_ground = False
        self._ground_timer = 0.0

    def enabled(self) -> bool:
        """True when a staging config block is present (staging gate active)."""
        return bool(self.cfg)

    def maybe_auto_detect(self, backend, step_dt: float) -> None:
        """Flip ``feet_on_ground`` once the base has held a ground pose.

        No-op unless ``auto_ground_detect`` is enabled. Uses the same
        wxyz-quaternion convention as the deploy observation stack.
        """
        if self.feet_on_ground:
            return
        if not self.cfg.get("auto_ground_detect", False):
            return
        s = backend.read_state()
        z = float(s["base_pos_w"][2])
        roll, pitch, _ = quat_to_rpy_wxyz(s["base_quat_wxyz"])
        z_ok = z <= float(self.cfg.get("ground_base_z_max", 0.28))
        rp_ok = max(abs(roll), abs(pitch)) <= float(
            self.cfg.get("ground_rp_max_rad", 0.35))
        if z_ok and rp_ok:
            self._ground_timer += float(step_dt)
            if self._ground_timer >= float(self.cfg.get("ground_hold_s", 0.5)):
                self.feet_on_ground = True
                print("[Staging] feet_on_ground=True (auto-detect)")
        else:
            self._ground_timer = 0.0
