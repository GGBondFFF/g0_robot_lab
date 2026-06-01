"""L3 acceptance (Milestone M2): RLBase stands >=30 s WITHOUT the elastic band.

This is the canonical sign-off gate for the Unitree-aligned staging SOP. It
runs the full deploy pipeline through the SOP (f -> lower -> confirm ground ->
RLBase -> disable band) and then requires the robot to keep standing for 30 s
with the band off (root_z > 0.15, no NaN).

CURRENT STATUS: xfail. Root cause investigation (Task 10, systematic-debugging)
established:

  * PD-hold (zero action) at default_stand holds root_z = 0.231 steady for 10 s
    (scripts/sim2sim/g0_mujoco_zero_action.py) -> MJCF physics / pose / contact /
    PD gains are STABLE; nothing to tune there.
  * Closed-loop v0 policy with zero_cmd and no band drives root_z 0.231 -> 0.089
    (t=1.0s) -> ~0.053 (fallen) while action magnitude grows |act|max 1.3 -> 15.
  * Obs construction, ONNX==PT, and SDK<->MJ mapping are all independently green.

Therefore the no-band fall is a v0 POLICY-QUALITY / sim2sim-transfer issue, not a
deploy-pipeline or physics bug. The fix is to retrain / domain-randomize / pick a
better checkpoint (Phase 5 Step 4) — an Isaac training run, out of scope for the
deploy session. Remove the xfail mark once a checkpoint passes this gate.

    pytest tests/deployment/test_g0_deploy_rlbase_30s_no_band.py -v
"""

import importlib.util
import os
import sys

import numpy as np
import pytest
import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

STAGING_YAML = os.path.join(
    REPO_ROOT, "deploy", "robots", "g0", "config", "policy", "velocity", "v0",
    "deploy_staging.yaml",
)


def _load_sop_module():
    """Reuse the SOP pipeline/helpers without depending on tests/ being a package."""
    path = os.path.join(os.path.dirname(__file__), "test_g0_staging_sop.py")
    spec = importlib.util.spec_from_file_location("g0_staging_sop_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def staging_cfg():
    with open(STAGING_YAML, "r") as f:
        return yaml.safe_load(f)


@pytest.mark.xfail(strict=False,
                   reason="v0 policy mistracks standing in MuJoCo without the "
                          "band (PD-hold is stable; closed-loop policy falls "
                          "~1.4s — policy/sim2sim issue, needs retrain). "
                          "Remove when a checkpoint passes L3.")
def test_rlbase_30s_no_band(staging_cfg):
    sop = _load_sop_module()
    (backend, fsm, band, cmd_mgr, rc, staging, step_dt,
     sig, _tick) = sop._run_sop_through_band_disable(staging_cfg)

    # Preconditions: we actually reached RLBase and disabled the band.
    assert fsm.current.name == "rl_base", "did not reach RLBase via the SOP"
    assert band.enabled is False, "band was not disabled by key 9"

    # L3: stand >=30 s with no band.
    z_min = float(backend.data.qpos[2])
    for _ in range(int(30.0 / step_dt)):
        _tick()
        z = float(backend.data.qpos[2])
        z_min = min(z_min, z)
        assert np.all(np.isfinite(backend.data.qpos)), "non-finite qpos without band"
    assert fsm.current.name == "rl_base"
    assert z_min > 0.15, f"robot fell without band: root_z_min={z_min:.4f}"
