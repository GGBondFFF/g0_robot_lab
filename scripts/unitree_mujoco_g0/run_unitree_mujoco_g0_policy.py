#!/usr/bin/env python3
"""Run live TorchScript policy through the Unitree LowCmd-style bridge."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.unitree_mujoco_g0.run_unitree_mujoco_g0 import main as _main


if __name__ == "__main__":
    sys.argv.extend(["--mode", "policy"]) if "--mode" not in sys.argv else None
    raise SystemExit(_main())
