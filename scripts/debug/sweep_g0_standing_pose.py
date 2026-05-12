#!/usr/bin/env python3
from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    sweep_script = Path(__file__).with_name("sweep_zero_action_standing_pose.py")
    runpy.run_path(str(sweep_script), run_name="__main__")
