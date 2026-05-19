from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests.conftest import (
    EXPORTED_POLICY_ONNX,
    EXPORTED_POLICY_TORCHSCRIPT,
    RAW_RSL_RL_CHECKPOINT,
    REPO_ROOT,
)


pytestmark = [pytest.mark.release_gate, pytest.mark.slow]


def test_play_py_exports_policy_artifacts_from_raw_checkpoint():
    if not RAW_RSL_RL_CHECKPOINT.exists():
        pytest.skip(f"Raw RSL-RL checkpoint is absent: {RAW_RSL_RL_CHECKPOINT}")

    command = [
        sys.executable,
        "scripts/rsl_rl/play.py",
        "--task",
        "G0-Velocity-v0",
        "--num_envs",
        "1",
        "--checkpoint",
        str(RAW_RSL_RL_CHECKPOINT),
        "--headless",
        "--video",
        "--video_length",
        "1",
    ]
    env = os.environ.copy()
    env["G0_ALLOW_HARDWARE"] = "0"
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, timeout=180, env=env)
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert EXPORTED_POLICY_TORCHSCRIPT.exists() or EXPORTED_POLICY_ONNX.exists()
