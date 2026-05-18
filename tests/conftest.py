from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "source" / "g0_robot_lab"
G0_SOURCE = SOURCE_ROOT / "g0_robot_lab" / "assets" / "robots" / "g0" / "g0.py"
G0_ACTUATORS_SOURCE = SOURCE_ROOT / "g0_robot_lab" / "assets" / "robots" / "g0" / "g0_actuators.py"
VELOCITY_ENV_CFG_SOURCE = (
    SOURCE_ROOT / "g0_robot_lab" / "tasks" / "locomotion" / "robots" / "g0" / "velocity_env_cfg.py"
)
APPROVED_SPEC = REPO_ROOT / "docs" / "superpowers" / "specs" / "2026-05-18-pre-deployment-validation-design.md"
CANDIDATE_RUN_DIR = REPO_ROOT / "logs" / "rsl_rl" / "g0_velocity" / "2026-05-14_18-29-19"
RAW_RSL_RL_CHECKPOINT = CANDIDATE_RUN_DIR / "model_9999.pt"
RAW_RSL_RL_CHECKPOINT_SHA256 = "1dc0c434a4b991eaaa435a21b9d4265e0267eb781b69b132bd75a0b5883928cd"
EXPORTED_POLICY_DIR = CANDIDATE_RUN_DIR / "exported"
EXPORTED_POLICY_TORCHSCRIPT = EXPORTED_POLICY_DIR / "policy.pt"
EXPORTED_POLICY_ONNX = EXPORTED_POLICY_DIR / "policy.onnx"
POLICY_OBSERVATION_DIM = 385
POLICY_ACTION_DIM = 22


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def dryrun_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("G0_ALLOW_HARDWARE", "0")


@pytest.fixture
def forbid_hardware_transports(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> None:
    marker_selected = (
        request.node.get_closest_marker("deployment_dryrun") is not None
        or request.node.get_closest_marker("hardware_forbidden") is not None
    )
    if not marker_selected:
        return

    class GuardedSocket(socket.socket):
        def send(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("Real socket.send is forbidden in deployment dry-run tests.")

        def sendto(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise AssertionError("Real socket.sendto is forbidden in deployment dry-run tests.")

    monkeypatch.setenv("G0_ALLOW_HARDWARE", "0")
    monkeypatch.setattr(socket, "socket", GuardedSocket)


@pytest.fixture(autouse=True)
def _apply_marker_scoped_hardware_guard(
    request: pytest.FixtureRequest,
    forbid_hardware_transports: None,
) -> None:
    if request.node.get_closest_marker("hardware_forbidden") is not None:
        assert os.environ.get("G0_ALLOW_HARDWARE", "0") == "0"
