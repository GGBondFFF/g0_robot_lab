from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.conftest import (
    EXPORTED_POLICY_ONNX,
    EXPORTED_POLICY_TORCHSCRIPT,
    POLICY_ACTION_DIM,
    POLICY_OBSERVATION_DIM,
)


pytestmark = [pytest.mark.hardware_forbidden, pytest.mark.release_gate]


def _torchscript_policy_or_skip():
    if not EXPORTED_POLICY_TORCHSCRIPT.exists():
        pytest.skip(f"Exported TorchScript policy is absent: {EXPORTED_POLICY_TORCHSCRIPT}")
    return EXPORTED_POLICY_TORCHSCRIPT


def _onnx_policy_or_skip():
    if not EXPORTED_POLICY_ONNX.exists():
        pytest.skip(f"Exported ONNX policy is absent: {EXPORTED_POLICY_ONNX}")
    return EXPORTED_POLICY_ONNX


def test_exported_policy_artifact_exists_or_skips():
    if not EXPORTED_POLICY_TORCHSCRIPT.exists() and not EXPORTED_POLICY_ONNX.exists():
        pytest.skip(
            "No exported policy artifacts found. Run the release-gate policy export test to create "
            f"{EXPORTED_POLICY_TORCHSCRIPT} and/or {EXPORTED_POLICY_ONNX}."
        )
    assert EXPORTED_POLICY_TORCHSCRIPT.exists() or EXPORTED_POLICY_ONNX.exists()


def test_exported_policy_torchscript_io_shape_contract():
    artifact = _torchscript_policy_or_skip()
    policy = torch.jit.load(str(artifact), map_location="cpu")
    policy.eval()
    with torch.inference_mode():
        action = policy(torch.zeros(1, POLICY_OBSERVATION_DIM, dtype=torch.float32))
    assert tuple(action.shape) == (1, POLICY_ACTION_DIM)


def test_exported_policy_onnx_io_shape_contract():
    artifact = _onnx_policy_or_skip()
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise AssertionError(
            f"onnxruntime is required to validate existing exported policy.onnx: {artifact}"
        ) from exc
    session = ort.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    obs = np.zeros((1, POLICY_OBSERVATION_DIM), dtype=np.float32)
    action = session.run([output_meta.name], {input_meta.name: obs})[0]
    assert tuple(action.shape) == (1, POLICY_ACTION_DIM)
