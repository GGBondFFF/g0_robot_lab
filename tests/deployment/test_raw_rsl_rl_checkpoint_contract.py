from __future__ import annotations

import hashlib

import pytest
import torch

from tests.conftest import RAW_RSL_RL_CHECKPOINT, RAW_RSL_RL_CHECKPOINT_SHA256


pytestmark = [pytest.mark.hardware_forbidden, pytest.mark.release_gate]


def _raw_checkpoint_or_skip():
    if not RAW_RSL_RL_CHECKPOINT.exists():
        pytest.skip(f"Raw RSL-RL checkpoint is absent: {RAW_RSL_RL_CHECKPOINT}")
    return RAW_RSL_RL_CHECKPOINT


def test_raw_rsl_rl_checkpoint_sha256_matches_expected():
    checkpoint = _raw_checkpoint_or_skip()
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == RAW_RSL_RL_CHECKPOINT_SHA256


def test_raw_rsl_rl_checkpoint_can_be_loaded_for_sanity():
    checkpoint = _raw_checkpoint_or_skip()
    loaded = torch.load(checkpoint, map_location="cpu")
    assert isinstance(loaded, dict)
    assert loaded
