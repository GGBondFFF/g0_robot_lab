from __future__ import annotations

import numpy as np

from scripts.sim2sim import g0_sim2sim_config as cfg


def test_history_flatten_order_is_term_major_oldest_to_newest() -> None:
    frames = np.vstack(
        [
            np.arange(cfg.get_single_frame_observation_dim(), dtype=np.float64) + 1000.0 * index
            for index in range(cfg.POLICY_HISTORY_LENGTH)
        ]
    )
    flattened = cfg.flatten_history_term_major(frames)

    manual = np.concatenate(
        [
            np.concatenate([frame[term_slice] for frame in frames])
            for term_slice in cfg.get_observation_term_slices().values()
        ]
    )
    frame_major = frames.reshape(-1)
    np.testing.assert_allclose(flattened, manual)
    assert not np.array_equal(flattened, frame_major)


def test_split_policy_observation_recovers_term_histories() -> None:
    frames = np.vstack(
        [
            np.arange(cfg.get_single_frame_observation_dim(), dtype=np.float64) + 1000.0 * index
            for index in range(cfg.POLICY_HISTORY_LENGTH)
        ]
    )
    terms = cfg.split_policy_observation(cfg.flatten_history_term_major(frames))
    for term, term_slice in cfg.get_observation_term_slices().items():
        np.testing.assert_allclose(terms[term], frames[:, term_slice])
