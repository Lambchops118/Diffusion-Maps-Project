"""Tests for the high-dimensional observation model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.observations import (  # noqa: E402
    BASE_FEATURE_NAMES,
    ObservationConfig,
    build_features,
    generate_observations,
    standardize,
)


def _sample_state(n: int = 300) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(-1.5, 1.5, size=n)


def test_base_features_present_and_named() -> None:
    x = _sample_state()
    features, names = build_features(x, include_extra_features=False)
    assert features.shape == (x.size, len(BASE_FEATURE_NAMES))
    assert names == BASE_FEATURE_NAMES
    # First column is exactly x.
    np.testing.assert_allclose(features[:, 0], x)


def test_extra_features_append_columns() -> None:
    x = _sample_state()
    base, _ = build_features(x, include_extra_features=False)
    ext, _ = build_features(x, include_extra_features=True)
    assert ext.shape[1] == base.shape[1] + 2


def test_observations_have_requested_dimension() -> None:
    x = _sample_state()
    for dim in (10, 12, 15, 20):
        cfg = ObservationConfig(output_dim=dim, seed=1)
        obs, _, _ = generate_observations(x, cfg)
        assert obs.shape == (x.size, dim)


def test_observations_are_standardized() -> None:
    x = _sample_state()
    cfg = ObservationConfig(output_dim=15, standardize=True, seed=2)
    obs, _, _ = generate_observations(x, cfg)
    np.testing.assert_allclose(obs.mean(axis=0), 0.0, atol=1e-10)
    np.testing.assert_allclose(obs.std(axis=0), 1.0, atol=1e-8)


def test_observations_are_reproducible() -> None:
    x = _sample_state()
    cfg = ObservationConfig(output_dim=15, measurement_noise=0.1, seed=5)
    obs1, _, _ = generate_observations(x, cfg)
    obs2, _, _ = generate_observations(x, cfg)
    np.testing.assert_array_equal(obs1, obs2)


def test_standardize_handles_constant_column() -> None:
    m = np.column_stack([np.ones(50), np.arange(50.0)])
    s = standardize(m)
    assert np.all(np.isfinite(s))
    # Constant column becomes all-zero after centering.
    np.testing.assert_allclose(s[:, 0], 0.0)


@pytest.mark.parametrize("bad_kwargs", [
    {"output_dim": 1},
    {"measurement_noise": -0.5},
])
def test_invalid_observation_config_raises(bad_kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ObservationConfig(**bad_kwargs)
