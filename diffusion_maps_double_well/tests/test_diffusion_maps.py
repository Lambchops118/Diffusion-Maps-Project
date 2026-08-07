"""Tests for the from-scratch diffusion-map implementation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.diffusion_maps import (  # noqa: E402
    DiffusionMapConfig,
    anisotropic_normalize,
    compute_diffusion_map,
    gaussian_kernel,
    markov_matrix,
    select_epsilon,
    squared_distance_matrix,
)
from src.observations import ObservationConfig, generate_observations  # noqa: E402
from src.simulation import SimulationConfig, simulate_double_well  # noqa: E402


@pytest.fixture(scope="module")
def observations() -> np.ndarray:
    """A modest standardized observation matrix from the default pipeline."""

    _, x = simulate_double_well(SimulationConfig(n_samples=600, seed=42))
    obs, _, _ = generate_observations(
        x, ObservationConfig(output_dim=15, measurement_noise=0.1, seed=123)
    )
    return obs


def test_squared_distance_matrix_is_symmetric_with_zero_diagonal(observations) -> None:
    d2 = squared_distance_matrix(observations)
    assert d2.shape == (observations.shape[0], observations.shape[0])
    np.testing.assert_allclose(d2, d2.T, atol=1e-9)
    np.testing.assert_allclose(np.diag(d2), 0.0, atol=1e-12)
    assert np.all(d2 >= 0.0)


def test_initial_kernel_matrix_is_symmetric(observations) -> None:
    d2 = squared_distance_matrix(observations)
    eps = select_epsilon(d2)
    kernel = gaussian_kernel(d2, eps)
    np.testing.assert_allclose(kernel, kernel.T, atol=1e-12)
    # Diagonal is exp(0) = 1.
    np.testing.assert_allclose(np.diag(kernel), 1.0, atol=1e-12)


def test_markov_matrix_rows_sum_to_one(observations) -> None:
    d2 = squared_distance_matrix(observations)
    eps = select_epsilon(d2)
    kernel = gaussian_kernel(d2, eps)
    kernel_tilde = anisotropic_normalize(kernel, alpha=1.0)
    p, _ = markov_matrix(kernel_tilde)
    row_sums = p.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)


def test_eigenvalues_sorted_descending(observations) -> None:
    cfg = DiffusionMapConfig(epsilon="auto", alpha=1.0, n_components=5)
    result = compute_diffusion_map(observations, cfg)
    ev = result.eigenvalues
    assert np.all(np.diff(ev) <= 1e-9)  # non-increasing


def test_leading_eigenvalue_is_approximately_one(observations) -> None:
    cfg = DiffusionMapConfig(epsilon="auto", alpha=1.0, n_components=5)
    result = compute_diffusion_map(observations, cfg)
    assert result.eigenvalues[0] == pytest.approx(1.0, abs=1e-6)


def test_diffusion_coordinates_are_finite(observations) -> None:
    cfg = DiffusionMapConfig(epsilon="auto", alpha=1.0, n_components=5)
    result = compute_diffusion_map(observations, cfg)
    assert result.coordinates.shape == (observations.shape[0], 5)
    assert np.all(np.isfinite(result.coordinates))


def test_trivial_eigenvector_is_constant(observations) -> None:
    cfg = DiffusionMapConfig(epsilon="auto", alpha=1.0, n_components=3)
    result = compute_diffusion_map(observations, cfg)
    trivial = result.eigenvectors[:, 0]
    # Constant vector: near-zero relative spread.
    assert np.std(trivial) / (abs(np.mean(trivial)) + 1e-12) < 1e-6


def test_select_epsilon_uses_percentile() -> None:
    d2 = np.array([[0.0, 1.0, 4.0], [1.0, 0.0, 9.0], [4.0, 9.0, 0.0]])
    # Nonzero off-diagonal squared distances: {1, 4, 9}; median = 4.
    assert select_epsilon(d2, percentile=50.0) == pytest.approx(4.0)


def test_alpha_zero_matches_plain_kernel() -> None:
    kernel = np.array([[1.0, 0.5], [0.5, 1.0]])
    out = anisotropic_normalize(kernel, alpha=0.0)
    np.testing.assert_allclose(out, kernel)


@pytest.mark.parametrize("bad_kwargs", [
    {"epsilon": -1.0},
    {"epsilon": "nope"},
    {"alpha": 2.0},
    {"n_components": 0},
    {"epsilon_percentile": 150.0},
])
def test_invalid_diffusion_config_raises(bad_kwargs: dict) -> None:
    with pytest.raises(ValueError):
        DiffusionMapConfig(**bad_kwargs)
