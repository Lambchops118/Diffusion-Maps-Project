"""Tests for the Euler--Maruyama double-well simulation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation import (  # noqa: E402
    SimulationConfig,
    count_well_transitions,
    drift,
    simulate_double_well,
)


def test_simulation_returns_correct_number_of_finite_values() -> None:
    cfg = SimulationConfig(n_samples=500, seed=0)
    t, x = simulate_double_well(cfg)
    assert x.shape == (500,)
    assert t.shape == (500,)
    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(t))


def test_identical_seeds_give_identical_simulations() -> None:
    cfg = SimulationConfig(n_samples=400, seed=7)
    _, x1 = simulate_double_well(cfg)
    _, x2 = simulate_double_well(cfg)
    np.testing.assert_array_equal(x1, x2)


def test_different_seeds_give_different_simulations() -> None:
    _, x1 = simulate_double_well(SimulationConfig(n_samples=400, seed=1))
    _, x2 = simulate_double_well(SimulationConfig(n_samples=400, seed=2))
    assert not np.array_equal(x1, x2)


def test_initial_condition_is_respected() -> None:
    cfg = SimulationConfig(n_samples=10, x0=0.3, seed=3)
    _, x = simulate_double_well(cfg)
    assert x[0] == pytest.approx(0.3)


def test_time_grid_matches_dt() -> None:
    cfg = SimulationConfig(n_samples=100, dt=0.05, seed=0)
    t, _ = simulate_double_well(cfg)
    np.testing.assert_allclose(np.diff(t), 0.05)


def test_default_configuration_shows_transitions() -> None:
    # The default settings should visit both wells several times.
    cfg = SimulationConfig(n_samples=5000, seed=42)
    _, x = simulate_double_well(cfg)
    assert count_well_transitions(x) >= 2
    assert x.min() < 0 < x.max()


def test_drift_fixed_points() -> None:
    assert drift(0.0) == pytest.approx(0.0)
    assert drift(1.0) == pytest.approx(0.0)
    assert drift(-1.0) == pytest.approx(0.0)


@pytest.mark.parametrize("bad_kwargs", [
    {"n_samples": 1},
    {"dt": -0.1},
    {"process_noise": -1.0},
])
def test_invalid_config_raises(bad_kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SimulationConfig(**bad_kwargs)
