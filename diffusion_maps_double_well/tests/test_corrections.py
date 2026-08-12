"""Tests for the corrected experiment: subsampling, the RBF observation model,
connectivity-aware bandwidth selection, and the degeneracy diagnostics.

These cover the specific failure modes that the first version of this study
missed, so each test names the trap it guards against.
"""

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
    compute_diffusion_map,
    connectivity_scale,
    connectivity_threshold,
    count_graph_components,
    select_epsilon,
    select_epsilon_scan,
    spectral_gap_ratios,
    squared_distance_matrix,
)
from src.evaluation import metastability_score, two_cluster_variance_score  # noqa: E402
from src.observations import (  # noqa: E402
    ObservationConfig,
    build_features,
    generate_observations,
)
from src.simulation import (  # noqa: E402
    SimulationConfig,
    count_committed_transitions,
    kramers_escape_rate,
    simulate_double_well,
    simulate_double_well_full,
)


# --------------------------------------------------------------- simulation --

def test_subsampling_returns_requested_count_over_longer_horizon() -> None:
    cfg = SimulationConfig(n_samples=500, dt=0.01, subsample=10, seed=0)
    t, x = simulate_double_well(cfg)
    assert x.shape == (500,)
    # 500 kept points at stride 10 span ~10x the horizon of 500 raw steps.
    assert cfg.horizon == pytest.approx((500 - 1) * 10 * 0.01)
    np.testing.assert_allclose(np.diff(t), 0.1)


def test_subsampling_is_a_strict_subset_of_the_fine_path() -> None:
    cfg = SimulationConfig(n_samples=200, subsample=7, seed=3)
    _, x = simulate_double_well(cfg)
    _, x_full = simulate_double_well_full(cfg)
    np.testing.assert_array_equal(x, x_full[::7][:200])


def test_subsample_one_is_unchanged_behaviour() -> None:
    a = simulate_double_well(SimulationConfig(n_samples=300, seed=11))[1]
    b = simulate_double_well(SimulationConfig(n_samples=300, seed=11, subsample=1))[1]
    np.testing.assert_array_equal(a, b)


def test_committed_transitions_ignores_barrier_chatter() -> None:
    """A path that jitters across zero without reaching either core scores 0."""

    chatter = np.array([-0.9, -0.1, 0.1, -0.1, 0.2, -0.2, -0.9])
    assert count_committed_transitions(chatter, core=0.5) == 0
    # A genuine there-and-back is two committed transitions.
    real = np.array([-0.9, -0.1, 0.9, 0.1, -0.9])
    assert count_committed_transitions(real, core=0.5) == 2


def test_committed_never_exceeds_raw_sign_changes() -> None:
    _, x = simulate_double_well(SimulationConfig(n_samples=4000, seed=42))
    raw = int(np.sum(np.diff(np.sign(x)) != 0))
    assert count_committed_transitions(x) <= raw


def test_kramers_rate_decreases_with_decreasing_noise() -> None:
    rates = [kramers_escape_rate(s) for s in (0.45, 0.5, 0.55, 0.7)]
    assert all(a < b for a, b in zip(rates, rates[1:]))


def test_invalid_subsample_raises() -> None:
    with pytest.raises(ValueError):
        SimulationConfig(subsample=0)


# ------------------------------------------------------------- observations --

def test_rbf_features_have_requested_shape_and_are_bounded() -> None:
    x = np.linspace(-2.0, 2.0, 200)
    f, names = build_features(x, feature_map="rbf", rbf_centers=12, rbf_width=0.35)
    assert f.shape == (200, 12)
    assert len(names) == 12
    assert np.all((f > 0.0) & (f <= 1.0))


def test_rbf_channel_peaks_at_its_own_centre() -> None:
    x = np.linspace(-2.2, 2.2, 401)
    f, _ = build_features(x, feature_map="rbf", rbf_centers=5, rbf_width=0.3)
    centers = np.linspace(-2.2, 2.2, 5)
    for j, c in enumerate(centers):
        assert x[np.argmax(f[:, j])] == pytest.approx(c, abs=0.02)


def test_rbf_is_more_curved_than_polynomial() -> None:
    """The point of the RBF map: PCA should explain far less of it linearly."""

    from sklearn.decomposition import PCA

    rng = np.random.default_rng(0)
    x = rng.uniform(-1.8, 1.8, 800)
    evr = {}
    for fm in ("polynomial", "rbf"):
        obs, _, _ = generate_observations(
            x, ObservationConfig(feature_map=fm, measurement_noise=0.0, seed=1)
        )
        evr[fm] = PCA(n_components=1, random_state=0).fit(obs).explained_variance_ratio_[0]
    assert evr["rbf"] < evr["polynomial"]


def test_unknown_feature_map_raises() -> None:
    with pytest.raises(ValueError):
        ObservationConfig(feature_map="quadratic")
    with pytest.raises(ValueError):
        build_features(np.zeros(5), feature_map="quadratic")


# ----------------------------------------------------- bandwidth selection ---

@pytest.fixture(scope="module")
def curve_observations() -> np.ndarray:
    _, x = simulate_double_well(SimulationConfig(n_samples=400, subsample=5, seed=42))
    obs, _, _ = generate_observations(
        x, ObservationConfig(output_dim=15, measurement_noise=0.05, seed=123)
    )
    return obs


def test_connectivity_threshold_uses_explicit_affinity_cutoff(curve_observations) -> None:
    d2 = squared_distance_matrix(curve_observations)
    threshold = connectivity_threshold(d2, tol=1e-12)
    assert 0.0 < threshold < connectivity_scale(d2)
    # The threshold is an infimum because adjacency uses a strict > cutoff.
    assert count_graph_components(d2, 1.001 * threshold, tol=1e-12) == 1


def test_scan_selects_at_or_above_the_conservative_mst_scale(curve_observations) -> None:
    d2 = squared_distance_matrix(curve_observations)
    eps, slope = select_epsilon_scan(d2)
    assert eps >= connectivity_scale(d2)
    assert count_graph_components(d2, eps) == 1
    # 2 * slope estimates intrinsic dimension; the data lie on a curve.
    assert 0.3 < 2 * slope < 4.0


def test_scan_is_far_smaller_than_the_median_heuristic(curve_observations) -> None:
    """The median heuristic overshoots; that was the original bug."""

    d2 = squared_distance_matrix(curve_observations)
    eps_scan, _ = select_epsilon_scan(d2)
    assert select_epsilon(d2, 50.0) > eps_scan


# ----------------------------------------------------------- degeneracy ------

def test_tiny_bandwidth_is_flagged_as_degenerate(curve_observations) -> None:
    """The trap: a nearly reducible kernel can yield a misleading coordinate."""

    d2 = squared_distance_matrix(curve_observations)
    tiny = 0.01 * connectivity_scale(d2)
    with pytest.warns(UserWarning, match="nearly reducible"):
        result = compute_diffusion_map(
            curve_observations, DiffusionMapConfig(epsilon=tiny, n_components=3)
        )
    assert result.is_degenerate
    assert result.n_trivial > 1
    # ...and the coordinate really is near-constant, which is what fools rank
    # correlation.
    assert result.n_distinct < curve_observations.shape[0] // 2


def test_healthy_bandwidth_is_not_degenerate(curve_observations) -> None:
    result = compute_diffusion_map(
        curve_observations, DiffusionMapConfig(epsilon="scan", n_components=3)
    )
    assert not result.is_degenerate
    assert result.n_trivial == 1
    assert result.n_distinct > 0.9 * curve_observations.shape[0]


# ------------------------------------------------------------- diagnostics ---

def test_gap_ratios_start_at_one_and_increase() -> None:
    lam = np.array([1.0, 0.9, 0.8, 0.6, 0.5])
    r = spectral_gap_ratios(lam, n=4)
    assert r[0] == pytest.approx(1.0)
    assert np.all(np.diff(r) > 0)


def test_gap_ratios_recover_the_k_squared_law_exactly() -> None:
    """For lambda_k = 1 - c*k^2 the ratios must be 1, 4, 9, 16."""

    c = 1e-3
    lam = np.array([1.0] + [1.0 - c * k**2 for k in (1, 2, 3, 4)])
    np.testing.assert_allclose(
        spectral_gap_ratios(lam, n=4), [1.0, 4.0, 9.0, 16.0], rtol=1e-9
    )


def test_two_cluster_score_is_bounded_and_ordered() -> None:
    rng = np.random.default_rng(0)
    continuum = rng.uniform(-1, 1, 2000)
    gapped = np.concatenate([rng.normal(-3, 0.1, 1000), rng.normal(3, 0.1, 1000)])
    for v in (continuum, gapped):
        assert 0.0 <= two_cluster_variance_score(v) <= 1.0
    assert two_cluster_variance_score(gapped) > 0.95
    assert two_cluster_variance_score(gapped) > two_cluster_variance_score(continuum)


def test_two_cluster_score_of_constant_signal_is_zero() -> None:
    assert two_cluster_variance_score(np.full(100, 2.5)) == pytest.approx(0.0)


def test_uniform_continuum_can_have_a_high_two_cluster_score() -> None:
    continuum = np.linspace(-1.0, 1.0, 10000)
    assert two_cluster_variance_score(continuum) == pytest.approx(0.75, abs=2e-3)


def test_metastability_prefers_a_step_over_a_ramp() -> None:
    """A well indicator should beat a pure arclength coordinate."""

    x = np.linspace(-1.5, 1.5, 1000)
    ramp = x
    step = np.tanh(20.0 * x)
    assert metastability_score(step, x) > metastability_score(ramp, x)


def test_metastability_is_sign_invariant() -> None:
    x = np.linspace(-1.5, 1.5, 500)
    v = np.tanh(5.0 * x)
    assert metastability_score(v, x) == pytest.approx(metastability_score(-v, x))
