"""Experiment orchestration: the full pipeline and the sensitivity studies.

This module wires together simulation, observation, diffusion maps, PCA, and
evaluation into a single reproducible pipeline, plus the four studies that
carry the scientific content:

``alpha``
    The anisotropic normalization exponent, swept over ``[0, 1]`` for both
    observation models.  This is the decisive experiment: theory says
    ``alpha = 1`` divides out the sampling density and therefore returns pure
    geometry, while ``alpha = 1/2`` retains sampling-density information and
    yields a Fokker--Planck-type operator in the observation manifold's induced
    metric. Since the observations are memoryless, density is the only trace of
    the dynamics available to this point-cloud construction.

``bandwidth``
    The kernel bandwidth, swept around a conservative MST scale and up past the
    median heuristic. Very small bandwidths make the kernel nearly reducible;
    very large ones wash out local geometry.

``noise``
    Measurement-noise robustness, for both observation models.

``barrier``
    The process noise ``sigma``, which sets the barrier-to-noise ratio
    ``dV / D``.  Deep barriers make the point cloud itself geometrically
    clustered, so geometry and dynamics agree and the choice of ``alpha``
    stops mattering; shallow barriers separate them.

All randomness flows from a small set of seeds so the whole experiment is
bit-reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import visualization as viz
from .diffusion_maps import (
    DiffusionMapConfig,
    DiffusionMapResult,
    compute_diffusion_map,
    connectivity_scale,
    select_epsilon,
    select_epsilon_scan,
    spectral_gap_ratios,
    squared_distance_matrix,
)
from .evaluation import (
    two_cluster_variance_score,
    compute_pca,
    evaluate_embedding,
    sign_align,
)
from .observations import ObservationConfig, generate_observations
from .simulation import (
    SimulationConfig,
    count_committed_transitions,
    count_well_transitions,
    kramers_escape_rate,
    simulate_double_well,
    simulate_double_well_full,
)


@dataclass
class ExperimentConfig:
    """Top-level configuration for a full experiment run."""

    # Simulation
    n_samples: int = 3000
    dt: float = 0.01
    process_noise: float = 0.7
    x0: float = -1.0
    seed: int = 42
    subsample: int = 20

    # Observations
    output_dim: int = 15
    measurement_noise: float = 0.1
    include_extra_features: bool = True
    observation_seed: int = 123
    feature_map: str = "polynomial"
    rbf_centers: int = 12
    rbf_width: float = 0.35

    # Diffusion map
    epsilon: float | str = "scan"
    epsilon_percentile: float = 50.0
    alpha: float = 0.5
    n_components: int = 5
    diffusion_time: float = 1.0

    # Studies
    feature_maps: tuple[str, ...] = ("polynomial", "rbf")
    alpha_levels: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    noise_levels: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20, 0.40)
    bandwidth_multipliers: tuple[float, ...] = (
        0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0,
    )
    barrier_sigmas: tuple[float, ...] = (0.45, 0.50, 0.55, 0.70)

    # Output
    output_dir: str = "outputs"

    def simulation_config(self, process_noise: float | None = None) -> SimulationConfig:
        return SimulationConfig(
            n_samples=self.n_samples,
            dt=self.dt,
            process_noise=self.process_noise
            if process_noise is None
            else process_noise,
            x0=self.x0,
            seed=self.seed,
            subsample=self.subsample,
        )

    def observation_config(
        self,
        measurement_noise: float | None = None,
        feature_map: str | None = None,
    ) -> ObservationConfig:
        return ObservationConfig(
            output_dim=self.output_dim,
            measurement_noise=self.measurement_noise
            if measurement_noise is None
            else measurement_noise,
            include_extra_features=self.include_extra_features,
            standardize=True,
            seed=self.observation_seed,
            feature_map=self.feature_map if feature_map is None else feature_map,
            rbf_centers=self.rbf_centers,
            rbf_width=self.rbf_width,
        )

    def diffusion_config(
        self, epsilon: float | str | None = None, alpha: float | None = None
    ) -> DiffusionMapConfig:
        return DiffusionMapConfig(
            epsilon=self.epsilon if epsilon is None else epsilon,
            epsilon_percentile=self.epsilon_percentile,
            alpha=self.alpha if alpha is None else alpha,
            n_components=self.n_components,
            diffusion_time=self.diffusion_time,
        )


@dataclass
class ModelArtifacts:
    """Everything produced for one observation model at the default settings."""

    feature_map: str
    observations: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]
    dm_result: DiffusionMapResult
    pca_scores: np.ndarray
    pca_evr: np.ndarray
    epsilon_scan: float
    epsilon_median: float
    epsilon_mst_scale: float
    scan_slope: float


@dataclass
class PipelineArtifacts:
    """Everything produced by a single default pipeline run."""

    t: np.ndarray
    x: np.ndarray
    x_full: np.ndarray
    n_transitions: int
    n_sign_changes: int
    horizon: float
    kramers_time: float
    occupancy_right: float
    two_cluster_x: float
    models: dict[str, ModelArtifacts] = field(default_factory=dict)


def _build_model(
    config: ExperimentConfig, x: np.ndarray, feature_map: str
) -> ModelArtifacts:
    """Run the observation model, bandwidth selection, diffusion map, and PCA."""

    obs_cfg = config.observation_config(feature_map=feature_map)
    observations, features, feature_names = generate_observations(x, obs_cfg)

    d2 = squared_distance_matrix(observations)
    eps_scan, slope = select_epsilon_scan(d2)
    eps_median = select_epsilon(d2, config.epsilon_percentile)
    eps_mst_scale = connectivity_scale(d2)

    dm_cfg = config.diffusion_config()
    dm_result = compute_diffusion_map(observations, dm_cfg)

    pca_scores, pca_evr = compute_pca(observations, n_components=config.n_components)

    return ModelArtifacts(
        feature_map=feature_map,
        observations=observations,
        features=features,
        feature_names=feature_names,
        dm_result=dm_result,
        pca_scores=pca_scores,
        pca_evr=pca_evr,
        epsilon_scan=eps_scan,
        epsilon_median=eps_median,
        epsilon_mst_scale=eps_mst_scale,
        scan_slope=slope,
    )


def run_pipeline(config: ExperimentConfig) -> PipelineArtifacts:
    """Run the default end-to-end pipeline once and return all artifacts."""

    sim_cfg = config.simulation_config()
    t, x = simulate_double_well(sim_cfg)

    # Transition statistics belong on the finely resolved path, not the
    # subsampled one: subsampling would miss short excursions entirely.
    _, x_full = simulate_double_well_full(sim_cfg)
    n_transitions = count_committed_transitions(x_full)
    n_sign_changes = count_well_transitions(x_full)

    models = {
        name: _build_model(config, x, name) for name in config.feature_maps
    }

    return PipelineArtifacts(
        t=t,
        x=x,
        x_full=x_full,
        n_transitions=n_transitions,
        n_sign_changes=n_sign_changes,
        horizon=sim_cfg.horizon,
        kramers_time=1.0 / kramers_escape_rate(config.process_noise),
        occupancy_right=float((x > 0).mean()),
        two_cluster_x=two_cluster_variance_score(x),
        models=models,
    )


def _metrics_row(
    x: np.ndarray,
    dm_result: DiffusionMapResult,
    pca_coord: np.ndarray,
    base: dict[str, float | str],
) -> dict[str, float | str]:
    """Assemble one row of a study table."""

    dm_coord = dm_result.coordinates[:, 0]
    row: dict[str, float | str] = dict(base)
    row["epsilon"] = float(dm_result.epsilon)
    row["n_trivial"] = int(dm_result.n_trivial)
    row["degenerate"] = bool(dm_result.is_degenerate)

    row.update(evaluate_embedding(dm_coord, x).as_dict("dm_"))
    row.update(evaluate_embedding(pca_coord, x).as_dict("pca_"))

    eig = dm_result.eigenvalues
    for i, val in enumerate(eig[: min(5, eig.size)]):
        row[f"eigenvalue_{i}"] = float(val)

    ratios = spectral_gap_ratios(eig, n=4)
    for i, val in enumerate(ratios):
        row[f"gap_ratio_{i + 1}"] = float(val)
    return row


def run_alpha_study(config: ExperimentConfig, artifacts: PipelineArtifacts) -> pd.DataFrame:
    """Sweep the anisotropic normalization exponent for both observation models.

    The bandwidth is held at each model's scan-selected value so that only
    ``alpha`` varies.
    """

    x = artifacts.x
    rows: list[dict[str, float | str]] = []
    for name, model in artifacts.models.items():
        pca_coord = model.pca_scores[:, 0]
        for alpha in config.alpha_levels:
            dm_cfg = config.diffusion_config(
                epsilon=model.epsilon_scan, alpha=alpha
            )
            dm_result = compute_diffusion_map(model.observations, dm_cfg)
            rows.append(
                _metrics_row(
                    x,
                    dm_result,
                    pca_coord,
                    {
                        "feature_map": name,
                        "alpha": float(alpha),
                        "measurement_noise": float(config.measurement_noise),
                        "process_noise": float(config.process_noise),
                    },
                )
            )
    return pd.DataFrame(rows)


def run_bandwidth_study(
    config: ExperimentConfig, artifacts: PipelineArtifacts
) -> pd.DataFrame:
    """Sweep bandwidth around the conservative MST scale for both models.

    Multipliers are applied to each model's scan-selected bandwidth, so
    ``multiplier = 1`` is the automatic choice and the sweep spans both the
    nearly reducible regime below it and the over-smoothing regime above.
    """

    x = artifacts.x
    rows: list[dict[str, float | str]] = []
    for name, model in artifacts.models.items():
        pca_coord = model.pca_scores[:, 0]
        for mult in config.bandwidth_multipliers:
            epsilon = mult * model.epsilon_scan
            dm_cfg = config.diffusion_config(epsilon=epsilon)
            dm_result = compute_diffusion_map(model.observations, dm_cfg)
            rows.append(
                _metrics_row(
                    x,
                    dm_result,
                    pca_coord,
                    {
                        "feature_map": name,
                        "bandwidth_multiplier": float(mult),
                        "alpha": float(config.alpha),
                        "epsilon_mst_scale": float(model.epsilon_mst_scale),
                        "epsilon_median": float(model.epsilon_median),
                        "measurement_noise": float(config.measurement_noise),
                    },
                )
            )
    return pd.DataFrame(rows)


def run_noise_study(config: ExperimentConfig, artifacts: PipelineArtifacts) -> pd.DataFrame:
    """Sweep the measurement-noise level for both observation models.

    The bandwidth is held **fixed** at each model's value selected at the
    default noise level, rather than re-selected at every level.  Re-selecting
    would confound two effects: the degradation caused by noise itself, and the
    instability of the automatic selector (which on these data can jump by more
    than an order of magnitude between adjacent noise levels; see
    :func:`select_epsilon_scan` and the discussion of the two-scale kernel-sum
    curve).  Holding it fixed isolates the variable actually under study.
    """

    x = artifacts.x
    rows: list[dict[str, float | str]] = []
    for name in config.feature_maps:
        epsilon = artifacts.models[name].epsilon_scan
        for noise in config.noise_levels:
            obs_cfg = config.observation_config(
                measurement_noise=noise, feature_map=name
            )
            observations, _, _ = generate_observations(x, obs_cfg)

            dm_cfg = config.diffusion_config(epsilon=epsilon)
            dm_result = compute_diffusion_map(observations, dm_cfg)
            pca_scores, _ = compute_pca(
                observations, n_components=config.n_components
            )
            rows.append(
                _metrics_row(
                    x,
                    dm_result,
                    pca_scores[:, 0],
                    {
                        "feature_map": name,
                        "measurement_noise": float(noise),
                        "alpha": float(config.alpha),
                    },
                )
            )
    return pd.DataFrame(rows)


def run_barrier_study(config: ExperimentConfig) -> pd.DataFrame:
    """Sweep the process noise, i.e. the barrier-to-noise ratio.

    Each ``sigma`` gets its own trajectory.  The horizon is held fixed in units
    of the Kramers escape time so that every setting sees a comparable number
    of transitions; otherwise a deeper barrier would simply mean fewer events
    and the comparison would confound depth with sample size.
    """

    rows: list[dict[str, float | str]] = []
    base_rate = kramers_escape_rate(config.process_noise)
    base_horizon = config.simulation_config().horizon

    for sigma in config.barrier_sigmas:
        rate = kramers_escape_rate(sigma)
        # Hold horizon / escape-time constant.
        target_horizon = base_horizon * (base_rate / rate)
        subsample = max(1, int(round(target_horizon / (config.dt * config.n_samples))))
        sim_cfg = SimulationConfig(
            n_samples=config.n_samples,
            dt=config.dt,
            process_noise=sigma,
            x0=config.x0,
            seed=config.seed,
            subsample=subsample,
        )
        _, x = simulate_double_well(sim_cfg)
        _, x_full = simulate_double_well_full(sim_cfg)
        n_trans = count_committed_transitions(x_full)

        for name in config.feature_maps:
            obs_cfg = config.observation_config(feature_map=name)
            observations, _, _ = generate_observations(x, obs_cfg)
            d2 = squared_distance_matrix(observations)
            eps, _ = select_epsilon_scan(d2)
            pca_scores, _ = compute_pca(
                observations, n_components=config.n_components
            )
            for alpha in (0.5, 1.0):
                dm_cfg = config.diffusion_config(epsilon=eps, alpha=alpha)
                dm_result = compute_diffusion_map(observations, dm_cfg)
                rows.append(
                    _metrics_row(
                        x,
                        dm_result,
                        pca_scores[:, 0],
                        {
                            "feature_map": name,
                            "process_noise": float(sigma),
                            "barrier_ratio": float(0.25 / (sigma**2 / 2.0)),
                            "alpha": float(alpha),
                            "horizon": float(sim_cfg.horizon),
                            "n_transitions": int(n_trans),
                            "two_cluster_x": two_cluster_variance_score(x),
                        },
                    )
                )
    return pd.DataFrame(rows)


def generate_all_figures(
    artifacts: PipelineArtifacts,
    alpha_df: pd.DataFrame,
    bandwidth_df: pd.DataFrame,
    noise_df: pd.DataFrame,
    barrier_df: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    """Create and save every figure. Returns the saved paths."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    x = artifacts.x
    paths: list[Path] = []

    paths.append(
        viz.plot_state_vs_time(
            artifacts.t, x, figures_dir / "01_state_vs_time.png"
        )
    )
    paths.append(
        viz.plot_state_histogram(x, figures_dir / "02_state_histogram.png")
    )

    for name, model in artifacts.models.items():
        tag = "poly" if name == "polynomial" else "rbf"
        paths.append(
            viz.plot_feature_traces(
                artifacts.t,
                model.features,
                model.feature_names,
                figures_dir / f"03_{tag}_feature_traces.png",
            )
        )
        dm1 = sign_align(model.dm_result.coordinates[:, 0], x)
        dm2 = model.dm_result.coordinates[:, 1]
        pca1 = sign_align(model.pca_scores[:, 0], x)
        pca2 = model.pca_scores[:, 1]
        paths.append(
            viz.plot_coordinate_vs_state(
                x, dm1, figures_dir / f"04_{tag}_dm_coordinate_vs_state.png"
            )
        )
        paths.append(
            viz.plot_coordinate_vs_state(
                x,
                pca1,
                figures_dir / f"05_{tag}_pca_coordinate_vs_state.png",
                ylabel="PC 1",
                title=r"PC$_1$ vs. true state $X$",
            )
        )
        paths.append(
            viz.plot_embedding_scatter(
                dm1,
                dm2,
                x,
                figures_dir / f"06_{tag}_dm_embedding_scatter.png",
                title=f"Diffusion map, {name} observations",
                xlabel=r"$\psi_1$",
                ylabel=r"$\psi_2$",
            )
        )
        paths.append(
            viz.plot_embedding_scatter(
                pca1,
                pca2,
                x,
                figures_dir / f"07_{tag}_pca_embedding_scatter.png",
                title=f"PCA, {name} observations",
                xlabel="PC 1",
                ylabel="PC 2",
            )
        )

    paths.append(
        viz.plot_alpha_study(
            alpha_df, artifacts.two_cluster_x, figures_dir / "08_alpha_study.png"
        )
    )
    paths.append(
        viz.plot_bandwidth_study(
            bandwidth_df, figures_dir / "09_bandwidth_study.png"
        )
    )
    paths.append(
        viz.plot_performance_vs_noise(noise_df, figures_dir / "10_noise_study.png")
    )
    paths.append(
        viz.plot_barrier_study(barrier_df, figures_dir / "11_barrier_study.png")
    )
    paths.append(
        viz.plot_alpha_spectra(alpha_df, figures_dir / "12_alpha_spectra.png")
    )
    return paths
