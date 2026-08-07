"""Experiment orchestration: the full pipeline and sensitivity studies.

This module wires together simulation, observation, diffusion maps, PCA, and
evaluation into a single reproducible pipeline, plus the two sensitivity
studies (measurement noise and kernel bandwidth) required by the project.

All randomness flows from a small set of seeds so the whole experiment is
bit-reproducible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import visualization as viz
from .diffusion_maps import (
    DiffusionMapConfig,
    DiffusionMapResult,
    compute_diffusion_map,
    select_epsilon,
    squared_distance_matrix,
)
from .evaluation import compute_pca, evaluate_embedding, sign_align
from .observations import ObservationConfig, generate_observations
from .simulation import SimulationConfig, count_well_transitions, simulate_double_well


@dataclass
class ExperimentConfig:
    """Top-level configuration for a full experiment run."""

    # Simulation
    n_samples: int = 3000
    dt: float = 0.01
    process_noise: float = 0.7
    x0: float = -1.0
    seed: int = 42

    # Observations
    output_dim: int = 15
    measurement_noise: float = 0.1
    include_extra_features: bool = True
    observation_seed: int = 123

    # Diffusion map
    epsilon: float | str = "auto"
    epsilon_percentile: float = 50.0
    alpha: float = 1.0
    n_components: int = 5
    diffusion_time: float = 1.0

    # Sensitivity studies
    noise_levels: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)
    epsilon_multipliers: tuple[float, ...] = (0.5, 1.0, 2.0)

    # Output
    output_dir: str = "outputs"

    def simulation_config(self) -> SimulationConfig:
        return SimulationConfig(
            n_samples=self.n_samples,
            dt=self.dt,
            process_noise=self.process_noise,
            x0=self.x0,
            seed=self.seed,
        )

    def observation_config(self, measurement_noise: float | None = None) -> ObservationConfig:
        return ObservationConfig(
            output_dim=self.output_dim,
            measurement_noise=self.measurement_noise
            if measurement_noise is None
            else measurement_noise,
            include_extra_features=self.include_extra_features,
            standardize=True,
            seed=self.observation_seed,
        )

    def diffusion_config(
        self, epsilon: float | str | None = None
    ) -> DiffusionMapConfig:
        return DiffusionMapConfig(
            epsilon=self.epsilon if epsilon is None else epsilon,
            epsilon_percentile=self.epsilon_percentile,
            alpha=self.alpha,
            n_components=self.n_components,
            diffusion_time=self.diffusion_time,
        )


@dataclass
class PipelineArtifacts:
    """Everything produced by a single default pipeline run."""

    t: np.ndarray
    x: np.ndarray
    observations: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...]
    dm_result: DiffusionMapResult
    pca_scores: np.ndarray
    pca_evr: np.ndarray
    n_transitions: int
    default_epsilon: float


def run_pipeline(config: ExperimentConfig) -> PipelineArtifacts:
    """Run the default end-to-end pipeline once and return all artifacts."""

    sim_cfg = config.simulation_config()
    t, x = simulate_double_well(sim_cfg)
    n_transitions = count_well_transitions(x)

    obs_cfg = config.observation_config()
    observations, features, feature_names = generate_observations(x, obs_cfg)

    dm_cfg = config.diffusion_config()
    dm_result = compute_diffusion_map(observations, dm_cfg)

    pca_scores, pca_evr = compute_pca(observations, n_components=config.n_components)

    # The reference default bandwidth (used to scale the epsilon study).
    d2 = squared_distance_matrix(observations)
    default_epsilon = select_epsilon(d2, config.epsilon_percentile)

    return PipelineArtifacts(
        t=t,
        x=x,
        observations=observations,
        features=features,
        feature_names=feature_names,
        dm_result=dm_result,
        pca_scores=pca_scores,
        pca_evr=pca_evr,
        n_transitions=n_transitions,
        default_epsilon=default_epsilon,
    )


def _metrics_row(
    x: np.ndarray,
    dm_coord: np.ndarray,
    pca_coord: np.ndarray,
    eigenvalues: np.ndarray,
    epsilon: float,
    measurement_noise: float,
    extra: dict[str, float] | None = None,
) -> dict[str, float]:
    """Assemble one row of the combined metrics table."""

    dm_metrics = evaluate_embedding(dm_coord, x)
    pca_metrics = evaluate_embedding(pca_coord, x)

    # Save the leading (nontrivial) eigenvalues as separate columns.
    row: dict[str, float] = {
        "epsilon": float(epsilon),
        "measurement_noise": float(measurement_noise),
        "dm_pearson": dm_metrics.pearson,
        "dm_spearman": dm_metrics.spearman,
        "dm_well_score": dm_metrics.well_score,
        "pca_pearson": pca_metrics.pearson,
        "pca_spearman": pca_metrics.spearman,
        "pca_well_score": pca_metrics.well_score,
    }
    # Leading eigenvalues (lambda_0 is trivial ~1).
    for i, val in enumerate(eigenvalues[: min(5, eigenvalues.size)]):
        row[f"eigenvalue_{i}"] = float(val)
    if extra:
        row.update(extra)
    return row


def run_noise_study(config: ExperimentConfig) -> pd.DataFrame:
    """Sensitivity study over measurement-noise levels.

    For each noise level, regenerate observations, recompute the diffusion map
    (with automatic bandwidth) and PCA, and record all metrics.
    """

    sim_cfg = config.simulation_config()
    t, x = simulate_double_well(sim_cfg)

    rows: list[dict[str, float]] = []
    for noise in config.noise_levels:
        obs_cfg = config.observation_config(measurement_noise=noise)
        observations, _, _ = generate_observations(x, obs_cfg)

        dm_cfg = config.diffusion_config(epsilon="auto")
        dm_result = compute_diffusion_map(observations, dm_cfg)

        pca_scores, _ = compute_pca(observations, n_components=config.n_components)

        row = _metrics_row(
            x=x,
            dm_coord=dm_result.coordinates[:, 0],
            pca_coord=pca_scores[:, 0],
            eigenvalues=dm_result.eigenvalues,
            epsilon=dm_result.epsilon,
            measurement_noise=noise,
        )
        rows.append(row)

    return pd.DataFrame(rows)


def run_epsilon_study(
    config: ExperimentConfig, default_epsilon: float, observations: np.ndarray, x: np.ndarray
) -> pd.DataFrame:
    """Sensitivity study over kernel-bandwidth multiples of the default epsilon."""

    pca_scores, _ = compute_pca(observations, n_components=config.n_components)
    pca_coord = pca_scores[:, 0]

    rows: list[dict[str, float]] = []
    for mult in config.epsilon_multipliers:
        epsilon = mult * default_epsilon
        dm_cfg = config.diffusion_config(epsilon=epsilon)
        dm_result = compute_diffusion_map(observations, dm_cfg)

        row = _metrics_row(
            x=x,
            dm_coord=dm_result.coordinates[:, 0],
            pca_coord=pca_coord,
            eigenvalues=dm_result.eigenvalues,
            epsilon=epsilon,
            measurement_noise=config.measurement_noise,
            extra={"epsilon_multiplier": float(mult)},
        )
        rows.append(row)

    return pd.DataFrame(rows)


def generate_all_figures(
    artifacts: PipelineArtifacts,
    noise_df: pd.DataFrame,
    epsilon_df: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    """Create and save all ten required figures. Returns the saved paths."""

    figures_dir.mkdir(parents=True, exist_ok=True)
    x = artifacts.x

    # Sign-align embeddings with X for interpretable scatter plots.
    dm1 = sign_align(artifacts.dm_result.coordinates[:, 0], x)
    dm2 = artifacts.dm_result.coordinates[:, 1]
    pca1 = sign_align(artifacts.pca_scores[:, 0], x)
    pca2 = artifacts.pca_scores[:, 1]

    paths = [
        viz.plot_state_vs_time(artifacts.t, x, figures_dir / "01_state_vs_time.png"),
        viz.plot_state_histogram(x, figures_dir / "02_state_histogram.png"),
        viz.plot_feature_traces(
            artifacts.t,
            artifacts.features,
            artifacts.feature_names,
            figures_dir / "03_feature_traces.png",
        ),
        viz.plot_eigenvalues(
            artifacts.dm_result.eigenvalues, figures_dir / "04_diffusion_eigenvalues.png"
        ),
        viz.plot_coordinate_vs_time(
            artifacts.t, dm1, figures_dir / "05_dm_coordinate_vs_time.png"
        ),
        viz.plot_coordinate_vs_state(
            x, dm1, figures_dir / "06_dm_coordinate_vs_state.png"
        ),
        viz.plot_embedding_scatter(
            dm1,
            dm2,
            x,
            figures_dir / "07_dm_embedding_scatter.png",
            title="Diffusion map: first two coordinates (coloured by $X$)",
            xlabel=r"$\psi_1$",
            ylabel=r"$\psi_2$",
        ),
        viz.plot_embedding_scatter(
            pca1,
            pca2,
            x,
            figures_dir / "08_pca_embedding_scatter.png",
            title="PCA: first two components (coloured by $X$)",
            xlabel="PC 1",
            ylabel="PC 2",
        ),
        viz.plot_performance_vs_noise(
            noise_df, figures_dir / "09_performance_vs_noise.png"
        ),
        viz.plot_performance_vs_epsilon(
            epsilon_df, figures_dir / "10_performance_vs_epsilon.png"
        ),
    ]
    return paths
