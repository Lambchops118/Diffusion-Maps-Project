"""Publication-quality plotting helpers.

Every function saves a single, self-contained 300-DPI PNG figure with a title,
labelled axes, and a colour bar or legend where appropriate.  Matplotlib is
used with the non-interactive ``Agg`` backend so figures render headless.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless, file-only rendering
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DPI = 300
_WELL_CMAP = "coolwarm"


def _finalize(fig: plt.Figure, path: Path) -> Path:
    """Tidy layout, save at 300 DPI, and close the figure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_state_vs_time(
    t: np.ndarray, x: np.ndarray, path: Path
) -> Path:
    """Figure 1: latent state ``X`` versus time."""

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t, x, lw=0.7, color="#1f4e79")
    ax.axhline(1.0, ls="--", lw=0.8, color="grey")
    ax.axhline(-1.0, ls="--", lw=0.8, color="grey")
    ax.axhline(0.0, ls=":", lw=0.8, color="black")
    ax.set_xlabel("time $t$")
    ax.set_ylabel("state $X_t$")
    ax.set_title("Double-well SDE trajectory (latent state vs. time)")
    return _finalize(fig, path)


def plot_state_histogram(x: np.ndarray, path: Path) -> Path:
    """Figure 2: histogram / density of the latent state ``X``."""

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(x, bins=60, density=True, color="#5b9bd5", alpha=0.8, edgecolor="white")
    ax.axvline(-1.0, ls="--", lw=1.0, color="firebrick", label="wells $x=\\pm 1$")
    ax.axvline(1.0, ls="--", lw=1.0, color="firebrick")
    ax.axvline(0.0, ls=":", lw=1.0, color="black", label="barrier $x=0$")
    ax.set_xlabel("state $x$")
    ax.set_ylabel("empirical density")
    ax.set_title("Bimodal stationary density of the double-well state")
    ax.legend()
    return _finalize(fig, path)


def plot_feature_traces(
    t: np.ndarray,
    features: np.ndarray,
    feature_names: tuple[str, ...],
    path: Path,
    selection: tuple[int, ...] = (0, 1, 3, 5),
) -> Path:
    """Figure 3: a selection of observation features versus time."""

    fig, ax = plt.subplots(figsize=(9, 4))
    for idx in selection:
        if idx < features.shape[1]:
            ax.plot(t, features[:, idx], lw=0.7, label=feature_names[idx])
    ax.set_xlabel("time $t$")
    ax.set_ylabel("feature value")
    ax.set_title("Selected nonlinear observation features vs. time")
    ax.legend(ncol=2, fontsize=8)
    return _finalize(fig, path)


def plot_eigenvalues(eigenvalues: np.ndarray, path: Path) -> Path:
    """Figure 4: leading diffusion-map eigenvalues (spectrum)."""

    fig, ax = plt.subplots(figsize=(6, 4))
    k = np.arange(eigenvalues.size)
    ax.plot(k, eigenvalues, "o-", color="#2e7d32")
    ax.axhline(1.0, ls=":", color="grey", lw=0.8)
    ax.set_xlabel("index $k$")
    ax.set_ylabel(r"eigenvalue $\lambda_k$")
    ax.set_title("Diffusion-map spectrum (trivial $\\lambda_0 \\approx 1$)")
    ax.set_xticks(k)
    return _finalize(fig, path)


def plot_coordinate_vs_time(
    t: np.ndarray, coordinate: np.ndarray, path: Path
) -> Path:
    """Figure 5: first nontrivial diffusion coordinate versus time."""

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t, coordinate, lw=0.7, color="#8e44ad")
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"diffusion coordinate $\psi_1$")
    ax.set_title("First nontrivial diffusion coordinate vs. time")
    return _finalize(fig, path)


def plot_coordinate_vs_state(
    x: np.ndarray, coordinate: np.ndarray, path: Path
) -> Path:
    """Figure 6: first diffusion coordinate versus the true latent state."""

    fig, ax = plt.subplots(figsize=(5.5, 5))
    sc = ax.scatter(x, coordinate, c=x, cmap=_WELL_CMAP, s=6, alpha=0.7)
    ax.set_xlabel("true state $X$")
    ax.set_ylabel(r"diffusion coordinate $\psi_1$")
    ax.set_title(r"$\psi_1$ vs. true state $X$")
    fig.colorbar(sc, ax=ax, label="true state $X$")
    return _finalize(fig, path)


def plot_embedding_scatter(
    coord_x: np.ndarray,
    coord_y: np.ndarray,
    color_by: np.ndarray,
    path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str = "true state $X$",
) -> Path:
    """Figures 7 & 8: 2-D embedding scatter coloured by the latent state."""

    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(coord_x, coord_y, c=color_by, cmap=_WELL_CMAP, s=6, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, label=colorbar_label)
    return _finalize(fig, path)


def plot_performance_vs_noise(
    metrics: pd.DataFrame, path: Path
) -> Path:
    """Figure 9: performance versus measurement-noise level.

    Expects columns ``measurement_noise``, ``dm_pearson``, ``dm_spearman``,
    ``pca_pearson``, ``pca_spearman``, ``dm_well_score``, ``pca_well_score``.
    """

    df = metrics.sort_values("measurement_noise")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(df["measurement_noise"], df["dm_pearson"], "o-", label="DM Pearson")
    ax1.plot(df["measurement_noise"], df["dm_spearman"], "s--", label="DM Spearman")
    ax1.plot(df["measurement_noise"], df["pca_pearson"], "o-", label="PCA Pearson")
    ax1.plot(df["measurement_noise"], df["pca_spearman"], "s--", label="PCA Spearman")
    ax1.set_xlabel("measurement-noise level")
    ax1.set_ylabel("|correlation| with $X$")
    ax1.set_title("Correlation vs. measurement noise")
    ax1.set_ylim(0, 1.02)
    ax1.legend(fontsize=8)

    ax2.plot(df["measurement_noise"], df["dm_well_score"], "o-", label="DM well acc.")
    ax2.plot(df["measurement_noise"], df["pca_well_score"], "s--", label="PCA well acc.")
    ax2.axhline(0.5, ls=":", color="grey", lw=0.8, label="chance")
    ax2.set_xlabel("measurement-noise level")
    ax2.set_ylabel("balanced accuracy")
    ax2.set_title("Well classification vs. measurement noise")
    ax2.set_ylim(0.4, 1.02)
    ax2.legend(fontsize=8)

    fig.suptitle("Performance vs. measurement noise (diffusion maps vs. PCA)")
    return _finalize(fig, path)


def plot_performance_vs_epsilon(
    metrics: pd.DataFrame, path: Path
) -> Path:
    """Figure 10: performance versus epsilon (bandwidth) multiplier.

    Expects columns ``epsilon_multiplier``, ``dm_pearson``, ``dm_spearman``,
    ``dm_well_score``.
    """

    df = metrics.sort_values("epsilon_multiplier")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(df["epsilon_multiplier"], df["dm_pearson"], "o-", label="DM Pearson")
    ax.plot(df["epsilon_multiplier"], df["dm_spearman"], "s--", label="DM Spearman")
    ax.plot(df["epsilon_multiplier"], df["dm_well_score"], "^-", label="DM well acc.")
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"bandwidth multiplier ($\epsilon / \epsilon_\mathrm{default}$)")
    ax.set_ylabel("score")
    ax.set_title("Diffusion-map performance vs. kernel bandwidth")
    ax.set_ylim(0, 1.02)
    ax.legend()
    return _finalize(fig, path)
