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
    x: np.ndarray,
    coordinate: np.ndarray,
    path: Path,
    ylabel: str = r"diffusion coordinate $\psi_1$",
    title: str = r"$\psi_1$ vs. true state $X$",
) -> Path:
    """A 1-D embedding coordinate versus the true latent state."""

    fig, ax = plt.subplots(figsize=(5.5, 5))
    sc = ax.scatter(x, coordinate, c=x, cmap=_WELL_CMAP, s=6, alpha=0.7)
    ax.set_xlabel("true state $X$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
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


_MODEL_STYLE = {
    "polynomial": ("#1f4e79", "o-"),
    "rbf": ("#c0392b", "s--"),
}


def plot_performance_vs_noise(metrics: pd.DataFrame, path: Path) -> Path:
    """Measurement-noise robustness, both observation models."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, grp in metrics.groupby("feature_map"):
        grp = grp.sort_values("measurement_noise")
        color, _ = _MODEL_STYLE.get(str(name), ("black", "o-"))
        ax1.plot(grp["measurement_noise"], grp["dm_spearman"], "o-",
                 color=color, label=f"DM ({name})")
        ax1.plot(grp["measurement_noise"], grp["pca_spearman"], "s--",
                 color=color, alpha=0.6, label=f"PCA ({name})")
        ax2.plot(grp["measurement_noise"], grp["dm_well_score"], "o-",
                 color=color, label=f"DM ({name})")
        ax2.plot(grp["measurement_noise"], grp["pca_well_score"], "s--",
                 color=color, alpha=0.6, label=f"PCA ({name})")

    ax1.set_xlabel("measurement-noise level $\\eta$")
    ax1.set_ylabel(r"$|\rho_{\mathrm{Spearman}}|$ with $X$")
    ax1.set_title("State recovery vs. measurement noise")
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8)

    ax2.axhline(0.5, ls=":", color="grey", lw=0.8, label="chance")
    ax2.set_xlabel("measurement-noise level $\\eta$")
    ax2.set_ylabel("balanced accuracy")
    ax2.set_title("Well classification vs. measurement noise")
    ax2.set_ylim(0.4, 1.05)
    ax2.legend(fontsize=8)

    fig.suptitle("Robustness to measurement noise")
    return _finalize(fig, path)


def plot_alpha_study(
    metrics: pd.DataFrame, two_cluster_x: float, path: Path
) -> Path:
    """The decisive figure: what alpha does to geometry and to dynamics."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax1, ax2, ax3 = axes

    for name, grp in metrics.groupby("feature_map"):
        grp = grp.sort_values("alpha")
        color, _ = _MODEL_STYLE.get(str(name), ("black", "o-"))
        ax1.plot(grp["alpha"], grp["dm_spearman"], "o-", color=color,
                 label=f"DM ({name})")
        ax1.plot(grp["alpha"], grp["pca_spearman"], ":", color=color,
                 alpha=0.6, label=f"PCA ({name})")
        ax2.plot(grp["alpha"], grp["dm_metastability"], "o-", color=color,
                 label=f"DM ({name})")
        ax3.plot(grp["alpha"], grp["dm_two_cluster"], "o-", color=color,
                 label=f"DM ({name})")

    ax1.set_ylabel(r"$|\rho_{\mathrm{Spearman}}|$ with $X$")
    ax1.set_title("Geometry: recovery of the state")
    ax1.set_ylim(0, 1.05)

    ax2.set_ylabel(r"$|r(\psi_1,\ \mathrm{sign}\,X)|$")
    ax2.set_title("Dynamics: alignment with well membership")
    ax2.set_ylim(0, 1.05)

    ax3.axhline(two_cluster_x, ls="--", color="black", lw=1.0,
                label=f"true state ({two_cluster_x:.2f})")
    ax3.set_ylabel("2-cluster variance score")
    ax3.set_title("Shape: compact two-level structure")
    ax3.set_ylim(0, 1.05)

    for ax in axes:
        ax.set_xlabel(r"normalization exponent $\alpha$")
        ax.axvline(0.5, ls=":", color="grey", lw=0.8)
        ax.legend(fontsize=7)

    fig.suptitle(
        r"Effect of the anisotropic normalization $\alpha$ "
        r"($\alpha=1/2$: density-sensitive; $\alpha=1$: density-free)"
    )
    return _finalize(fig, path)


def plot_alpha_spectra(metrics: pd.DataFrame, path: Path) -> Path:
    """Timescale gap mu_2 / mu_1 versus alpha: the spectral view of the same effect."""

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, grp in metrics.groupby("feature_map"):
        grp = grp.sort_values("alpha")
        color, _ = _MODEL_STYLE.get(str(name), ("black", "o-"))
        ax.plot(grp["alpha"], grp["gap_ratio_2"], "o-", color=color, label=str(name))
    ax.axhline(4.0, ls="--", color="grey", lw=1.0,
               label=r"$k^2$ law ($\mu_2/\mu_1=4$): 1-D interval")
    ax.set_xlabel(r"normalization exponent $\alpha$")
    ax.set_ylabel(r"$\mu_2/\mu_1 = (1-\lambda_2)/(1-\lambda_1)$")
    ax.set_title("Timescale separation vs. normalization")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    return _finalize(fig, path)


def plot_bandwidth_study(metrics: pd.DataFrame, path: Path) -> Path:
    """Performance versus bandwidth, marking numerical near-reducibility."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, grp in metrics.groupby("feature_map"):
        grp = grp.sort_values("bandwidth_multiplier")
        color, _ = _MODEL_STYLE.get(str(name), ("black", "o-"))
        ok = ~grp["degenerate"].astype(bool)
        bad = grp["degenerate"].astype(bool)

        ax1.plot(grp["bandwidth_multiplier"][ok], grp["dm_spearman"][ok], "o-",
                 color=color, label=f"DM ({name})")
        if bad.any():
            ax1.plot(grp["bandwidth_multiplier"][bad], grp["dm_spearman"][bad],
                     "x", color=color, ms=9, mew=2,
                     label=f"{name}: near-reducible kernel")
        ax1.axhline(float(grp["pca_spearman"].iloc[0]), ls=":", color=color,
                    alpha=0.7, lw=1.2)

        ax2.plot(grp["bandwidth_multiplier"][ok], grp["dm_well_score"][ok], "o-",
                 color=color, label=f"DM ({name})")
        if bad.any():
            ax2.plot(grp["bandwidth_multiplier"][bad], grp["dm_well_score"][bad],
                     "x", color=color, ms=9, mew=2)

    ax1.set_ylabel(r"$|\rho_{\mathrm{Spearman}}|$ with $X$")
    ax1.set_title("State recovery vs. bandwidth\n(dotted: PCA baseline)")
    ax2.axhline(0.5, ls=":", color="grey", lw=0.8)
    ax2.set_ylabel("well balanced accuracy")
    ax2.set_title("Well classification vs. bandwidth")
    for ax in (ax1, ax2):
        ax.set_xscale("log", base=2)
        ax.set_xlabel(r"$\epsilon / \epsilon_{\mathrm{scan}}$")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)

    fig.suptitle("Kernel bandwidth: near-reducibility below, over-smoothing above")
    return _finalize(fig, path)


def plot_barrier_study(metrics: pd.DataFrame, path: Path) -> Path:
    """Metastability recovery versus barrier depth, split by alpha."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, name in zip(axes, sorted(metrics["feature_map"].unique())):
        sub = metrics[metrics["feature_map"] == name]
        for alpha, grp in sub.groupby("alpha"):
            grp = grp.sort_values("barrier_ratio")
            ax.plot(grp["barrier_ratio"], grp["dm_metastability"], "o-",
                    label=rf"DM $\alpha={alpha}$")
        grp0 = sub.sort_values("barrier_ratio")
        ax.plot(grp0["barrier_ratio"], grp0["pca_metastability"], "s:",
                color="grey", label="PCA")
        ax.set_xlabel(r"barrier-to-noise ratio $\Delta V / D$")
        ax.set_title(f"{name} observations")
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$|r(\psi_1,\ \mathrm{sign}\,X)|$")
    axes[0].set_ylim(0, 1.05)
    fig.suptitle(
        "Deep barriers make geometry and dynamics agree; "
        r"shallow barriers separate them"
    )
    return _finalize(fig, path)
