"""From-scratch diffusion-map embedding.

This module implements diffusion maps directly (no specialized library) with
anisotropic density normalization and automatic bandwidth selection.

Algorithm
---------
Given observations ``y_1, ..., y_N`` in ``R^d``:

1. **Pairwise squared distances.**  Compute ``D2[i, j] = ||y_i - y_j||**2``.

2. **Gaussian affinity.**  Form the kernel

       K[i, j] = exp(-D2[i, j] / epsilon).

   ``epsilon`` sets the bandwidth (a squared length scale).

3. **Bandwidth selection.**  ``epsilon`` may be given manually, chosen by a
   legacy distance percentile, or chosen by a kernel-sum scan constrained by a
   conservative minimum-spanning-tree (MST) scale.

4. **Anisotropic (alpha) normalization.**  With the density estimate
   ``q[i] = sum_j K[i, j]`` form

       K_tilde[i, j] = K[i, j] / (q[i]**alpha * q[j]**alpha).

   ``alpha = 0`` recovers the classical graph Laplacian (density matters);
   ``alpha = 1`` (the Laplace--Beltrami normalization of Coifman & Lafon)
   removes the sampling density so the operator approximates the
   Laplace--Beltrami operator on the underlying manifold; ``alpha = 1/2``
   corresponds to a density-sensitive Fokker--Planck-type normalization in the
   observation manifold's induced metric.

5. **Row normalization (Markov matrix).**  With
   ``d[i] = sum_j K_tilde[i, j]`` the transition matrix is

       P[i, j] = K_tilde[i, j] / d[i],

   which is row-stochastic (each row sums to one).

6. **Symmetric eigendecomposition.**  ``P`` is not symmetric, but it is
   conjugate to the symmetric matrix

       P_sym = d^{-1/2} K_tilde d^{-1/2},   i.e.  P = d^{-1/2} P_sym d^{1/2}.

   We eigendecompose the *symmetric* ``P_sym`` (numerically stable, real
   eigenvalues via :func:`scipy.linalg.eigh`) and recover the right
   eigenvectors of ``P`` by ``psi_k = d^{-1/2} v_k``.  ``P`` and ``P_sym``
   share the same eigenvalues.

7. **Trivial eigenvector removal.**  The largest eigenvalue is ``1`` with the
   constant right eigenvector; we drop it and return the leading *nontrivial*
   eigenvectors, optionally scaled by their eigenvalues (the diffusion
   coordinates ``lambda_k**t * psi_k`` at diffusion time ``t``).

Memory warning
--------------
The dense distance/kernel matrices are ``O(N**2)`` in both memory and time.
For ``N = 5000`` samples a single ``float64`` ``N x N`` matrix is ~200 MB and
several such matrices are allocated.  A soft warning is emitted for large
``N``; see :data:`DENSE_MATRIX_WARN_THRESHOLD`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform

# Emit an O(N^2) memory warning above this many samples.
DENSE_MATRIX_WARN_THRESHOLD: int = 4000

# An eigenvalue within this tolerance of 1 is treated as numerically stationary.
# Exact components give repeated unit eigenvalues; weak links can instead give
# additional eigenvalues merely very close to 1.  We flag both cases without
# claiming literal disconnection.
TRIVIAL_EIGENVALUE_TOL: float = 1e-8


@dataclass(frozen=True)
class DiffusionMapConfig:
    """Configuration for the diffusion-map embedding.

    Parameters
    ----------
    epsilon:
        Kernel bandwidth (a squared length scale).  Either a positive float for
        a manual bandwidth, or the string ``"auto"`` for automatic selection.
    epsilon_percentile:
        Percentile (0--100) of nonzero pairwise squared distances used for
        automatic bandwidth selection.  ``50`` is the median.
    alpha:
        Anisotropic density-normalization exponent (0, 0.5, or 1 are the
        usual choices).
    n_components:
        Number of leading nontrivial diffusion coordinates to return.
    diffusion_time:
        Diffusion time ``t``.  Coordinates are scaled by ``lambda_k**t``.
        ``t = 0`` returns the bare eigenvectors.
    """

    epsilon: float | str = "scan"
    epsilon_percentile: float = 50.0
    alpha: float = 1.0
    n_components: int = 5
    diffusion_time: float = 1.0
    scan_grid: int = 32
    scan_safety: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.epsilon, str):
            if self.epsilon not in ("auto", "scan"):
                raise ValueError(
                    f"epsilon must be a positive float, 'auto' (percentile "
                    f"heuristic) or 'scan' (MST-scale-constrained scan), "
                    f"got {self.epsilon!r}."
                )
        elif self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}.")
        if not (0.0 <= self.epsilon_percentile <= 100.0):
            raise ValueError(
                f"epsilon_percentile must be in [0, 100], got {self.epsilon_percentile}."
            )
        if not (0.0 <= self.alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}.")
        if self.n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {self.n_components}.")
        if self.diffusion_time < 0.0:
            raise ValueError(
                f"diffusion_time must be non-negative, got {self.diffusion_time}."
            )


@dataclass
class DiffusionMapResult:
    """Container for the output of a diffusion-map computation.

    Attributes
    ----------
    coordinates:
        Leading nontrivial diffusion coordinates, shape
        ``(n_samples, n_components)``.
    eigenvalues:
        Eigenvalues in descending order, *including* the trivial leading
        eigenvalue (approximately 1) at index 0.
    eigenvectors:
        Right eigenvectors of the Markov matrix (columns), aligned with
        ``eigenvalues``, including the trivial constant eigenvector at column 0.
    epsilon:
        The bandwidth actually used.
    """

    coordinates: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    epsilon: float
    n_trivial: int = 1
    n_distinct: int = 0

    @property
    def is_degenerate(self) -> bool:
        """True if the kernel is disconnected or numerically nearly reducible.

        Exact components produce repeated eigenvalues at 1; very weak links
        produce eigenvalues numerically indistinguishable from 1.  In either
        case the leading coordinate can behave like a component indicator, and
        rank correlation may remain misleadingly high because it ignores
        magnitude. ``n_trivial`` is therefore a count of near-unit modes, not a
        literal component count.
        """

        return self.n_trivial > 1


def squared_distance_matrix(observations: np.ndarray) -> np.ndarray:
    """Compute the dense matrix of pairwise squared Euclidean distances.

    Parameters
    ----------
    observations:
        Array of shape ``(n_samples, n_features)``.

    Returns
    -------
    d2:
        Symmetric array of shape ``(n_samples, n_samples)`` with
        ``d2[i, j] = ||y_i - y_j||**2`` and zero diagonal.

    Warns
    -----
    UserWarning
        If ``n_samples`` exceeds :data:`DENSE_MATRIX_WARN_THRESHOLD`, because
        the dense matrices scale as ``O(n_samples**2)``.
    """

    observations = np.asarray(observations, dtype=np.float64)
    if observations.ndim != 2:
        raise ValueError(
            f"observations must be 2-D (n_samples, n_features), got shape "
            f"{observations.shape}."
        )
    n = observations.shape[0]
    if n > DENSE_MATRIX_WARN_THRESHOLD:
        approx_mb = (n * n * 8) / (1024**2)
        warnings.warn(
            f"Building a dense {n}x{n} distance matrix (~{approx_mb:.0f} MB per "
            f"float64 matrix, several are allocated). Diffusion maps are "
            f"O(N^2) in memory and time; consider subsampling for large N.",
            stacklevel=2,
        )

    # pdist -> squareform is memory efficient and avoids catastrophic
    # cancellation from the (a^2 - 2ab + b^2) expansion.
    condensed = pdist(observations, metric="sqeuclidean")
    d2 = squareform(condensed)
    np.fill_diagonal(d2, 0.0)
    return d2


def select_epsilon(d2: np.ndarray, percentile: float = 50.0) -> float:
    """Select the bandwidth from a percentile of nonzero squared distances.

    Parameters
    ----------
    d2:
        Pairwise squared-distance matrix.
    percentile:
        Percentile in ``[0, 100]``; ``50`` gives the median.

    Returns
    -------
    epsilon:
        A strictly positive bandwidth.
    """

    # Use only the strict upper triangle (each pair once, exclude the diagonal).
    iu = np.triu_indices_from(d2, k=1)
    offdiag = d2[iu]
    nonzero = offdiag[offdiag > 0.0]
    if nonzero.size == 0:
        raise ValueError(
            "All pairwise distances are zero; cannot select a bandwidth "
            "(are all observations identical?)."
        )
    epsilon = float(np.percentile(nonzero, percentile))
    if epsilon <= 0.0:
        # Fall back to the smallest positive distance if the percentile is 0.
        epsilon = float(nonzero.min())
    return epsilon


def connectivity_scale(d2: np.ndarray) -> float:
    """Return a conservative minimum-spanning-tree bandwidth scale.

    This is the largest edge of the minimum spanning tree of the squared-distance
    graph. Setting ``epsilon`` equal to this value makes every MST edge have
    affinity at least ``exp(-1)``. It is intentionally conservative and is not
    the mathematical connectivity threshold of a Gaussian graph, which is
    complete in exact arithmetic.

    Returns
    -------
    scale:
        Maximum MST edge weight, in the same squared-distance units as
        ``epsilon``.
    """

    d2 = np.asarray(d2, dtype=np.float64)
    mst = minimum_spanning_tree(d2)
    if mst.nnz == 0:
        raise ValueError("Cannot build a spanning tree; are all points identical?")
    return float(mst.max())


def connectivity_threshold(d2: np.ndarray, tol: float = 1e-12) -> float:
    """Operational bandwidth threshold for an affinity cutoff ``tol``.

    If an edge is present when ``exp(-d2 / epsilon) > tol``, the infimum
    bandwidth retaining every edge of a minimum spanning tree is the MST scale
    divided by ``-log(tol)``. Without an explicit cutoff, a Gaussian graph is
    connected for every positive bandwidth.
    """

    if not (0.0 < tol < 1.0):
        raise ValueError(f"tol must be in (0, 1), got {tol}.")
    return float(connectivity_scale(d2) / (-np.log(tol)))


def count_graph_components(d2: np.ndarray, epsilon: float, tol: float = 1e-12) -> int:
    """Number of connected components of the thresholded affinity graph."""

    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}.")
    if not (0.0 <= tol < 1.0):
        raise ValueError(f"tol must be in [0, 1), got {tol}.")
    adjacency = np.exp(-np.asarray(d2, dtype=np.float64) / epsilon) > tol
    return int(connected_components(adjacency, directed=False)[0])


def select_epsilon_scan(
    d2: np.ndarray,
    n_grid: int = 32,
    safety: float = 1.0,
    upper_percentile: float = 90.0,
) -> tuple[float, float]:
    """Select the bandwidth by an MST-scale-constrained kernel-sum scan.

    This replaces the median heuristic of :func:`select_epsilon`, which has no
    connection to the geometry of the data and was found to overshoot by one to
    two orders of magnitude on curved manifolds.

    The criterion is due to Coifman and Lafon.  Let

        S(epsilon) = sum_{i,j} exp(-d2[i, j] / epsilon).

    As ``epsilon -> 0`` the kernel becomes the identity and ``S -> N``; as
    ``epsilon -> infinity`` every entry tends to one and ``S -> N**2``.  Between
    these plateaus ``log S`` is approximately linear in ``log epsilon`` with
    slope ``m ~ d_intrinsic / 2``, and that linear region is exactly the range
    of bandwidths at which the kernel resolves the manifold rather than either
    isolated points or the whole data set.  We take the bandwidth at which the
    slope is largest.

    The scan is restricted from below by the conservative
    :func:`connectivity_scale`, without which the maximum-slope point can land in
    a nearly reducible regime. There the leading eigenvectors behave like
    component indicators while rank correlation can remain near-perfect because
    it ignores coordinate magnitude.

    Returns
    -------
    epsilon:
        Selected bandwidth.
    slope:
        The maximum log-log slope; ``2 * slope`` estimates the intrinsic
        dimension and is a useful sanity check (it should be near 1 for a
        curve).
    """

    d2 = np.asarray(d2, dtype=np.float64)
    lower = safety * connectivity_scale(d2)
    iu = np.triu_indices_from(d2, k=1)
    upper = float(np.percentile(d2[iu], upper_percentile))
    if not np.isfinite(lower) or lower <= 0.0:
        raise ValueError("MST scale is not positive; degenerate data.")
    if upper <= lower:
        # Degenerate spread: fall back to the conservative MST scale itself.
        return lower, float("nan")

    grid = np.logspace(np.log10(lower), np.log10(upper), n_grid)
    sums = np.array([np.exp(-d2 / e).sum() for e in grid])
    slopes = np.gradient(np.log(sums), np.log(grid))
    best = int(np.argmax(slopes))
    return float(grid[best]), float(slopes[best])


def gaussian_kernel(d2: np.ndarray, epsilon: float) -> np.ndarray:
    """Return the Gaussian affinity ``exp(-d2 / epsilon)``."""

    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}.")
    return np.exp(-d2 / epsilon)


def anisotropic_normalize(kernel: np.ndarray, alpha: float) -> np.ndarray:
    """Apply the anisotropic (alpha) density normalization.

    Computes ``K_tilde[i, j] = K[i, j] / (q[i]**alpha * q[j]**alpha)`` with
    ``q[i] = sum_j K[i, j]``.
    """

    if alpha == 0.0:
        return kernel.copy()
    q = kernel.sum(axis=1)
    q_alpha = np.power(q, alpha)
    # Outer product q_alpha[i] * q_alpha[j] in the denominator.
    return kernel / np.outer(q_alpha, q_alpha)


def markov_matrix(kernel_tilde: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalize to a Markov matrix and return the row sums.

    Returns
    -------
    p:
        Row-stochastic matrix ``P[i, j] = K_tilde[i, j] / d[i]``.
    d:
        The row sums ``d[i] = sum_j K_tilde[i, j]`` (the stationary weights).
    """

    d = kernel_tilde.sum(axis=1)
    if np.any(d <= 0.0):
        raise FloatingPointError(
            "Encountered a non-positive row sum while building the Markov "
            "matrix; the bandwidth may be far too small."
        )
    p = kernel_tilde / d[:, None]
    return p, d


def compute_diffusion_map(
    observations: np.ndarray, config: DiffusionMapConfig
) -> DiffusionMapResult:
    """Compute the diffusion-map embedding of a set of observations.

    See the module docstring for the full algorithm and normalization details.

    Parameters
    ----------
    observations:
        Array of shape ``(n_samples, n_features)``.  Should be standardized
        beforehand for meaningful Euclidean distances.
    config:
        A :class:`DiffusionMapConfig` instance.

    Returns
    -------
    result:
        A :class:`DiffusionMapResult` with the leading nontrivial diffusion
        coordinates, the eigenvalues/eigenvectors (trivial mode included), and
        the bandwidth used.
    """

    observations = np.asarray(observations, dtype=np.float64)
    n_samples = observations.shape[0]

    d2 = squared_distance_matrix(observations)

    if config.epsilon == "scan":
        epsilon, _ = select_epsilon_scan(
            d2, n_grid=config.scan_grid, safety=config.scan_safety
        )
    elif config.epsilon == "auto":
        epsilon = select_epsilon(d2, config.epsilon_percentile)
    else:
        epsilon = float(config.epsilon)

    kernel = gaussian_kernel(d2, epsilon)
    kernel_tilde = anisotropic_normalize(kernel, config.alpha)

    # Symmetric conjugate of the Markov matrix: P_sym = d^{-1/2} K_tilde d^{-1/2}.
    d = kernel_tilde.sum(axis=1)
    if np.any(d <= 0.0):
        raise FloatingPointError(
            "Non-positive degree encountered; the bandwidth epsilon is likely "
            "too small for the data. Try a larger epsilon."
        )
    d_inv_sqrt = 1.0 / np.sqrt(d)
    p_sym = kernel_tilde * np.outer(d_inv_sqrt, d_inv_sqrt)
    # Enforce exact symmetry against tiny floating-point asymmetries.
    p_sym = 0.5 * (p_sym + p_sym.T)

    # Number of eigenpairs: nontrivial components + the trivial one, capped.
    n_pairs = min(config.n_components + 1, n_samples)

    # eigh returns ascending eigenvalues; request the top n_pairs.
    lo = n_samples - n_pairs
    eigvals, eigvecs_sym = eigh(p_sym, subset_by_index=[lo, n_samples - 1])

    # Sort descending.
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs_sym = eigvecs_sym[:, order]

    # Recover right eigenvectors of P from those of the symmetric matrix:
    #   psi_k = d^{-1/2} v_k.
    eigvecs = eigvecs_sym * d_inv_sqrt[:, None]

    # Normalize the trivial eigenvector so its entries are constant/positive.
    # (psi_0 should be proportional to the all-ones vector.)
    if eigvecs[0, 0] < 0:
        # Flip so the leading (constant) eigenvector is positive; harmless.
        eigvecs[:, 0] *= -1.0

    # Drop the trivial leading mode (eigenvalue ~ 1, constant eigenvector).
    nontrivial_vecs = eigvecs[:, 1:]
    nontrivial_vals = eigvals[1:]

    # Diffusion coordinates: scale eigenvectors by lambda_k**t.
    if config.diffusion_time == 0.0:
        scaling = np.ones_like(nontrivial_vals)
    else:
        # Clip tiny negative eigenvalues (numerical) before exponentiation.
        safe_vals = np.clip(nontrivial_vals, 0.0, None)
        scaling = np.power(safe_vals, config.diffusion_time)

    coordinates = nontrivial_vecs * scaling[None, :]

    # Keep only the requested number of components.
    coordinates = coordinates[:, : config.n_components]

    if not np.all(np.isfinite(coordinates)):
        raise FloatingPointError(
            "Diffusion coordinates contain non-finite values; check epsilon "
            "and the observation scaling."
        )

    # Degeneracy diagnostics: number of eigenvalues numerically close to 1 and
    # distinct values of the leading coordinate. Exact disconnection produces
    # repeated unit eigenvalues; near-disconnection produces near-unit modes.
    # Both can make correlation metrics silently misleading.
    n_trivial = int(np.sum(eigvals > 1.0 - TRIVIAL_EIGENVALUE_TOL))
    n_distinct = int(np.unique(np.round(coordinates[:, 0], 10)).size)

    if n_trivial > 1:
        warnings.warn(
            f"The kernel has {n_trivial} eigenvalues numerically near 1 at "
            f"epsilon={epsilon:.4g}, indicating a disconnected or nearly "
            f"reducible graph. The leading nontrivial eigenvector may behave "
            f"like a component indicator rather than a manifold coordinate; "
            f"correlation metrics can therefore be misleading. Increase "
            f"epsilon or inspect the affinity spectrum and coordinate scale.",
            stacklevel=2,
        )

    return DiffusionMapResult(
        coordinates=coordinates,
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        epsilon=epsilon,
        n_trivial=n_trivial,
        n_distinct=n_distinct,
    )


def spectral_gap_ratios(eigenvalues: np.ndarray, n: int = 4) -> np.ndarray:
    """Return ``(1 - lambda_k) / (1 - lambda_1)`` for ``k = 1..n``.

    These ratios estimate the ratios of the continuum eigenvalues, since
    ``lambda_k ~ 1 - c * epsilon * mu_k`` for small ``epsilon``, and the
    unknown constant cancels.  They are the natural way to read a diffusion-map
    spectrum, because the raw eigenvalues depend strongly on the bandwidth
    while these ratios do not.

    Two readings are useful:

    * ``mu_k / mu_1 -> k**2`` is the signature of the Laplace--Beltrami
      operator on a one-dimensional interval, i.e.\\ of an embedding that has
      recovered *geometry* (the Neumann cosine modes).
    * ``mu_2 / mu_1 >> 1`` is the signature of a *timescale separation*, i.e.\\
      of a genuine metastable two-state structure with one slow mode.
    """

    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    if eigenvalues.size < 2:
        raise ValueError("Need at least two eigenvalues.")
    gaps = 1.0 - eigenvalues[1 : n + 1]
    if gaps.size == 0 or not np.isfinite(gaps[0]) or abs(gaps[0]) < 1e-15:
        return np.full(max(gaps.size, 1), np.nan)
    return gaps / gaps[0]
