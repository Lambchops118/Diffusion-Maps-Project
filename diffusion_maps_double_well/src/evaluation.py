"""Evaluation metrics and the PCA baseline.

We compare the first nontrivial diffusion coordinate and the first principal
component against the *true* latent state ``X`` using:

* **Pearson correlation** -- linear association with ``X``;
* **Spearman rank correlation** -- monotone association with ``X`` (robust to
  the nonlinear warping a diffusion coordinate may apply);
* **well-classification score** -- balanced accuracy of a simple sign-threshold
  classifier separating ``X < 0`` from ``X > 0``.

Eigenvectors and principal components have an arbitrary global sign.  Every
metric here is made sign-invariant, either by taking absolute correlations or
by sign-aligning the embedding with ``X`` before thresholding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score


@dataclass
class ComparisonMetrics:
    """Metrics comparing a 1-D embedding coordinate against the latent state.

    Attributes
    ----------
    pearson, spearman:
        Association with the continuous latent state ``X``; these measure
        *geometric* recovery, i.e.\\ whether the coordinate parameterizes the
        observation manifold.
    well_score:
        Balanced accuracy of a 2-means split against true well membership.
    metastability:
        ``|Pearson(coordinate, sign(X))|``: association with well membership
        rather than with position.  A coordinate that merely parameterizes
        arclength scores lower here than one that behaves like a well
        indicator, so this separates *dynamical* from *geometric* recovery in
        a way ``well_score`` does not.
    two_cluster:
        Fraction of coordinate variance explained by a 2-means partition. This
        is a descriptive two-level score, not a formal test of bimodality: even
        a uniform continuum can score substantially above zero.
    n_distinct:
        Number of distinct values the coordinate takes, as a fraction of the
        sample size. Far below 1 indicates a disconnected or nearly reducible
        embedding whose correlations cannot be trusted.
    """

    pearson: float
    spearman: float
    well_score: float
    metastability: float = float("nan")
    two_cluster: float = float("nan")
    n_distinct: float = float("nan")

    def as_dict(self, prefix: str = "") -> dict[str, float]:
        return {
            f"{prefix}pearson": self.pearson,
            f"{prefix}spearman": self.spearman,
            f"{prefix}well_score": self.well_score,
            f"{prefix}metastability": self.metastability,
            f"{prefix}two_cluster": self.two_cluster,
            f"{prefix}n_distinct": self.n_distinct,
        }


def two_cluster_variance_score(v: np.ndarray) -> float:
    """Fraction of a 1-D signal's variance explained by a 2-means partition.

    Returns ``1 - (within-cluster variance) / (total variance)`` using the
    *sample-weighted* within-cluster variance, so the result is the standard
    variance decomposition and always lies in ``[0, 1]``.

    Values approaching 1 indicate two compact levels, but a high value alone is
    not proof of a bimodal density. For example, an ideal uniform continuum on
    an interval scores 0.75. The metric is useful only as a descriptive comparison
    with the same score on the latent state.
    """

    v = np.asarray(v, dtype=np.float64).reshape(-1, 1)
    total = float(v.var())
    if total < 1e-30:
        return 0.0
    labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(v)
    within = 0.0
    for c in (0, 1):
        member = v[labels == c]
        if member.size:
            within += member.size * float(member.var())
    within /= v.shape[0]
    return float(np.clip(1.0 - within / total, 0.0, 1.0))


def bimodality_score(v: np.ndarray) -> float:
    """Backward-compatible alias for :func:`two_cluster_variance_score`."""

    return two_cluster_variance_score(v)


def metastability_score(embedding: np.ndarray, x: np.ndarray) -> float:
    """``|Pearson(embedding, sign(x))|``: alignment with well membership.

    A coordinate that behaves like a smoothed indicator of which well the system
    occupies scores near 1. This is a supervised shape diagnostic, not by itself
    a measurement of a dynamical timescale.
    """

    embedding = np.asarray(embedding, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return abs(_safe_pearson(embedding, np.sign(x)))


def sign_align(embedding: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Flip the sign of ``embedding`` so it positively correlates with ``reference``.

    Resolves the arbitrary sign of eigenvectors / principal components.
    """

    embedding = np.asarray(embedding, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    # Use covariance sign; robust even when correlation is weak.
    cov = np.cov(embedding, reference)[0, 1]
    return -embedding if cov < 0 else embedding


def _safe_pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(pearsonr(a, b)[0])


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    rho = spearmanr(a, b).correlation
    return float(rho) if np.isfinite(rho) else 0.0


def well_classification_score(embedding: np.ndarray, x: np.ndarray) -> float:
    """Balanced accuracy of separating the two wells from a 1-D coordinate.

    The wells are generally not equally populated, so a fixed median threshold
    is inappropriate. Instead we split the 1-D coordinate into
    two clusters with 2-means (an unsupervised, offset- and sign-invariant
    thresholding at the midpoint of the two cluster centres) and measure the
    balanced accuracy of the resulting partition against the true well
    membership ``sign(x)``.  Cluster labels are matched to wells in whichever
    orientation scores higher, resolving the arbitrary cluster naming.
    """

    embedding = np.asarray(embedding, dtype=np.float64).reshape(-1, 1)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    true_labels = (x >= 0.0).astype(int)
    if true_labels.min() == true_labels.max():
        # All samples in one well; classification is degenerate.
        return float("nan")

    km = KMeans(n_clusters=2, n_init=10, random_state=0)
    clusters = km.fit_predict(embedding)

    # Resolve the arbitrary cluster->well assignment by taking the better score.
    acc_direct = balanced_accuracy_score(true_labels, clusters)
    acc_flipped = balanced_accuracy_score(true_labels, 1 - clusters)
    return float(max(acc_direct, acc_flipped))


def evaluate_embedding(embedding: np.ndarray, x: np.ndarray) -> ComparisonMetrics:
    """Compute Pearson, Spearman, and well-classification metrics for a 1-D coord.

    Correlations are reported as absolute values (sign-invariant); the
    well-classification score uses an explicit sign alignment.
    """

    embedding = np.asarray(embedding, dtype=np.float64).reshape(-1)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if embedding.shape != x.shape:
        raise ValueError(
            f"embedding and x must have the same length, got {embedding.shape} "
            f"and {x.shape}."
        )

    pearson = abs(_safe_pearson(embedding, x))
    spearman = abs(_safe_spearman(embedding, x))
    well = well_classification_score(embedding, x)
    meta = metastability_score(embedding, x)
    two_cluster = two_cluster_variance_score(embedding)
    distinct = float(np.unique(np.round(embedding, 10)).size) / float(embedding.size)
    return ComparisonMetrics(
        pearson=pearson,
        spearman=spearman,
        well_score=well,
        metastability=meta,
        two_cluster=two_cluster,
        n_distinct=distinct,
    )


def compute_pca(observations: np.ndarray, n_components: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Run PCA on the standardized observations.

    Parameters
    ----------
    observations:
        Standardized observation matrix, shape ``(n_samples, n_features)``.
    n_components:
        Number of principal components to keep.

    Returns
    -------
    scores:
        Principal-component scores, shape ``(n_samples, n_components)``.
    explained_variance_ratio:
        Fraction of variance explained by each retained component.
    """

    observations = np.asarray(observations, dtype=np.float64)
    n_components = min(n_components, observations.shape[1], observations.shape[0])
    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(observations)
    return scores, pca.explained_variance_ratio_
