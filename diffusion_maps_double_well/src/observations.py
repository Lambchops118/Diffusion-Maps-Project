"""High-dimensional, nonlinear, noisy observation model.

The diffusion-map algorithm must *not* see the scalar latent state ``X``
directly.  Instead we build a nonlinear feature vector from each ``X`` value,
lift it into a configurable 10--20 dimensional observation space with a seeded
random linear map, add Gaussian measurement noise, and standardize the result.

This mimics a realistic situation in which we observe many indirect,
entangled measurements of an unknown low-dimensional state.  The question is
whether the intrinsic one-dimensional geometry survives this lifting and can be
recovered.

Feature map
-----------
For each scalar ``x`` the base features are

    [x, x**2, x**3, sin(x), cos(x), sin(2x), cos(2x)]

optionally extended with the smooth extras ``[tanh(x), exp(-x**2)]``.

Observation map
---------------
Let ``F`` be the ``(n_samples, n_features)`` feature matrix and ``A`` a fixed
random ``(n_features, output_dim)`` Gaussian matrix.  The observations are

    Y = F @ A + measurement_noise * E,

with ``E`` standard Gaussian.  ``Y`` is then column-standardized (zero mean,
unit variance) before being handed to PCA or diffusion maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BASE_FEATURE_NAMES: tuple[str, ...] = (
    "x",
    "x^2",
    "x^3",
    "sin(x)",
    "cos(x)",
    "sin(2x)",
    "cos(2x)",
)

EXTRA_FEATURE_NAMES: tuple[str, ...] = (
    "tanh(x)",
    "exp(-x^2)",
)


@dataclass(frozen=True)
class ObservationConfig:
    """Configuration for the observation model.

    Parameters
    ----------
    output_dim:
        Dimension of the observation space (recommended 10--20).
    measurement_noise:
        Standard deviation of the additive Gaussian measurement noise, applied
        *after* the random linear lift and *before* standardization.
    include_extra_features:
        If ``True``, append the smooth extras ``tanh(x)`` and ``exp(-x**2)``.
    standardize:
        If ``True``, column-standardize the final observations.
    seed:
        Seed for the random linear map and measurement noise.
    """

    output_dim: int = 15
    measurement_noise: float = 0.1
    include_extra_features: bool = True
    standardize: bool = True
    seed: int = 123
    feature_names: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if not (2 <= self.output_dim <= 100):
            raise ValueError(
                f"output_dim should be a small positive integer (10--20 "
                f"recommended), got {self.output_dim}."
            )
        if self.measurement_noise < 0.0:
            raise ValueError(
                f"measurement_noise must be non-negative, got {self.measurement_noise}."
            )


def build_features(
    x: np.ndarray, include_extra_features: bool = True
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build the nonlinear feature matrix from the scalar latent state.

    Parameters
    ----------
    x:
        Array of shape ``(n_samples,)`` with the latent state.
    include_extra_features:
        Whether to append the smooth extra features.

    Returns
    -------
    features:
        Array of shape ``(n_samples, n_features)``.
    names:
        Tuple of feature names, aligned with the columns of ``features``.
    """

    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(x)):
        raise ValueError("Latent state x contains non-finite values.")

    columns = [
        x,
        x**2,
        x**3,
        np.sin(x),
        np.cos(x),
        np.sin(2.0 * x),
        np.cos(2.0 * x),
    ]
    names = list(BASE_FEATURE_NAMES)

    if include_extra_features:
        columns.extend([np.tanh(x), np.exp(-(x**2))])
        names.extend(EXTRA_FEATURE_NAMES)

    features = np.column_stack(columns)
    return features, tuple(names)


def generate_observations(
    x: np.ndarray, config: ObservationConfig
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Lift the latent state into a noisy high-dimensional observation space.

    Parameters
    ----------
    x:
        Latent state, shape ``(n_samples,)``.
    config:
        An :class:`ObservationConfig` instance.

    Returns
    -------
    observations:
        Standardized observation matrix, shape ``(n_samples, output_dim)``.
    features:
        The intermediate feature matrix, shape ``(n_samples, n_features)``.
    feature_names:
        Names of the feature columns.
    """

    features, feature_names = build_features(x, config.include_extra_features)
    n_features = features.shape[1]

    rng = np.random.default_rng(config.seed)

    # Seeded random linear lift into the observation space.
    linear_map = rng.standard_normal((n_features, config.output_dim))
    observations = features @ linear_map

    # Additive Gaussian measurement noise.
    if config.measurement_noise > 0.0:
        noise = rng.standard_normal(observations.shape)
        observations = observations + config.measurement_noise * noise

    if config.standardize:
        observations = standardize(observations)

    if not np.all(np.isfinite(observations)):
        raise FloatingPointError("Observation matrix contains non-finite values.")

    return observations, features, feature_names


def standardize(matrix: np.ndarray) -> np.ndarray:
    """Column-standardize a matrix to zero mean and unit variance.

    Columns with (near) zero variance are left centered but not scaled, to
    avoid division by zero.
    """

    matrix = np.asarray(matrix, dtype=np.float64)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    safe_std = np.where(std < 1e-12, 1.0, std)
    return (matrix - mean) / safe_std
