"""Euler--Maruyama simulation of the stochastic double-well system.

The overdamped Langevin dynamics for a particle in the symmetric quartic
potential ``V(x) = -x**2 / 2 + x**4 / 4`` are

    dX_t = -V'(X_t) dt + sigma dW_t
         = (X_t - X_t**3) dt + sigma dW_t,

where ``W_t`` is a standard Wiener process and ``sigma`` is the process-noise
(diffusion) coefficient.  The deterministic drift ``X - X**3`` has stable fixed
points at ``x = -1`` and ``x = +1`` (the two wells) and an unstable fixed point
at ``x = 0`` (the barrier).  For a moderate ``sigma`` the trajectory spends long
stretches near one well and occasionally makes a noise-induced transition to the
other -- exactly the metastable, two-state behaviour we want the diffusion map
to recover.

We integrate the SDE with the Euler--Maruyama scheme

    X[n+1] = X[n] + (X[n] - X[n]**3) * dt + sigma * sqrt(dt) * Z[n],

with ``Z[n] ~ N(0, 1)`` i.i.d.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for the double-well SDE simulation.

    Parameters
    ----------
    n_samples:
        Number of time samples to return, including the initial condition.
    dt:
        Integration time step.
    process_noise:
        Diffusion coefficient ``sigma`` multiplying the Wiener increment.
    x0:
        Initial condition ``X[0]``.
    seed:
        Seed for the random-number generator (reproducibility).
    """

    n_samples: int = 5000
    dt: float = 0.01
    process_noise: float = 0.7
    x0: float = -1.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_samples < 2:
            raise ValueError(f"n_samples must be >= 2, got {self.n_samples}.")
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}.")
        if self.process_noise < 0.0:
            raise ValueError(
                f"process_noise (sigma) must be non-negative, got {self.process_noise}."
            )
        if not np.isfinite(self.x0):
            raise ValueError(f"x0 must be finite, got {self.x0}.")


def drift(x: np.ndarray | float) -> np.ndarray | float:
    """Return the deterministic drift ``x - x**3`` of the double-well SDE."""

    return x - x**3


def potential(x: np.ndarray | float) -> np.ndarray | float:
    """Return the double-well potential ``V(x) = -x**2 / 2 + x**4 / 4``."""

    return -0.5 * x**2 + 0.25 * x**4


def simulate_double_well(config: SimulationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Simulate the double-well SDE with Euler--Maruyama.

    Parameters
    ----------
    config:
        A :class:`SimulationConfig` instance holding all parameters.

    Returns
    -------
    t:
        Array of shape ``(n_samples,)`` with the sample times
        ``[0, dt, 2*dt, ...]``.
    x:
        Array of shape ``(n_samples,)`` with the simulated latent state.

    Notes
    -----
    A dedicated :class:`numpy.random.Generator` seeded with ``config.seed`` is
    used, so identical configurations produce bit-identical trajectories.
    """

    rng = np.random.default_rng(config.seed)

    n = config.n_samples
    dt = config.dt
    sigma = config.process_noise
    sqrt_dt = np.sqrt(dt)

    x = np.empty(n, dtype=np.float64)
    x[0] = config.x0

    # Pre-draw all Gaussian increments for speed and reproducibility.
    z = rng.standard_normal(n - 1)

    for k in range(n - 1):
        xk = x[k]
        x[k + 1] = xk + drift(xk) * dt + sigma * sqrt_dt * z[k]

    if not np.all(np.isfinite(x)):
        raise FloatingPointError(
            "Simulation produced non-finite values. Try reducing dt or "
            "process_noise; the explicit Euler--Maruyama scheme can blow up "
            "if dt is too large relative to the cubic drift."
        )

    t = np.arange(n, dtype=np.float64) * dt
    return t, x


def count_well_transitions(x: np.ndarray) -> int:
    """Count sign changes of the trajectory (crossings of the barrier at 0).

    This is a convenient diagnostic: a healthy default configuration should
    show several transitions between the ``x < 0`` and ``x > 0`` wells.
    """

    x = np.asarray(x)
    signs = np.sign(x)
    # Ignore exact zeros by forward-filling the previous nonzero sign.
    nonzero = signs != 0
    if not np.any(nonzero):
        return 0
    filled = signs.copy()
    last = 0.0
    for i in range(filled.size):
        if filled[i] == 0.0:
            filled[i] = last
        else:
            last = filled[i]
    return int(np.sum(np.abs(np.diff(filled)) > 0))
