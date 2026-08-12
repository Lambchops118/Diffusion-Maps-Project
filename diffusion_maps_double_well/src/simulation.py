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
        Number of time samples to *return*, including the initial condition.
        With ``subsample > 1`` the integrator takes more steps than this; see
        below.
    dt:
        Integration time step.
    process_noise:
        Diffusion coefficient ``sigma`` multiplying the Wiener increment.
    x0:
        Initial condition ``X[0]``.
    seed:
        Seed for the random-number generator (reproducibility).
    subsample:
        Keep every ``subsample``-th integration step.  The SDE is always
        integrated at the fine step ``dt`` (needed for accuracy and stability),
        but only every ``subsample``-th point is retained, so the returned
        series spans ``n_samples * subsample * dt`` time units while holding
        ``n_samples`` points.

        This decouples two requirements that otherwise conflict.  Resolving the
        *dynamics* needs a horizon many times the Kramers escape time, i.e.\\
        many integration steps; the diffusion map is ``O(N^2)`` in the number of
        retained points and so caps ``N`` at a few thousand.  Subsampling in
        time buys horizon without buying points.  Because the retained points
        are still drawn from the same trajectory, they are still distributed
        according to the invariant density -- which is the only property the
        kernel construction actually uses.
    """

    n_samples: int = 3000
    dt: float = 0.01
    process_noise: float = 0.7
    x0: float = -1.0
    seed: int = 42
    subsample: int = 1

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
        if self.subsample < 1:
            raise ValueError(f"subsample must be >= 1, got {self.subsample}.")

    @property
    def n_steps(self) -> int:
        """Total number of integration steps actually taken."""

        return (self.n_samples - 1) * self.subsample + 1

    @property
    def horizon(self) -> float:
        """Total simulated time ``T`` spanned by the returned series."""

        return (self.n_steps - 1) * self.dt


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

    n_steps = config.n_steps
    dt = config.dt
    sigma = config.process_noise
    sqrt_dt = np.sqrt(dt)

    x_full = np.empty(n_steps, dtype=np.float64)
    x_full[0] = config.x0

    # Pre-draw all Gaussian increments for speed and reproducibility.
    z = rng.standard_normal(n_steps - 1)

    for k in range(n_steps - 1):
        xk = x_full[k]
        x_full[k + 1] = xk + drift(xk) * dt + sigma * sqrt_dt * z[k]

    if not np.all(np.isfinite(x_full)):
        raise FloatingPointError(
            "Simulation produced non-finite values. Try reducing dt or "
            "process_noise; the explicit Euler--Maruyama scheme can blow up "
            "if dt is too large relative to the cubic drift."
        )

    # Retain every subsample-th point (subsample == 1 keeps everything).
    x = x_full[:: config.subsample][: config.n_samples]
    t = np.arange(x.size, dtype=np.float64) * dt * config.subsample
    return t, x


def simulate_double_well_full(
    config: SimulationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate and return the *unsubsampled* trajectory.

    Useful for transition statistics, which should be measured on the finely
    resolved path rather than on the retained subset.
    """

    fine = SimulationConfig(
        n_samples=config.n_steps,
        dt=config.dt,
        process_noise=config.process_noise,
        x0=config.x0,
        seed=config.seed,
        subsample=1,
    )
    return simulate_double_well(fine)


def count_committed_transitions(x: np.ndarray, core: float = 0.5) -> int:
    """Count *committed* transitions between the two wells.

    A bare sign-change count badly overestimates the number of transitions when
    the barrier is low: the path recrosses ``x = 0`` many times in quick
    succession before committing to a well, and every recrossing is counted.
    Here a transition is registered only when the path enters the core of one
    well (``x > core``) having last been in the core of the other
    (``x < -core``).  Excursions that approach the barrier and fall back are
    correctly ignored.

    Parameters
    ----------
    x:
        Trajectory; should be the *finely resolved* path, not a subsampled one.
    core:
        Half-width of the barrier region excluded from the cores.  With wells
        at ``+-1`` the default ``0.5`` places the cores at ``|x| > 0.5``.
    """

    x = np.asarray(x, dtype=np.float64)
    state = np.zeros(x.size, dtype=np.int8)
    state[x > core] = 1
    state[x < -core] = -1

    last = 0
    n = 0
    for s in state:
        if s != 0:
            if last != 0 and s != last:
                n += 1
            last = s
    return int(n)


def kramers_escape_rate(sigma: float) -> float:
    """Kramers' asymptotic escape rate from one well of ``V(x)``.

    For the overdamped one-dimensional case with diffusion constant
    ``D = sigma**2 / 2``,

        k ~ sqrt(V''(well) * |V''(barrier)|) / (2 pi) * exp(-dV / D),

    with ``V''(+-1) = 2``, ``|V''(0)| = 1`` and barrier height ``dV = 1/4``.

    The formula is asymptotic in ``dV / D >> 1``; at the default
    ``sigma = 0.7`` this ratio is only about ``1.02``, so the returned rate
    should be read as an order-of-magnitude guide rather than a precise
    prediction.
    """

    if sigma <= 0.0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    diffusion = sigma**2 / 2.0
    barrier = 0.25
    prefactor = np.sqrt(2.0 * 1.0) / (2.0 * np.pi)
    return float(prefactor * np.exp(-barrier / diffusion))


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
