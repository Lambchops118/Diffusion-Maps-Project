"""Recovering the Hidden State of a Stochastic Double-Well System Using Diffusion Maps.

This package provides a small, self-contained implementation of:

* Euler--Maruyama simulation of the overdamped double-well SDE
  ``dX = (X - X**3) dt + sigma dW``;
* a nonlinear, noisy, high-dimensional observation model;
* a from-scratch diffusion-map embedding with anisotropic (alpha) density
  normalization and automatic bandwidth selection;
* a PCA baseline and a set of evaluation metrics;
* experiment drivers and publication-quality plotting helpers.

The modules are intentionally decoupled so they can be read, tested, and
reused independently.
"""

from __future__ import annotations

__all__ = [
    "simulation",
    "observations",
    "diffusion_maps",
    "evaluation",
    "experiments",
    "visualization",
]

__version__ = "1.0.0"
