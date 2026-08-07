# Bibliography

References for the project *Recovering the Hidden State of a Stochastic
Double-Well System Using Diffusion Maps*.  The first entry is the paper provided
with the project; the remaining four are the core supporting references for the
diffusion-map methodology and the Euler–Maruyama simulation used here.

## Provided source

1. R. Banisch, Z. Trstanova, A. Bittracher, S. Klus, and P. Koltai,
   "Diffusion maps tailored to arbitrary non-degenerate Itô processes,"
   arXiv preprint **arXiv:1710.03484** [math.DS], 11 October 2017.
   *(Later published in Applied and Computational Harmonic Analysis, 48(1):242–265, 2020.)*
   — Generalizes classical diffusion maps to approximate the forward/backward
   generators of arbitrary non-degenerate Itô diffusions, removing the sampling-density
   bias. Directly motivates using diffusion maps to recover the slow reaction
   coordinate of an SDE such as the double well studied here.

## Additional sources

2. R. R. Coifman and S. Lafon,
   "Diffusion maps,"
   *Applied and Computational Harmonic Analysis*, 21(1):5–30, 2006.
   — The foundational paper introducing diffusion maps, the anisotropic
   ($\alpha$) family of kernel normalizations, and the diffusion-distance
   embedding implemented from scratch in this project.

3. M. Belkin and P. Niyogi,
   "Laplacian eigenmaps for dimensionality reduction and data representation,"
   *Neural Computation*, 15(6):1373–1396, 2003.
   — Establishes the graph-Laplacian / spectral-embedding framework on which
   diffusion maps build, and the eigenvector-based manifold-learning viewpoint
   underlying the method.

4. B. Nadler, S. Lafon, R. R. Coifman, and I. G. Kevrekidis,
   "Diffusion maps, spectral clustering and reaction coordinates of dynamical systems,"
   *Applied and Computational Harmonic Analysis*, 21(1):113–127, 2006.
   — Connects the leading diffusion coordinates to the slow reaction coordinates
   and metastable states of stochastic dynamical systems — the theoretical basis
   for expecting $\psi_1$ to recover the double-well "which-well" variable.

5. D. J. Higham,
   "An algorithmic introduction to numerical simulation of stochastic differential equations,"
   *SIAM Review*, 43(3):525–546, 2001.
   — An accessible, widely cited introduction to the Euler–Maruyama scheme used
   to simulate the double-well SDE $dX = (X - X^3)\,dt + \sigma\,dW$.
