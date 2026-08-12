# Recovering the Hidden State of a Stochastic Double-Well System Using Diffusion Maps

A self-contained, reproducible project for a graduate stochastic-differential-equations
course.  It asks whether **diffusion maps** can recover the hidden
one-dimensional state of a stochastic double-well process from **noisy,
nonlinear, high-dimensional observations**, and benchmarks them against
**PCA**.

---

## 1. Research question

> Given only a high-dimensional, nonlinear, noisy encoding of a hidden scalar
> state that hops between two potential wells, can an unsupervised manifold-learning
> method recover that scalar state — in particular the *slow* variable that
> tracks which well the system occupies — and does it do so better than a linear
> method (PCA)?

## 2. Hypothesis

> Because diffusion maps preserve local geometric connectivity, the leading
> nontrivial diffusion coordinate should recover the slow variable associated
> with motion between the two potential wells. It should separate the two
> metastable states even when the scalar state is represented through nonlinear,
> noisy, high-dimensional observations.

## 3. Mathematical model

The hidden state is the overdamped Langevin dynamics of a particle in the
symmetric quartic double-well potential

$$ V(x) = -\tfrac{1}{2}x^2 + \tfrac{1}{4}x^4, \qquad -V'(x) = x - x^3, $$

giving the Itô stochastic differential equation

$$ dX_t = (X_t - X_t^3)\,dt + \sigma\,dW_t, $$

where $W_t$ is a standard Wiener process and $\sigma$ is the process-noise
(diffusion) coefficient.  The drift $x-x^3$ has **stable** fixed points at
$x=\pm 1$ (the two wells) and an **unstable** fixed point at $x=0$ (the barrier).
For a moderate $\sigma$ the trajectory is *metastable*: it dwells near one well
for a long time and occasionally makes a noise-induced transition over the
barrier to the other.  The stationary density is bimodal,
$\rho_s(x) \propto \exp(-2V(x)/\sigma^2)$.  The **slow variable** is exactly
"which well am I in", and it is what we hope to recover.

## 4. Euler–Maruyama method

We integrate the SDE with the explicit Euler–Maruyama scheme on a uniform grid
$t_n = n\,\Delta t$:

$$ X_{n+1} = X_n + (X_n - X_n^3)\,\Delta t + \sigma\sqrt{\Delta t}\,Z_n,
\qquad Z_n \stackrel{\text{iid}}{\sim} \mathcal N(0,1). $$

The $\sqrt{\Delta t}$ scaling of the noise is the defining feature of the scheme:
the Wiener increment over a step has variance $\Delta t$.  Configurable
parameters include the initial condition, fine time step, retained sample count,
subsampling stride, process noise, and RNG seed. The defaults retain every 20th
fine step, giving $N=3000$ observations over $T=599.8$ and 61 committed
transitions for the fixed seed.

> Implemented in [`src/simulation.py`](src/simulation.py).

## 5. High-dimensional observation model

The diffusion map never sees $X$ directly. Two nonlinear feature maps are used.
The polynomial map begins with

$$ \phi(x) = \big[\,x,\; x^2,\; x^3,\; \sin x,\; \cos x,\; \sin 2x,\; \cos 2x
\,\big] $$

(extended by default with $\tanh x$ and $e^{-x^2}$). The curved RBF sensor map is

$$ \phi_j(x)=\exp\!\left[-(x-c_j)^2/(2w^2)\right], $$

with 12 centers on $[-2.2,2.2]$ and $w=0.35$. The feature matrix
$F \in \mathbb R^{N\times m}$ is then lifted into a configurable
$d$-dimensional observation space ($10 \le d \le 20$) with a **fixed, seeded**
random Gaussian matrix $A \in \mathbb R^{m\times d}$ and corrupted by additive
Gaussian **measurement noise**:

$$ Y = F A + \eta\,E, \qquad E_{ij} \stackrel{\text{iid}}{\sim}\mathcal N(0,1). $$

Finally $Y$ is **column-standardized** (zero mean, unit variance) so that
Euclidean distances are not dominated by a few high-variance coordinates.  The
intrinsic geometry is one-dimensional (a curve parameterized by $x$) embedded
nonlinearly in $\mathbb R^d$; the two maps test a gently bent and a strongly
curved encoding.

> Implemented in [`src/observations.py`](src/observations.py).

## 6. Diffusion-map derivation and normalization

Diffusion maps (Coifman & Lafon, 2006) build a random walk on the data whose
slow eigenmodes parameterize the underlying manifold.  Implemented **from
scratch** in [`src/diffusion_maps.py`](src/diffusion_maps.py):

1. **Pairwise squared distances**
   $\;D^2_{ij} = \lVert y_i - y_j\rVert^2\;$ (dense $N\times N$).

2. **Gaussian affinity**
   $\;K_{ij} = \exp\!\big(-D^2_{ij}/\varepsilon\big).$
   $\varepsilon$ is a *squared* length scale (the bandwidth).

3. **Anisotropic (α) density normalization.**  With the density proxy
   $q_i = \sum_j K_{ij}$,

   $$ \tilde K_{ij} = \frac{K_{ij}}{q_i^{\alpha}\, q_j^{\alpha}}. $$

   - $\alpha=0$: classical normalized graph Laplacian — sampling density
     influences the operator.
   - $\alpha=\tfrac12$ (**default**): density-sensitive, Fokker–Planck-type
     normalization in the observation manifold's induced metric.
   - $\alpha=1$: Laplace–Beltrami normalization — the sampling
     density is divided out, so the operator approximates the Laplace–Beltrami
     operator of the underlying manifold geometry alone.

4. **Row-stochastic Markov matrix.**  With $d_i = \sum_j \tilde K_{ij}$,

   $$ P_{ij} = \frac{\tilde K_{ij}}{d_i}, \qquad \sum_j P_{ij}=1. $$

   $P$ is the transition matrix of a random walk on the data; its stationary
   distribution weights are $d_i$.

5. **Symmetric eigendecomposition (numerical method).**  $P$ is *not* symmetric,
   but it is **conjugate** to the symmetric matrix

   $$ P_{\text{sym}} = D^{-1/2}\,\tilde K\,D^{-1/2}, \qquad D=\operatorname{diag}(d_i),$$

   via $P = D^{-1/2} P_{\text{sym}} D^{1/2}$.  Hence $P$ and $P_{\text{sym}}$
   share the same (real) eigenvalues, and if $v_k$ is an eigenvector of
   $P_{\text{sym}}$ then

   $$ \psi_k = D^{-1/2} v_k $$

   is the corresponding **right** eigenvector of $P$.  We therefore
   eigendecompose the symmetric, positive-semidefinite $P_{\text{sym}}$ with
   `scipy.linalg.eigh` — numerically stable, guaranteed real eigenvalues,
   orthonormal eigenvectors — and transform back.  This is both faster and far
   more stable than a direct nonsymmetric eigensolve of $P$.

6. **Trivial mode removal.**  The largest eigenvalue is $\lambda_0=1$ with the
   constant right eigenvector $\psi_0 \propto \mathbf 1$ (a stationary random
   walk carries no information).  We discard it and keep the leading
   **nontrivial** eigenvectors.

7. **Diffusion coordinates.**  At diffusion time $t$ the embedding is
   $\Psi_t(y_i) = \big(\lambda_1^{t}\psi_1(i),\,\lambda_2^{t}\psi_2(i),\dots\big)$.
   The first coordinate $\psi_1$ is the slowest kernel mode. Whether it behaves
   like a well indicator or a geometric coordinate depends on $\alpha$ and on
   the observation geometry.

## 7. Bandwidth selection

The bandwidth $\varepsilon$ controls locality. Too small and the kernel becomes
numerically nearly reducible; too large and all points look equally close.

- **Scan** (`--epsilon scan`, default): maximize the log-kernel-sum slope over a
  grid constrained below by a conservative minimum-spanning-tree scale.
- **Legacy percentile** (`--epsilon auto`): use a percentile of nonzero squared
  distances. The median is retained for comparison but oversmooths the RBF map.
- **Manual**: pass a positive float, e.g. `--epsilon 12.5`.

The bandwidth study sweeps
$\{0.25,0.5,1,2,4,8,16,32\}\times\varepsilon_{\text{scan}}$.

## 8. PCA comparison

PCA is applied to the *same* standardized observations.  The first principal
component (PC1) is the best **linear** 1-D summary; $\psi_1$ is a **nonlinear**
one.  We compare PC1 and $\psi_1$ against the true $X$ with:

- **Pearson** correlation (linear association, reported as $|r|$ to absorb the
  arbitrary eigenvector sign);
- **Spearman** rank correlation (monotone association — the fair metric, since a
  diffusion coordinate may warp $X$ monotonically);
- **well-classification balanced accuracy** — the 1-D coordinate is split into
  two clusters by 2-means and matched to the true well membership $\operatorname{sign}(X)$;
- **well-indicator alignment**, $|r(\psi_1,\operatorname{sign}X)|$;
- **two-cluster variance score**, a descriptive 2-means score that is explicitly
  not treated as a formal test of bimodality;
- **2-D scatter plots** of the leading two coordinates coloured by $X$.

Sign ambiguity of eigenvectors / PCs is handled everywhere by either taking
absolute correlations or sign-aligning to $X$ before thresholding
(see [`src/evaluation.py`](src/evaluation.py)).

## 9. Installation

Requires **Python 3.11+** (the code also runs on 3.9+).

```bash
cd diffusion_maps_double_well
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Unix:     source .venv/bin/activate
pip install -r requirements.txt
```

No internet access or external datasets are needed — everything is simulated.

## 10. Command-line examples

Run the complete default experiment:

```bash
python run_experiment.py
```

Override parameters (command-line values win over `config.yaml`):

```bash
python run_experiment.py \
    --n-samples 5000 \
    --dt 0.01 \
    --process-noise 0.7 \
    --measurement-noise 0.1 \
    --output-dim 15 \
    --epsilon auto \
    --alpha 1.0 \
    --seed 42
```

Point at a different config file, use a manual bandwidth, or change the
normalization:

```bash
python run_experiment.py --config config.yaml --epsilon 12.0 --alpha 0.5
```

Run the test suite:

```bash
pytest
```

## 11. Generated files

All paths are **relative** to the project directory; missing folders are created
automatically.

```
outputs/
├── figures/    # seventeen 300-DPI PNG figures
├── data/       # latent state plus poly/RBF observations and embeddings
└── metrics/
    ├── alpha_study.csv
    ├── bandwidth_study.csv
    ├── measurement_noise_study.csv
    ├── barrier_study.csv
    ├── combined_metrics.csv
    └── run_parameters.json
```

Rows include the resolved bandwidth, near-unit-mode diagnostic, leading
eigenvalues and gap ratios, state-recovery correlations, well accuracy,
well-indicator alignment, and the descriptive two-cluster variance score.

## 12. Interpretation guide (figures)

| # | File | What to look for |
|---|------|------------------|
| 1 | `01_state_vs_time.png` | $X_t$ vs time. Long dwells near $\pm 1$ with abrupt hops across $0$ ⇒ metastability. |
| 2 | `02_state_histogram.png` | Empirical density of $X$. **Bimodal**, peaks near $\pm 1$, trough at $0$. |
| 3 | `03_{poly,rbf}_feature_traces.png` | Feature traces for both observation models. |
| 4–7 | `{04..07}_{poly,rbf}_*.png` | DM/PCA coordinates and 2-D embeddings for each model. |
| 8 | `08_alpha_study.png` | Effect of density normalization on state ordering and well-oriented shape. |
| 9 | `09_bandwidth_study.png` | Near-reducibility at small bandwidth and oversmoothing at large bandwidth. |
| 10 | `10_noise_study.png` | Measurement-noise robustness. |
| 11 | `11_barrier_study.png` | Sensitivity to the barrier-to-noise ratio. |
| 12 | `12_alpha_spectra.png` | Spectral gap ratios versus normalization. |

## 13. Computational limitations: the O(N²) wall

The dense distance and kernel matrices are $N\times N$, so both **memory** and
**time** scale as $O(N^2)$ (the eigensolve is up to $O(N^3)$).  A single
`float64` $N\times N$ matrix is $8N^2$ bytes — e.g. ~200 MB at $N=5000$,
~800 MB at $N=10^4$ — and several are allocated at once.  The code emits a
`UserWarning` above `DENSE_MATRIX_WARN_THRESHOLD = 4000` samples.  For larger
$N$ one would use sparse $k$-nearest-neighbour kernels, the Nyström extension,
or landmark/subsampling schemes; those are deliberately out of scope here to
keep the implementation readable.  The default $N=3000$ runs in seconds and fits
comfortably in memory.

## 14. Expected conclusions

- The stationary density is clearly **bimodal** and the trajectory shows several
  **well transitions** — the metastable structure is present in the data.
- Diffusion maps and PCA both order the gently bent polynomial map, while
  diffusion maps strongly outperform PC1 on the curved RBF map.
- $\alpha$ controls how much sampling-density information survives; its practical
  effect is strong on the polynomial arc and weak where RBF geometry already
  separates the wells.
- The $\alpha=1/2$ spectrum tracks barrier-depth trends but is not claimed to be
  the exact physical generator because the observation map changes the metric.
- Very small bandwidths can create nearly reducible kernels with misleadingly
  high Spearman correlation; very large bandwidths drive the method toward PCA.

## 15. Project layout

```
diffusion_maps_double_well/
├── README.md              # this file
├── requirements.txt
├── config.yaml            # default parameters (CLI overrides these)
├── run_experiment.py      # command-line driver
├── src/
│   ├── __init__.py
│   ├── simulation.py      # Euler–Maruyama double-well SDE
│   ├── observations.py    # nonlinear features + random lift + noise
│   ├── diffusion_maps.py  # from-scratch diffusion maps
│   ├── evaluation.py      # metrics + PCA baseline
│   ├── experiments.py     # pipeline + sensitivity studies
│   └── visualization.py   # 300-DPI figure helpers
├── tests/
│   ├── test_simulation.py
│   ├── test_observations.py
│   └── test_diffusion_maps.py
└── outputs/               # created on first run
    ├── figures/  data/  metrics/
```

## References

Full details (with annotations and a BibTeX file) are in
[`BIBLIOGRAPHY.md`](BIBLIOGRAPHY.md) and [`references.bib`](references.bib).

1. R. Banisch, Z. Trstanova, A. Bittracher, S. Klus, P. Koltai, *Diffusion maps
   tailored to arbitrary non-degenerate Itô processes*, arXiv:1710.03484
   [math.DS] (2017); ACHA 48(1), 242–265 (2020).
2. R. R. Coifman and S. Lafon, *Diffusion maps*, Applied and Computational
   Harmonic Analysis 21(1), 5–30 (2006).
3. M. Belkin and P. Niyogi, *Laplacian eigenmaps for dimensionality reduction
   and data representation*, Neural Computation 15(6), 1373–1396 (2003).
4. B. Nadler, S. Lafon, R. Coifman, I. Kevrekidis, *Diffusion maps, spectral
   clustering and reaction coordinates of dynamical systems*, ACHA 21(1),
   113–127 (2006).
5. D. J. Higham, *An algorithmic introduction to numerical simulation of
   stochastic differential equations*, SIAM Review 43(3), 525–546 (2001).
