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
parameters: initial condition $X_0$, time step $\Delta t$, number of samples,
process noise $\sigma$, and the RNG seed.  Defaults ($\Delta t=0.01$,
$\sigma=0.7$, $N=3000$) produce visits to **both** wells and several transitions.

> Implemented in [`src/simulation.py`](src/simulation.py).

## 5. High-dimensional observation model

The diffusion map never sees $X$ directly.  Each scalar $x$ is first mapped to a
nonlinear **feature vector**

$$ \phi(x) = \big[\,x,\; x^2,\; x^3,\; \sin x,\; \cos x,\; \sin 2x,\; \cos 2x
\,\big] $$

(optionally extended with the smooth extras $\tanh x$ and $e^{-x^2}$).  The
feature matrix $F \in \mathbb R^{N\times m}$ is then lifted into a configurable
$d$-dimensional observation space ($10 \le d \le 20$) with a **fixed, seeded**
random Gaussian matrix $A \in \mathbb R^{m\times d}$ and corrupted by additive
Gaussian **measurement noise**:

$$ Y = F A + \eta\,E, \qquad E_{ij} \stackrel{\text{iid}}{\sim}\mathcal N(0,1). $$

Finally $Y$ is **column-standardized** (zero mean, unit variance) so that
Euclidean distances are not dominated by a few high-variance coordinates.  The
intrinsic geometry is one-dimensional (a curve parameterized by $x$) embedded
nonlinearly in $\mathbb R^d$; the question is whether that curve survives the
lift and the noise.

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
   - $\alpha=\tfrac12$: Fokker–Planck normalization.
   - $\alpha=1$ (**default**): Laplace–Beltrami normalization — the sampling
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
   The first coordinate $\psi_1$ is the slowest mode — for a metastable system
   this is the well-indicator we are after.

## 7. Bandwidth selection

The bandwidth $\varepsilon$ controls locality.  Too small and the graph
disconnects (every point is its own island); too large and all points look
equally close (the walk mixes instantly and geometry is lost).

- **Automatic** (`--epsilon auto`, default): $\varepsilon$ is set to a
  **percentile of the nonzero pairwise squared distances** (default: the
  **median**, `--epsilon-percentile 50`).  The median keeps a healthy fraction
  of neighbours at $O(1)$ affinity and is a standard, robust, scale-free
  heuristic.
- **Manual**: pass a positive float, e.g. `--epsilon 12.5`.

The kernel-bandwidth sensitivity study sweeps
$\{0.5,\,1.0,\,2.0\}\times\varepsilon_{\text{default}}$.

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
├── figures/    # ten 300-DPI PNG figures (see §12)
├── data/
│   ├── time.npy, latent_state.npy         # true state X_t and time grid
│   ├── observations.npy, features.npy     # high-D observations + features
│   ├── latent_state.csv, observations.csv # human-readable copies
└── metrics/
    ├── noise_study.csv        # metrics vs. measurement noise
    ├── epsilon_study.csv      # metrics vs. bandwidth multiplier
    ├── combined_metrics.csv   # both studies, tagged by 'study'
    └── run_parameters.json    # fully-resolved run parameters (reproducibility)
```

Each metrics row stores: `epsilon`, `measurement_noise`, the leading
eigenvalues (`eigenvalue_0..4`), diffusion-map Pearson/Spearman/well-score, and
PCA Pearson/Spearman/well-score.

## 12. Interpretation guide (figures)

| # | File | What to look for |
|---|------|------------------|
| 1 | `01_state_vs_time.png` | $X_t$ vs time. Long dwells near $\pm 1$ with abrupt hops across $0$ ⇒ metastability. |
| 2 | `02_state_histogram.png` | Empirical density of $X$. **Bimodal**, peaks near $\pm 1$, trough at $0$. |
| 3 | `03_feature_traces.png` | Selected observation features vs time. Nonlinear, entangled — no single one is obviously "$X$". |
| 4 | `04_diffusion_eigenvalues.png` | Spectrum. $\lambda_0\approx 1$ (trivial); a **spectral gap** after $\lambda_1$ signals a dominant slow mode. |
| 5 | `05_dm_coordinate_vs_time.png` | $\psi_1$ vs time. Should mirror the well-hopping of figure 1. |
| 6 | `06_dm_coordinate_vs_state.png` | $\psi_1$ vs true $X$. A **monotone, sigmoidal** curve (plateaus at the wells, steep through the barrier) ⇒ faithful recovery. |
| 7 | `07_dm_embedding_scatter.png` | $(\psi_1,\psi_2)$ coloured by $X$. Colour varies **smoothly** along $\psi_1$; two clusters = two wells. |
| 8 | `08_pca_embedding_scatter.png` | $(\mathrm{PC1},\mathrm{PC2})$ coloured by $X$. Compare separation/ordering with figure 7. |
| 9 | `09_performance_vs_noise.png` | Correlations & well accuracy vs measurement noise. Graceful degradation; DM vs PCA gap. |
| 10 | `10_performance_vs_epsilon.png` | DM performance vs bandwidth (log$_2$ axis). A **plateau** near $\times 1$ shows robustness; degradation at large $\varepsilon$. |

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
- The leading nontrivial diffusion coordinate $\psi_1$ is a **monotone function
  of the hidden state $X$** (high $|Pearson|$, Spearman $\approx 1$) and cleanly
  **separates the two wells** (balanced accuracy well above chance), *despite*
  the nonlinear, noisy, high-dimensional lift — confirming the hypothesis.
- Because the feature map here retains a strong monotone imprint of $X$, **PCA
  is a competitive baseline**: on this problem $\psi_1$ and PC1 both recover $X$,
  with diffusion maps typically matching or slightly edging PCA on well
  separation and remaining robust as measurement noise grows. The diffusion map's
  advantage is expected to widen on problems whose intrinsic manifold is more
  strongly curved or where the linear imprint of the latent state is weaker.
- Diffusion-map quality is **robust to the bandwidth** across the $0.5$–$2\times$
  range, degrading only when $\varepsilon$ becomes large enough to wash out local
  geometry.

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
