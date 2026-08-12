# Final paper

LaTeX source for *Geometry or Dynamics? What Diffusion Maps Recover from a
Stochastic Double-Well System*.

## Files

| File | Purpose |
|---|---|
| `main.tex` | the paper |
| `references.bib` | 20 references (the project's original 5 + 15 added) |

## Prerequisites

Figures are pulled from `../outputs/figures/` via `\graphicspath`, so run the
experiment at least once before building:

```bash
python run_experiment.py
```

Every number and figure in the paper comes from that single command. The
degeneracy demonstration in Appendix B is a separate snippet, reproduced
verbatim in the appendix itself.

## Building

No TeX distribution is installed on this machine. Install one first
(MiKTeX or TeX Live on Windows), then from this directory:

```bash
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Or, with `latexmk`:

```bash
latexmk -pdf main.tex
```

Alternatively, upload `main.tex`, `references.bib`, and the fifteen referenced
PNGs from `../outputs/figures/` to Overleaf. If you do, either keep the
`outputs/figures` directory structure or drop the PNGs into a `figures/`
subfolder next to `main.tex` — `\graphicspath` already searches both.

## Packages used

`geometry`, `amsmath`, `amssymb`, `amsthm`, `graphicx`, `booktabs`, `caption`,
`subcaption`, `algorithm`, `algpseudocode`, `natbib`, `xcolor`, `hyperref`,
`url`. All are in the standard TeX Live / MiKTeX distributions.

## Note on the superseded run

An earlier configuration of this study (short horizon, polynomial observations
only, median bandwidth, `alpha = 1`) reached materially different conclusions.
Its outputs are kept under `../outputs/_superseded/` and the paper analyses why
it was misleading in §8.3 — that analysis is part of the paper's argument, so
those files should not be deleted.
