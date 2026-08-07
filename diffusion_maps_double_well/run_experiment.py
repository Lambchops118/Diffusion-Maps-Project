"""Command-line driver for the double-well diffusion-map experiment.

Run the complete default experiment with::

    python run_experiment.py

Override any parameter on the command line (these win over the YAML config)::

    python run_experiment.py \
        --n-samples 5000 \
        --dt 0.01 \
        --process-noise 0.7 \
        --measurement-noise 0.1 \
        --output-dim 15 \
        --epsilon auto \
        --alpha 1.0 \
        --seed 42

A YAML configuration file (default ``config.yaml``) supplies the base values;
command-line arguments override it.

Outputs
-------
* ``outputs/data/``    -- latent state, observations, features (``.npy`` + CSV);
* ``outputs/metrics/`` -- combined metrics CSV, per-study CSVs, run params JSON;
* ``outputs/figures/`` -- the ten required 300-DPI PNG figures.

All paths are relative to the project directory; missing directories are
created automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Make ``src`` importable whether run as a script or a module.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import (  # noqa: E402
    ExperimentConfig,
    generate_all_figures,
    run_epsilon_study,
    run_noise_study,
    run_pipeline,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Recover the hidden state of a stochastic double-well system with "
            "diffusion maps, compared against PCA."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to a YAML configuration file (relative to project root).",
    )
    # Simulation
    parser.add_argument("--n-samples", type=int, default=None, help="Number of time samples.")
    parser.add_argument("--dt", type=float, default=None, help="Integration time step.")
    parser.add_argument(
        "--process-noise", type=float, default=None, help="SDE diffusion coefficient sigma."
    )
    parser.add_argument("--x0", type=float, default=None, help="Initial condition X[0].")
    parser.add_argument("--seed", type=int, default=None, help="Simulation random seed.")
    # Observations
    parser.add_argument(
        "--output-dim", type=int, default=None, help="Observation-space dimension (10-20)."
    )
    parser.add_argument(
        "--measurement-noise",
        type=float,
        default=None,
        help="Std. dev. of additive Gaussian measurement noise.",
    )
    parser.add_argument(
        "--observation-seed", type=int, default=None, help="Seed for the random lift/noise."
    )
    # Diffusion map
    parser.add_argument(
        "--epsilon",
        type=str,
        default=None,
        help="Kernel bandwidth: a positive float or 'auto'.",
    )
    parser.add_argument(
        "--epsilon-percentile",
        type=float,
        default=None,
        help="Percentile of nonzero sq. distances for auto epsilon (50 = median).",
    )
    parser.add_argument(
        "--alpha", type=float, default=None, help="Anisotropic normalization exponent (0-1)."
    )
    parser.add_argument(
        "--n-components", type=int, default=None, help="Number of diffusion coordinates."
    )
    parser.add_argument(
        "--diffusion-time", type=float, default=None, help="Diffusion time t."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory (relative)."
    )
    return parser.parse_args(argv)


def load_yaml_config(path: Path) -> dict:
    """Load a YAML config file if it exists, else return an empty dict."""

    if not path.exists():
        print(f"[info] No config file at '{path}', using built-in defaults.")
        return {}
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file '{path}' must contain a mapping at the top level.")
    return data


def _coerce_epsilon(value: str | float) -> float | str:
    """Interpret the epsilon argument: 'auto' or a positive float."""

    if isinstance(value, str):
        if value.strip().lower() == "auto":
            return "auto"
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(
                f"--epsilon must be 'auto' or a number, got {value!r}."
            ) from exc
    return value


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Merge built-in defaults, YAML config, and CLI overrides (CLI wins)."""

    config_path = (PROJECT_ROOT / args.config).resolve()
    yaml_cfg = load_yaml_config(config_path)

    # Start from dataclass defaults, layer YAML, then CLI overrides.
    merged: dict = asdict(ExperimentConfig())
    for key, value in yaml_cfg.items():
        if key in merged:
            merged[key] = value
        else:
            print(f"[warn] Ignoring unknown config key '{key}'.")

    cli_overrides = {
        "n_samples": args.n_samples,
        "dt": args.dt,
        "process_noise": args.process_noise,
        "x0": args.x0,
        "seed": args.seed,
        "output_dim": args.output_dim,
        "measurement_noise": args.measurement_noise,
        "observation_seed": args.observation_seed,
        "epsilon": _coerce_epsilon(args.epsilon) if args.epsilon is not None else None,
        "epsilon_percentile": args.epsilon_percentile,
        "alpha": args.alpha,
        "n_components": args.n_components,
        "diffusion_time": args.diffusion_time,
        "output_dir": args.output_dir,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            merged[key] = value

    # Coerce list-like fields to tuples.
    for key in ("noise_levels", "epsilon_multipliers"):
        if isinstance(merged.get(key), list):
            merged[key] = tuple(merged[key])

    # epsilon from YAML might be the string "auto" or a number.
    merged["epsilon"] = _coerce_epsilon(merged["epsilon"])

    return ExperimentConfig(**merged)


def save_run_parameters(config: ExperimentConfig, path: Path) -> None:
    """Save the fully-resolved run parameters to JSON for reproducibility."""

    path.parent.mkdir(parents=True, exist_ok=True)
    params = asdict(config)
    # Tuples are not JSON native; convert to lists.
    for key, value in params.items():
        if isinstance(value, tuple):
            params[key] = list(value)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(params, fh, indent=2)


def save_data_arrays(artifacts, data_dir: Path) -> None:
    """Persist the latent state, observations, and features to disk."""

    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(data_dir / "latent_state.npy", artifacts.x)
    np.save(data_dir / "time.npy", artifacts.t)
    np.save(data_dir / "observations.npy", artifacts.observations)
    np.save(data_dir / "features.npy", artifacts.features)

    # Human-readable CSVs too.
    pd.DataFrame({"time": artifacts.t, "latent_state": artifacts.x}).to_csv(
        data_dir / "latent_state.csv", index=False
    )
    obs_cols = [f"obs_{i}" for i in range(artifacts.observations.shape[1])]
    pd.DataFrame(artifacts.observations, columns=obs_cols).to_csv(
        data_dir / "observations.csv", index=False
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point: run the full experiment and write all outputs."""

    args = parse_args(argv)
    config = build_config(args)

    output_dir = (PROJECT_ROOT / config.output_dir).resolve()
    figures_dir = output_dir / "figures"
    data_dir = output_dir / "data"
    metrics_dir = output_dir / "metrics"
    for d in (figures_dir, data_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Double-well diffusion-map experiment")
    print("=" * 70)
    print(f"  n_samples         = {config.n_samples}")
    print(f"  dt                = {config.dt}")
    print(f"  process_noise     = {config.process_noise}")
    print(f"  output_dim        = {config.output_dim}")
    print(f"  measurement_noise = {config.measurement_noise}")
    print(f"  epsilon           = {config.epsilon}")
    print(f"  alpha             = {config.alpha}")
    print(f"  seed              = {config.seed}")
    print("-" * 70)

    # 1. Default pipeline ----------------------------------------------------
    print("[1/5] Running default pipeline (simulate -> observe -> DM + PCA) ...")
    artifacts = run_pipeline(config)
    print(
        f"      simulated {config.n_samples} steps, "
        f"{artifacts.n_transitions} well transitions, "
        f"default epsilon = {artifacts.default_epsilon:.4g}"
    )
    if artifacts.n_transitions < 2:
        print(
            "[warn] Fewer than 2 well transitions observed. Consider raising "
            "--process-noise or --n-samples so both wells are visited."
        )

    # 2. Sensitivity studies -------------------------------------------------
    print("[2/5] Measurement-noise sensitivity study ...")
    noise_df = run_noise_study(config)
    print("[3/5] Kernel-bandwidth sensitivity study ...")
    epsilon_df = run_epsilon_study(
        config, artifacts.default_epsilon, artifacts.observations, artifacts.x
    )

    # 3. Combined metrics ----------------------------------------------------
    noise_df_tagged = noise_df.copy()
    noise_df_tagged.insert(0, "study", "measurement_noise")
    epsilon_df_tagged = epsilon_df.copy()
    epsilon_df_tagged.insert(0, "study", "epsilon")
    combined = pd.concat([noise_df_tagged, epsilon_df_tagged], ignore_index=True)

    noise_df.to_csv(metrics_dir / "noise_study.csv", index=False)
    epsilon_df.to_csv(metrics_dir / "epsilon_study.csv", index=False)
    combined.to_csv(metrics_dir / "combined_metrics.csv", index=False)
    print(f"      wrote combined metrics ({len(combined)} rows).")

    # 4. Persist data + parameters ------------------------------------------
    print("[4/5] Saving data arrays and run parameters ...")
    save_data_arrays(artifacts, data_dir)
    save_run_parameters(config, metrics_dir / "run_parameters.json")

    # 5. Figures -------------------------------------------------------------
    print("[5/5] Generating figures ...")
    figure_paths = generate_all_figures(artifacts, noise_df, epsilon_df, figures_dir)
    for p in figure_paths:
        print(f"      figure: {p.relative_to(PROJECT_ROOT)}")

    # Quick summary of the headline default-setting result.
    default_noise_row = noise_df.loc[
        (noise_df["measurement_noise"] - config.measurement_noise).abs().idxmin()
    ]
    print("-" * 70)
    print("Headline result (at default measurement noise):")
    print(
        f"  Diffusion map : |Pearson|={default_noise_row['dm_pearson']:.3f}  "
        f"|Spearman|={default_noise_row['dm_spearman']:.3f}  "
        f"well acc.={default_noise_row['dm_well_score']:.3f}"
    )
    print(
        f"  PCA           : |Pearson|={default_noise_row['pca_pearson']:.3f}  "
        f"|Spearman|={default_noise_row['pca_spearman']:.3f}  "
        f"well acc.={default_noise_row['pca_well_score']:.3f}"
    )
    print("=" * 70)
    print("Done. See the 'outputs/' directory for figures, data, and metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
