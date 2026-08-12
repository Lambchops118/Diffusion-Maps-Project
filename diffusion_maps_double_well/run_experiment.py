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
* ``outputs/figures/`` -- seventeen 300-DPI PNG figures.

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
    run_alpha_study,
    run_bandwidth_study,
    run_barrier_study,
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
    parser.add_argument(
        "--subsample",
        type=int,
        default=None,
        help="Keep every k-th integration step (buys horizon without cost).",
    )
    parser.add_argument(
        "--feature-map",
        type=str,
        default=None,
        choices=["polynomial", "rbf"],
        help="Default observation model for single-model runs.",
    )
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
        help="Kernel bandwidth: a positive float, 'scan', or 'auto'.",
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
    """Interpret the epsilon argument: 'scan', 'auto', or a positive float."""

    if isinstance(value, str):
        if value.strip().lower() in ("auto", "scan"):
            return value.strip().lower()
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
        "subsample": args.subsample,
        "feature_map": args.feature_map,
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
    for key in (
        "feature_maps",
        "alpha_levels",
        "noise_levels",
        "bandwidth_multipliers",
        "barrier_sigmas",
    ):
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
    np.save(data_dir / "latent_state_full.npy", artifacts.x_full)
    np.save(data_dir / "time.npy", artifacts.t)

    pd.DataFrame({"time": artifacts.t, "latent_state": artifacts.x}).to_csv(
        data_dir / "latent_state.csv", index=False
    )

    for name, model in artifacts.models.items():
        tag = "poly" if name == "polynomial" else "rbf"
        np.save(data_dir / f"observations_{tag}.npy", model.observations)
        np.save(data_dir / f"features_{tag}.npy", model.features)
        np.save(
            data_dir / f"dm_coordinates_{tag}.npy", model.dm_result.coordinates
        )
        np.save(data_dir / f"pca_scores_{tag}.npy", model.pca_scores)
        obs_cols = [f"obs_{i}" for i in range(model.observations.shape[1])]
        pd.DataFrame(model.observations, columns=obs_cols).to_csv(
            data_dir / f"observations_{tag}.csv", index=False
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

    sim_cfg = config.simulation_config()
    print("=" * 74)
    print("Double-well diffusion-map experiment")
    print("=" * 74)
    print(f"  n_samples (kept)  = {config.n_samples}")
    print(f"  dt / subsample    = {config.dt} / {config.subsample}")
    print(f"  horizon T         = {sim_cfg.horizon:.1f}")
    print(f"  process_noise     = {config.process_noise}")
    print(f"  output_dim        = {config.output_dim}")
    print(f"  measurement_noise = {config.measurement_noise}")
    print(f"  epsilon           = {config.epsilon}")
    print(f"  alpha             = {config.alpha}")
    print(f"  feature maps      = {', '.join(config.feature_maps)}")
    print(f"  seed              = {config.seed}")
    print("-" * 74)

    # 1. Default pipeline ----------------------------------------------------
    print("[1/7] Default pipeline (simulate -> observe -> DM + PCA) ...")
    artifacts = run_pipeline(config)
    print(
        f"      T={artifacts.horizon:.0f}  Kramers escape time="
        f"{artifacts.kramers_time:.1f}  "
        f"T/tau={artifacts.horizon / artifacts.kramers_time:.1f}"
    )
    print(
        f"      {artifacts.n_transitions} committed transitions "
        f"({artifacts.n_sign_changes} raw sign changes), "
        f"right-well occupancy={artifacts.occupancy_right:.3f}, "
        f"2-cluster score(X)={artifacts.two_cluster_x:.3f}"
    )
    if artifacts.n_transitions < 10:
        print(
            "[warn] Fewer than 10 committed transitions: dynamical statistics "
            "will be poorly resolved. Raise --subsample or --n-samples."
        )
    for name, model in artifacts.models.items():
        print(
            f"      [{name}] eps_scan={model.epsilon_scan:.4g} "
            f"(MST scale={model.epsilon_mst_scale:.4g}, "
            f"median={model.epsilon_median:.4g}, "
            f"dim~{2 * model.scan_slope:.2f})  "
            f"PCA EVR1={model.pca_evr[0]:.3f}"
        )

    # 2-5. Studies -----------------------------------------------------------
    print("[2/7] Normalization (alpha) study ...")
    alpha_df = run_alpha_study(config, artifacts)
    print("[3/7] Kernel-bandwidth study ...")
    bandwidth_df = run_bandwidth_study(config, artifacts)
    print("[4/7] Measurement-noise study ...")
    noise_df = run_noise_study(config, artifacts)
    print("[5/7] Barrier-depth study ...")
    barrier_df = run_barrier_study(config)

    # 6. Persist -------------------------------------------------------------
    print("[6/7] Saving metrics, data arrays, and run parameters ...")
    tagged = []
    for label, frame in (
        ("alpha", alpha_df),
        ("bandwidth", bandwidth_df),
        ("measurement_noise", noise_df),
        ("barrier", barrier_df),
    ):
        frame.to_csv(metrics_dir / f"{label}_study.csv", index=False)
        copy = frame.copy()
        copy.insert(0, "study", label)
        tagged.append(copy)
    combined = pd.concat(tagged, ignore_index=True)
    combined.to_csv(metrics_dir / "combined_metrics.csv", index=False)
    print(f"      wrote combined metrics ({len(combined)} rows).")

    save_data_arrays(artifacts, data_dir)
    save_run_parameters(config, metrics_dir / "run_parameters.json")

    # 7. Figures -------------------------------------------------------------
    print("[7/7] Generating figures ...")
    figure_paths = generate_all_figures(
        artifacts, alpha_df, bandwidth_df, noise_df, barrier_df, figures_dir
    )
    for p in figure_paths:
        print(f"      figure: {p.relative_to(PROJECT_ROOT)}")

    # Headline summary -------------------------------------------------------
    print("-" * 74)
    print("Headline results")
    print("-" * 74)
    print(f"{'model':<12}{'method':<16}{'|rho(X)|':>10}{'|r(signX)|':>12}"
          f"{'2-clust':>9}{'well acc':>10}")
    for name in config.feature_maps:
        sub = alpha_df[alpha_df["feature_map"] == name]
        pca = sub.iloc[0]
        print(f"{name:<12}{'PCA':<16}{pca['pca_spearman']:>10.4f}"
              f"{pca['pca_metastability']:>12.4f}"
              f"{pca['pca_two_cluster']:>9.3f}{pca['pca_well_score']:>10.4f}")
        for alpha in (0.5, 1.0):
            r = sub[np.isclose(sub["alpha"], alpha)]
            if r.empty:
                continue
            r = r.iloc[0]
            print(f"{'':<12}{f'DM alpha={alpha}':<16}{r['dm_spearman']:>10.4f}"
                  f"{r['dm_metastability']:>12.4f}"
                  f"{r['dm_two_cluster']:>9.3f}{r['dm_well_score']:>10.4f}")
    print(f"\n  reference: 2-cluster variance score of X = "
          f"{artifacts.two_cluster_x:.3f}")
    print("=" * 74)
    print("Done. See the 'outputs/' directory for figures, data, and metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
