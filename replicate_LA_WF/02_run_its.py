#!/usr/bin/env python3
# Description: Run ITS analysis using the its2s Python package
# Usage: python 02_run_its.py [--model MODEL] [--enc-type ENC] [--exposure EXPOSURE] [--cause CAUSE]
#   [--n-sim N] [--n-jobs N] [--best-params-yaml PATH]
# Dependencies: its2s, pandas, pyarrow, pyyaml
#
# NOTE: run_batch() is NOT used here because runner.py contains an import bug
# (from ..config import load_config should be from ..settings import load_config).
# run_single_its() is called directly in a loop instead.

import argparse
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd
import yaml

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import path_project  # noqa: E402

from its2s import run_single_its  # noqa: E402
from its2s.batch.seed_manager import derive_seed  # noqa: E402

# ---- Constants ---------------------------------------------------------------

LA_FILES_DIR = path_project / "replicate_LA_WF" / "LA_wildfire_files"
PARQUET_PATH = LA_FILES_DIR / "df-predict-sf.parquet"
CONFIG_PATH = Path(__file__).parent / "config_replicate.yaml"

INTERVENTION_DATE = "2025-01-07"
SEED = 112358  # matches LA_Wildfire model_config.yaml seed

CAUSE_COLS = {
    "enc":        "rate_enc",
    "enc_resp":   "rate_enc_resp",
    "enc_cardio": "rate_enc_cardio",
    "enc_injury": "rate_enc_injury",
    "enc_neuro":  "rate_enc_neuro",
}

COVARIATE_COLS = [
    "pr", "tmmx", "tmmn", "rmin", "rmax", "vs", "srad",
    "influenza.a", "influenza.b", "rsv", "sars.cov2",
]

# ---- Functions ---------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ITS analysis using its2s"
    )
    parser.add_argument(
        "--model",
        default="prophet_xgb",
        choices=["arima", "prophet_xgb", "prophet_then_xgb", "neuralprophet"],
        help="Forecasting model (default: prophet_xgb)",
    )
    parser.add_argument(
        "--enc-type",
        default=None,
        help="Restrict to one encounter type (default: all in data)",
    )
    parser.add_argument(
        "--exposure",
        default=None,
        help="Restrict to one exposure_category (default: all in data)",
    )
    parser.add_argument(
        "--cause",
        default=None,
        choices=list(CAUSE_COLS.keys()),
        help="Restrict to one cause (default: all five)",
    )
    parser.add_argument(
        "--n-sim",
        type=int,
        default=None,
        help="Override bootstrap n_sim (default: read from config_replicate.yaml = 1000)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs inside each bootstrap run (default: 1)",
    )
    parser.add_argument(
        "--best-params-yaml",
        default=None,
        metavar="PATH",
        help=(
            "Path to best_params.yaml produced by 03_build_best_params.py. "
            "When provided, per-series model hyperparameters are injected as "
            "config_overrides, replacing the defaults in config_replicate.yaml. "
            "Series not found in the YAML fall back to config_replicate.yaml defaults."
        ),
    )
    return parser.parse_args()


def load_best_params(yaml_path):
    """Load best_params.yaml and return dict keyed by series_id."""
    path = Path(yaml_path)
    if not path.exists():
        print(f"ERROR: --best-params-yaml not found: {path}")
        sys.exit(1)
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    print(f"Loaded best params for {len(data)} series from: {path.name}")
    for sid in data:
        print(f"  {sid}")
    return data


def build_series_overrides(series_id, base_overrides, best_params, model_name):
    """Merge base CLI overrides with per-series model params from best_params.yaml."""
    overrides = dict(base_overrides)
    if best_params and series_id in best_params:
        model_params = best_params[series_id]
        overrides = {**overrides, "models": {model_name: model_params}}
    return overrides if overrides else None


def load_series(df, enc_types, exposures, causes):
    """Return list of (series_id, df_series) tuples, one per enc_type x exposure x cause."""
    series = []
    for enc in enc_types:
        for exposure in exposures:
            for cause, rate_col in causes.items():
                mask = (df["enc_type"] == enc) & (df["exposure_category"] == exposure)
                sub = (
                    df[mask]
                    [["date", rate_col] + COVARIATE_COLS]
                    .rename(columns={"date": "ds", rate_col: "y"})
                    .sort_values("ds")
                    .reset_index(drop=True)
                )
                sub = sub.dropna(subset=["y"])
                if len(sub) == 0:
                    continue
                series.append((f"{enc}_{exposure}_{cause}", sub))
    return series


def summarize_results(results):
    """Build a summary DataFrame from a list of (series_id, PipelineResult) tuples."""
    rows = []
    for series_id, res in results:
        if res is None:
            rows.append({"series_id": series_id, "status": "FAILED"})
            continue
        period_exc = res.excess_table.period_excess
        total_excess = (
            period_exc["total_excess"].sum() if len(period_exc) else float("nan")
        )
        excess_lo = (
            period_exc["excess_ci_lo"].sum() if "excess_ci_lo" in period_exc.columns
            else float("nan")
        )
        excess_hi = (
            period_exc["excess_ci_hi"].sum() if "excess_ci_hi" in period_exc.columns
            else float("nan")
        )
        rows.append({
            "series_id":      series_id,
            "status":         "OK",
            "model":          res.model_name,
            "train_rmse":     round(res.metrics_train.rmse, 3),
            "train_r2":       round(res.metrics_train.r2, 3),
            "test_rmse":      round(res.metrics_test.rmse, 3),
            "test_r2":        round(res.metrics_test.r2, 3),
            "test_mape":      round(res.metrics_test.mape, 2),
            "n_bootstrap_ok": res.bootstrap_result.n_successful,
            "total_excess":   round(total_excess, 2),
            "excess_ci_lo":   round(excess_lo, 2),
            "excess_ci_hi":   round(excess_hi, 2),
        })
    return pd.DataFrame(rows)


# ---- Main --------------------------------------------------------------------


def main():
    args = parse_args()

    causes = (
        {args.cause: CAUSE_COLS[args.cause]} if args.cause else CAUSE_COLS
    )

    # Base config overrides from CLI (bootstrap settings)
    base_overrides = {}
    if args.n_sim is not None:
        base_overrides["bootstrap"] = {"n_sim": args.n_sim}
    if args.n_jobs != 1:
        base_overrides.setdefault("bootstrap", {})["n_jobs"] = args.n_jobs

    # Per-series best params (optional)
    best_params = load_best_params(args.best_params_yaml) if args.best_params_yaml else {}

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(name)s -- %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- Load data and discover enc_types / exposures
    print(f"\nLoading: {PARQUET_PATH}")
    if not PARQUET_PATH.exists():
        print("ERROR: parquet not found. Place df-predict-sf.parquet in path_project/replicate_LA_WF/LA_wildfire_files/")
        sys.exit(1)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])

    enc_types = [args.enc_type] if args.enc_type else sorted(df["enc_type"].unique().tolist())
    exposures = [args.exposure] if args.exposure else sorted(df["exposure_category"].unique().tolist())

    print("=" * 60)
    print("ITS Analysis")
    print(f"  Model           : {args.model}")
    print(f"  Enc types       : {enc_types}")
    print(f"  Exposures       : {exposures}")
    print(f"  Causes          : {list(causes.keys())}")
    print(f"  Intervention    : {INTERVENTION_DATE}")
    print(f"  Config          : {CONFIG_PATH}")
    print(f"  Best params     : {args.best_params_yaml or 'defaults (config_replicate.yaml)'}")
    print(f"  Bootstrap n_sim : {args.n_sim if args.n_sim else 'from config (1000)'}")
    print("=" * 60)

    # ---- Build series list
    series_list = load_series(df, enc_types, exposures, causes)
    print(f"\nSeries to run: {len(series_list)}")
    for sid, sdf in series_list:
        has_best = sid in best_params
        print(f"  {sid}: {len(sdf)} rows, "
              f"{sdf['ds'].min().date()} to {sdf['ds'].max().date()}"
              f"{'  [best params loaded]' if has_best else ''}")

    # ---- Output directory
    out_dir = path_project / "replicate_LA_WF" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {out_dir}")

    # ---- Run each series
    results = []
    n_total = len(series_list)
    for i, (series_id, sdf) in enumerate(series_list, start=1):
        print(f"\n[{i}/{n_total}] Running: {series_id}")
        series_out = out_dir / series_id
        series_out.mkdir(parents=True, exist_ok=True)

        series_overrides = build_series_overrides(
            series_id, base_overrides, best_params, args.model
        )

        series_seed = derive_seed(SEED, series_id)
        try:
            res = run_single_its(
                df=sdf,
                intervention_date=INTERVENTION_DATE,
                target_col="y",
                date_col="ds",
                covariate_cols=COVARIATE_COLS,
                model_name=args.model,
                config_path=CONFIG_PATH,
                config_overrides=series_overrides,
                output_dir=series_out,
                seed=series_seed,
            )
            results.append((series_id, res))
            print(f"  Done: test_r2={res.metrics_test.r2:.3f}, "
                  f"test_rmse={res.metrics_test.rmse:.3f}, "
                  f"bootstrap_ok={res.bootstrap_result.n_successful}")
        except Exception as exc:
            print(f"  FAILED: {exc}")
            results.append((series_id, None))

    # ---- Summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    summary = summarize_results(results)
    print(summary.to_string(index=False))

    summary_path = out_dir / "replication_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary saved: {summary_path}")

    failed = summary[summary["status"] == "FAILED"]
    if len(failed):
        print(f"\nWARNING: {len(failed)} series failed: {list(failed['series_id'])}")

    print("\nDone.")


if __name__ == "__main__":
    main()
