#!/usr/bin/env python3
# Description: Convert best tuning params from a performance_metrics CSV to its2s YAML
# Usage: python 03_build_best_params.py <performance_metrics_csv> [options]
# Dependencies: pandas, pyyaml
#
# Reads the performance_metrics_*.csv from an LA_Wildfire model run, converts the
# R/tidymodels hyperparameters to its2s (Python XGBoost + Prophet) format, and
# writes a best_params.yaml that 02_run_its.py can load with --best-params-yaml.
#
# Parameter mapping (R -> Python):
#   trees             -> n_estimators          (direct)
#   tree_depth        -> max_depth             (direct)
#   learn_rate        -> learning_rate         (direct)
#   min_n             -> min_child_weight      (direct)
#   sample_size       -> subsample             (direct)
#   mtry              -> colsample_bytree      (mtry / total_r_features; approximate)
#   loss_reduction    -> gamma                 (10 ^ loss_reduction; R stores log10)
#   stop_iter         -> NOT MAPPED            (needs eval_set in fit(), unsupported)
#   prior_scale_changepoints -> changepoint_prior_scale (direct, if column present)
#   changepoint_num   -> NOT MAPPED            (not exposed in its2s Prophet wrapper)
#   changepoint_range -> NOT MAPPED            (not exposed in its2s Prophet wrapper)

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import path_project  # noqa: E402

_DEFAULT_DATA_DIR = path_project / "replicate_LA_WF" / "LA_wildfire_files"

# ---- Constants ---------------------------------------------------------------

# Estimated total XGBoost features in the R model.
# R's mtry [10, 32] tuning range implies >= 32 total features:
#   11 covariates + ~21 recipe time/holiday features = ~32.
# The Python model has only 15 features (4 time + 11 covariates), so this
# conversion is approximate. Adjust with --r-features if needed.
DEFAULT_R_FEATURES = 32

# ---- Functions ---------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert LA_Wildfire best tuning params to its2s YAML format"
    )
    parser.add_argument(
        "csv_path",
        help="Path to performance_metrics_*.csv from an LA_Wildfire model run",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "best_params.yaml"),
        help="Output YAML path (default: best_params.yaml next to this script)",
    )
    parser.add_argument(
        "--r-features",
        type=int,
        default=DEFAULT_R_FEATURES,
        help=(
            f"Total XGB features in the R model, for mtry->colsample_bytree "
            f"conversion (default: {DEFAULT_R_FEATURES})"
        ),
    )
    return parser.parse_args()


def convert_row(row, total_r_features):
    """Convert one performance_metrics row to its2s model config dict.

    Returns a dict with keys 'prophet' and 'xgb', suitable for passing as
    config_overrides['models']['prophet_xgb'] in run_single_its().
    """
    xgb = {}

    _int_map = {
        "trees":      "n_estimators",
        "tree_depth": "max_depth",
        "min_n":      "min_child_weight",
    }
    for r_col, py_key in _int_map.items():
        if r_col in row and pd.notna(row[r_col]):
            xgb[py_key] = int(row[r_col])

    _float_map = {
        "learn_rate":  "learning_rate",
        "sample_size": "subsample",
    }
    for r_col, py_key in _float_map.items():
        if r_col in row and pd.notna(row[r_col]):
            xgb[py_key] = round(float(row[r_col]), 6)

    # mtry -> colsample_bytree (fraction; R uses integer count)
    if "mtry" in row and pd.notna(row["mtry"]):
        colsample = min(1.0, float(row["mtry"]) / total_r_features)
        xgb["colsample_bytree"] = round(colsample, 4)

    # loss_reduction -> gamma (R stores on log10 scale)
    if "loss_reduction" in row and pd.notna(row["loss_reduction"]):
        gamma = 10.0 ** float(row["loss_reduction"])
        xgb["gamma"] = round(gamma, 8)

    # stop_iter is intentionally skipped:
    # XGBRegressor.early_stopping_rounds requires eval_set passed to .fit(),
    # which its2s does not do. Including it in __init__ raises an XGBoost error.

    prophet = {
        "yearly_seasonality": True,
        "weekly_seasonality": True,
        "daily_seasonality": False,
    }
    if "prior_scale_changepoints" in row and pd.notna(row["prior_scale_changepoints"]):
        prophet["changepoint_prior_scale"] = round(
            float(row["prior_scale_changepoints"]), 6
        )
    else:
        prophet["changepoint_prior_scale"] = 0.05  # LA_Wildfire default

    return {"prophet": prophet, "xgb": xgb}


def print_conversion_notes(total_r_features):
    print("\n--- Conversion notes ---")
    print(f"  trees             -> n_estimators       (direct)")
    print(f"  tree_depth        -> max_depth           (direct)")
    print(f"  learn_rate        -> learning_rate       (direct)")
    print(f"  min_n             -> min_child_weight    (direct)")
    print(f"  sample_size       -> subsample           (direct)")
    print(f"  mtry              -> colsample_bytree    mtry / {total_r_features}")
    print(f"    NOTE: R model has ~{total_r_features} features; Python model has 15.")
    print(f"    Adjust with --r-features if colsample_bytree looks wrong.")
    print(f"  loss_reduction    -> gamma               10 ^ loss_reduction (log10)")
    print(f"  stop_iter         -> NOT MAPPED          (early_stopping_rounds needs")
    print(f"                                            eval_set; its2s unsupported)")
    print(f"  changepoint_num   -> NOT MAPPED          (not in its2s Prophet wrapper)")
    print(f"  changepoint_range -> NOT MAPPED          (not in its2s Prophet wrapper)")


# ---- Main --------------------------------------------------------------------


def main():
    args = parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        # Try resolving against the default data directory
        csv_path = _DEFAULT_DATA_DIR / csv_path.name
    if not csv_path.exists():
        print(f"ERROR: CSV not found. Tried:")
        print(f"  {Path(args.csv_path).resolve()}")
        print(f"  {_DEFAULT_DATA_DIR / Path(args.csv_path).name}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded: {csv_path.name}  ({len(df)} rows x {len(df.columns)} columns)")
    print(f"Columns: {list(df.columns)}")

    if "exposure_category" not in df.columns:
        print("\nWARNING: 'exposure_category' column not found; series IDs will omit exposure.")

    if len(df) == 0:
        print("ERROR: CSV is empty.")
        sys.exit(1)

    id_cols = [c for c in ["enc_type", "exposure_category", "cause"] if c in df.columns]
    print("\n--- Series found in CSV ---")
    print(df[id_cols].to_string(index=False))

    out_yaml = {}
    for _, row in df.iterrows():
        enc_type = str(row.get("enc_type", "unknown"))
        exposure = str(row.get("exposure_category", "unknown"))
        cause = str(row.get("cause", "unknown"))
        series_id = f"{enc_type}_{exposure}_{cause}"

        params = convert_row(row, args.r_features)
        out_yaml[series_id] = params

        print(f"\n--- {series_id} ---")
        print(f"  Prophet : {params['prophet']}")
        print(f"  XGB     : {params['xgb']}")

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        yaml.dump(out_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved: {out_path}")

    print_conversion_notes(args.r_features)

    print("\nNext step:")
    print(f"  python 02_run_its.py --best-params-yaml {out_path.name}")


if __name__ == "__main__":
    main()
