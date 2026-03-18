#!/usr/bin/env python3
# Description: Load and validate the input parquet for its2s replication
# Usage: python 01_prep_data.py
# Dependencies: pandas, pyarrow

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import path_project  # noqa: E402

# ---- Constants ---------------------------------------------------------------

LA_FILES_DIR = path_project / "replicate_LA_WF" / "LA_wildfire_files"
PARQUET_PATH = LA_FILES_DIR / "df-predict-sf.parquet"

INTERVENTION_DATE = "2025-01-07"

# Cause label -> rate column in the parquet
CAUSE_COLS = {
    "enc":       "rate_enc",
    "enc_resp":  "rate_enc_resp",
    "enc_cardio":"rate_enc_cardio",
    "enc_injury":"rate_enc_injury",
    "enc_neuro": "rate_enc_neuro",
}

COVARIATE_COLS = [
    "pr", "tmmx", "tmmn", "rmin", "rmax", "vs", "srad",
    "influenza.a", "influenza.b", "rsv", "sars.cov2",
]

# ---- Main --------------------------------------------------------------------


def main():
    print("=" * 60)
    print("ITS Replication -- Data Validation")
    print("=" * 60)

    print(f"\nParquet path : {PARQUET_PATH}")
    if not PARQUET_PATH.exists():
        print("\nERROR: parquet file not found.")
        print("Place df-predict-sf.parquet in path_project/replicate_LA_WF/LA_wildfire_files/ and re-run.")
        sys.exit(1)

    df = pd.read_parquet(PARQUET_PATH)
    df["date"] = pd.to_datetime(df["date"])

    print(f"Shape        : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Date range   : {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Columns      : {list(df.columns)}")

    print("\n--- Counts by enc_type x exposure_category ---")
    print(df.groupby(["enc_type", "exposure_category"]).size().to_string())

    exposures = sorted(df["exposure_category"].unique())
    enc_types = sorted(df["enc_type"].unique())

    # ---- Date coverage per enc_type x exposure
    print("\n--- Date coverage by enc_type x exposure_category ---")
    for exposure in exposures:
        for enc in enc_types:
            sub = df[(df["enc_type"] == enc) & (df["exposure_category"] == exposure)]
            if len(sub) == 0:
                continue
            dates = sub["date"].sort_values()
            print(f"  {enc} / {exposure}: {dates.min().date()} to {dates.max().date()} "
                  f"({len(dates)} rows, {dates.nunique()} unique dates)")

    # ---- Expected data splits (mirrors prepare_splits logic)
    intervention_ts = pd.Timestamp(INTERVENTION_DATE)
    test_start = intervention_ts - pd.Timedelta(days=67)
    holdout_end = intervention_ts + pd.Timedelta(days=14)

    print("\n--- Expected data splits (matching config_replicate.yaml) ---")
    print(f"  Train   : dates < {test_start.date()}")
    print(f"  Test    : {test_start.date()} to {(intervention_ts - pd.Timedelta(days=1)).date()}  (67 days)")
    print(f"  Holdout : {INTERVENTION_DATE} to {holdout_end.date()}  (14 days post-intervention)")

    for exposure in exposures:
        for enc in enc_types:
            sub = df[(df["enc_type"] == enc) & (df["exposure_category"] == exposure)]
            if len(sub) == 0:
                continue
            n_train = (sub["date"] < test_start).sum()
            n_test = ((sub["date"] >= test_start) & (sub["date"] < intervention_ts)).sum()
            n_hold = ((sub["date"] >= intervention_ts) & (sub["date"] <= holdout_end)).sum()
            print(f"  {enc} / {exposure}: train={n_train}, test={n_test}, holdout={n_hold}")

    # ---- Missing values in key columns
    rate_cols = list(CAUSE_COLS.values())
    check_cols = ["date"] + rate_cols + COVARIATE_COLS
    missing_cols = [c for c in check_cols if c not in df.columns]
    if missing_cols:
        print(f"\nWARNING: columns not found in parquet: {missing_cols}")

    present_check = [c for c in check_cols if c in df.columns]
    missing_counts = df[present_check].isnull().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if len(missing_counts):
        print("\n--- Missing values ---")
        print(missing_counts.to_string())
    else:
        print("\nNo missing values in outcome or covariate columns.")

    # ---- Preview: first enc_type + first exposure + rate_enc_resp
    first_enc = enc_types[0]
    first_exp = exposures[0]
    print(f"\n--- Preview: {first_enc} / {first_exp} / rate_enc_resp (first 5 and last 5 rows) ---")
    preview_cols = ["date", "rate_enc_resp"] + [c for c in COVARIATE_COLS[:4] if c in df.columns]
    preview = (
        df[(df["enc_type"] == first_enc) & (df["exposure_category"] == first_exp)]
        [preview_cols]
        .sort_values("date")
    )
    print(preview.head(5).to_string(index=False))
    print("...")
    print(preview.tail(5).to_string(index=False))

    # ---- Descriptive stats
    if "rate_enc_resp" in df.columns:
        print("\n--- Descriptive stats: rate_enc_resp by enc_type x exposure_category ---")
        for exposure in exposures:
            for enc in enc_types:
                sub = df[(df["enc_type"] == enc) & (df["exposure_category"] == exposure)]["rate_enc_resp"]
                if len(sub) == 0:
                    continue
                print(f"  {enc} / {exposure}: n={len(sub)}, mean={sub.mean():.2f}, "
                      f"sd={sub.std():.2f}, min={sub.min():.2f}, max={sub.max():.2f}")

    # ---- Series inventory
    print("\n--- Series to be run in 02_run_its.py ---")
    n_enc = len(enc_types)
    n_exp = len(exposures)
    n_causes = len(CAUSE_COLS)
    print(f"  enc_types        : {enc_types}")
    print(f"  exposure_category: {exposures}")
    print(f"  causes           : {list(CAUSE_COLS.keys())}")
    print(f"  total            : {n_enc} x {n_exp} x {n_causes} = {n_enc * n_exp * n_causes} series")

    print("\nValidation complete. Ready for 02_run_its.py.")


if __name__ == "__main__":
    main()
