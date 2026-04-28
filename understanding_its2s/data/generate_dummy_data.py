#!/usr/bin/env python3
# Description: Generate the dummy daily time series used by the understanding_its2s
#              teaching notebooks. Deterministic (seed=42). Writes dummy_data.csv
#              next to this script.
# Usage: python generate_dummy_data.py
# Dependencies: numpy, pandas

from pathlib import Path

import numpy as np
import pandas as pd


# ---- Constants ---------------------------------------------------------------

START_DATE        = "2018-01-01"
INTERVENTION_DATE = "2022-03-15"
POST_DAYS         = 42
SEED              = 42

BASE_LEVEL        = 50.0
TREND_END         = 5.0
SEASONAL_AMPLITUDE = 10.0
SEASONAL_PERIOD   = 365
NOISE_SD          = 2.0
INTERVENTION_EFFECT = 8.0

OUTPUT_PATH = Path(__file__).resolve().parent / "dummy_data.csv"


# ---- Functions ---------------------------------------------------------------

def build_dummy_series() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    end_date = pd.Timestamp(INTERVENTION_DATE) + pd.Timedelta(days=POST_DAYS - 1)
    dates = pd.date_range(START_DATE, end_date, freq="D")
    n = len(dates)

    trend       = np.linspace(0.0, TREND_END, n)
    seasonality = SEASONAL_AMPLITUDE * np.sin(np.arange(n) * 2 * np.pi / SEASONAL_PERIOD)
    noise       = rng.normal(0.0, NOISE_SD, n)
    y = BASE_LEVEL + trend + seasonality + noise

    intervention_idx = np.searchsorted(dates, pd.Timestamp(INTERVENTION_DATE))
    y[intervention_idx:] += INTERVENTION_EFFECT

    df = pd.DataFrame({"ds": dates, "y": y})
    df["covar_linear"] = np.linspace(0.0, 1.0, n) + rng.normal(0.0, 0.02, n)
    df["covar_dow"]    = df["ds"].dt.dayofweek.astype(np.float64)
    df["covar_noise"]  = rng.normal(0.0, 1.0, n)
    return df


# ---- Main --------------------------------------------------------------------

def main() -> None:
    df = build_dummy_series()
    df.to_csv(OUTPUT_PATH, index=False, date_format="%Y-%m-%d")
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")
    print(f"Date range     : {df['ds'].min().date()} -> {df['ds'].max().date()}")
    print(f"Intervention   : {INTERVENTION_DATE}  (+{INTERVENTION_EFFECT}/day for {POST_DAYS} days)")
    print(f"Columns        : {list(df.columns)}")


if __name__ == "__main__":
    main()
