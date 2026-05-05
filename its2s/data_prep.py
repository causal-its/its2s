# Description: Train/test/holdout splitting for ITS analysis.
# Usage: from its2s.data_prep import prepare_splits
# Dependencies: pandas

from dataclasses import dataclass

import pandas as pd


@dataclass
class TimeSeriesSplits:
    """Container for ITS train/test/holdout splits."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    holdout_df: pd.DataFrame
    full_predict_df: pd.DataFrame
    intervention_date: pd.Timestamp


def _compute_split_days_from_pct(df, intervention_date, date_col,
                                  test_pct, holdout_pct):
    """Convert row-count percentages to (test_days, holdout_days)."""
    n_pre = (df[date_col] < intervention_date).sum()
    n_post = (df[date_col] >= intervention_date).sum()
    test_days = max(1, int(round(test_pct * n_pre)))
    holdout_days = max(1, int(round(holdout_pct * n_post)))
    return test_days, holdout_days


def prepare_splits(df, intervention_date, date_col="ds",
                   split_method="percent",
                   test_pct=0.20, holdout_pct=1.0,
                   test_days=365, holdout_days=365):
    """Split a time series DataFrame into train, test, and holdout periods.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with at least a date column and target column.
    intervention_date : str or pd.Timestamp
        Date of the intervention.
    date_col : str
        Name of the date column.
    split_method : {"percent", "days"}
        "percent" (default): derive test/holdout window lengths from row-count
        percentages of the pre-/post-intervention slices.
        "days": use explicit `test_days` and `holdout_days`.
    test_pct : float
        Fraction of the pre-intervention slice used as the test window. Used
        only when ``split_method="percent"``.
    holdout_pct : float
        Fraction of the post-intervention slice used as the holdout window.
        Used only when ``split_method="percent"``.
    test_days : int
        Number of days before intervention used as the test window. Used only
        when ``split_method="days"``.
    holdout_days : int
        Number of days after intervention used as the holdout window. Used
        only when ``split_method="days"``.

    Returns
    -------
    TimeSeriesSplits
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    intervention_date = pd.Timestamp(intervention_date)

    if split_method == "percent":
        test_days, holdout_days = _compute_split_days_from_pct(
            df, intervention_date, date_col, test_pct, holdout_pct,
        )
    elif split_method != "days":
        raise ValueError(
            f"split_method must be 'percent' or 'days', got {split_method!r}."
        )

    test_start = intervention_date - pd.Timedelta(days=test_days)
    holdout_end = intervention_date + pd.Timedelta(days=holdout_days)

    train_df = df[df[date_col] < test_start].copy()
    test_df = df[(df[date_col] >= test_start) & (df[date_col] < intervention_date)].copy()
    holdout_df = df[(df[date_col] >= intervention_date) & (df[date_col] <= holdout_end)].copy()
    full_predict_df = df[(df[date_col] >= test_start) & (df[date_col] <= holdout_end)].copy()

    return TimeSeriesSplits(
        train_df=train_df,
        test_df=test_df,
        holdout_df=holdout_df,
        full_predict_df=full_predict_df,
        intervention_date=intervention_date,
    )
