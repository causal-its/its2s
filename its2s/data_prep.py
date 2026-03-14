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


def prepare_splits(df, intervention_date, date_col="ds", test_days=365, holdout_days=365):
    """Split a time series DataFrame into train, test, and holdout periods.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with at least a date column and target column.
    intervention_date : str or pd.Timestamp
        Date of the intervention.
    date_col : str
        Name of the date column.
    test_days : int
        Number of days before intervention used as the test (pre-intervention validation) window.
    holdout_days : int
        Number of days after intervention used as the holdout (post-intervention) window.

    Returns
    -------
    TimeSeriesSplits
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    intervention_date = pd.Timestamp(intervention_date)
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
