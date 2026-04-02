# Description: Shared utilities for Prophet-based models.
# Usage: from its2s.models.utils import make_time_features
# Dependencies: pandas

import pandas as pd


def make_time_features(df, date_col="ds"):
    """Generate numeric time features from a date column.

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: day_of_week, day_of_year, month, week_of_year.
    """
    dates = pd.to_datetime(df[date_col])
    out = pd.DataFrame(index=df.index)
    out["day_of_week"] = dates.dt.dayofweek
    out["day_of_year"] = dates.dt.dayofyear
    out["month"] = dates.dt.month
    out["week_of_year"] = dates.dt.isocalendar().week.astype(int).values
    return out
