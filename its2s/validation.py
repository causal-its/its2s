# Description: Input validation for the ITS pipeline.
# Usage: from its2s.validation import validate_inputs
# Dependencies: pandas, numpy

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_inputs(df, intervention_date, date_col, target_col,
                    covariate_cols, model_name,
                    split_method=None, test_pct=None, holdout_pct=None,
                    test_days=None, holdout_days=None):
    """Validate inputs before running the ITS pipeline.

    Raises ValueError with a clear message if any check fails.

    Parameters
    ----------
    df : pd.DataFrame
    intervention_date : str or pd.Timestamp
    date_col : str
    target_col : str
    covariate_cols : list[str] or None
    model_name : str
    """
    # M2-1: Check intervention_date type
    if not isinstance(intervention_date, (str, pd.Timestamp)):
        raise ValueError(
            f"intervention_date must be a str or pd.Timestamp, "
            f"got {type(intervention_date).__name__!r}. "
            "Example: '2022-03-15' or pd.Timestamp('2022-03-15')."
        )

    # Check required columns exist
    if date_col not in df.columns:
        raise ValueError(
            f"Date column '{date_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    # Check covariate columns exist
    if covariate_cols:
        missing = [c for c in covariate_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Covariate columns not found in DataFrame: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

        # M2-5: Check covariate columns for NaN values
        for col in covariate_cols:
            n_na = df[col].isna().sum()
            if n_na > 0:
                raise ValueError(
                    f"Covariate column '{col}' contains {n_na} missing value(s). "
                    "All covariate columns must be complete before model fitting. "
                    "Impute or drop missing rows before calling run_single_its()."
                )

    # Check DataFrame is not empty
    if len(df) == 0:
        raise ValueError("Input DataFrame is empty.")

    # Check target column has non-zero variance
    y = df[target_col].dropna()
    if len(y) > 0 and y.std() == 0:
        raise ValueError(
            f"Target column '{target_col}' has zero variance "
            f"(all values are {y.iloc[0]}). Models cannot fit constant data."
        )

    # Check intervention date is within the data range
    dates = pd.to_datetime(df[date_col])
    intervention_ts = pd.Timestamp(intervention_date)
    if intervention_ts < dates.min() or intervention_ts > dates.max():
        warnings.warn(
            f"Intervention date {intervention_ts.date()} is outside the data "
            f"range [{dates.min().date()}, {dates.max().date()}]. "
            "This may result in empty train or holdout splits.",
            UserWarning,
            stacklevel=3,
        )

    # Check sufficient data for model fitting
    n_before = (dates < intervention_ts).sum()
    if n_before < 10:
        warnings.warn(
            f"Only {n_before} observations before the intervention date. "
            "Most models require substantially more training data for "
            "reliable counterfactual estimation.",
            UserWarning,
            stacklevel=3,
        )

    # Split-method checks (issue 2.3): catch empty splits / out-of-range pcts
    if split_method == "percent":
        if test_pct is not None and not (0 < test_pct < 1):
            raise ValueError(
                f"periods.test_pct must be in (0, 1), got {test_pct}."
            )
        if holdout_pct is not None and not (0 < holdout_pct <= 1):
            raise ValueError(
                f"periods.holdout_pct must be in (0, 1], got {holdout_pct}."
            )
    elif split_method == "days":
        if test_days is not None and test_days >= n_before:
            raise ValueError(
                f"periods.test_days ({test_days}) >= number of pre-intervention "
                f"observations ({n_before}). The training split would be empty. "
                "Reduce test_days or switch to split_method='percent'."
            )

    # Check for excessive missing data
    n_missing = df[target_col].isna().sum()
    frac_missing = n_missing / len(df)
    if frac_missing > 0.2:
        warnings.warn(
            f"{frac_missing * 100:.1f}% of target column '{target_col}' values "
            f"are missing ({n_missing} / {len(df)} rows). "
            "This may degrade model performance.",
            UserWarning,
            stacklevel=3,
        )
