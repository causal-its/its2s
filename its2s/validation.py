# Description: Input validation for the ITS pipeline.
# Usage: from its2s.validation import validate_inputs
# Dependencies: pandas, numpy

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_inputs(df, intervention_date, date_col, target_col,
                    covariate_cols, model_name):
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
    intervention_date = pd.Timestamp(intervention_date)
    if intervention_date < dates.min() or intervention_date > dates.max():
        logger.warning(
            "Intervention date %s is outside the data range [%s, %s]. "
            "This may result in empty train or holdout splits.",
            intervention_date, dates.min(), dates.max(),
        )

    # Check sufficient data for model fitting
    n_before = (dates < intervention_date).sum()
    if n_before < 10:
        logger.warning(
            "Only %d observations before the intervention date. "
            "Most models require substantially more training data for "
            "reliable counterfactual estimation.",
            n_before,
        )

    # Check for excessive missing data
    n_missing = df[target_col].isna().sum()
    frac_missing = n_missing / len(df)
    if frac_missing > 0.2:
        logger.warning(
            "%.1f%% of target column '%s' values are missing (%d / %d rows). "
            "This may degrade model performance.",
            frac_missing * 100, target_col, n_missing, len(df),
        )
