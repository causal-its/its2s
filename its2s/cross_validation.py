# Description: Time-series cross-validation for model evaluation.
# Usage: from its2s.cross_validation import time_series_cv
# Dependencies: pandas, numpy

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics.error_metrics import MetricsResult, compute_metrics
from .settings import get_model_config, load_config

logger = logging.getLogger(__name__)


@dataclass
class CVFoldResult:
    """Result from a single cross-validation fold."""

    fold: int
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    metrics: MetricsResult


@dataclass
class CVResult:
    """Aggregated cross-validation results."""

    model_name: str
    folds: list[CVFoldResult]
    mean_rmse: float
    mean_mae: float
    mean_mape: float
    mean_r2: float
    std_rmse: float
    std_mae: float

    def summary(self):
        """Return a human-readable summary string."""
        lines = [
            f"Cross-validation: {self.model_name} ({len(self.folds)} folds)",
            f"  RMSE: {self.mean_rmse:.4f} +/- {self.std_rmse:.4f}",
            f"  MAE:  {self.mean_mae:.4f} +/- {self.std_mae:.4f}",
            f"  MAPE: {self.mean_mape:.2f}%",
            f"  R2:   {self.mean_r2:.4f}",
        ]
        return "\n".join(lines)


def time_series_cv(df, intervention_date, model_name="arima",
                   n_folds=5, test_days=90, min_train_days=365,
                   skip_days=0, cv_end_date=None,
                   date_col=None, target_col=None, covariate_cols=None,
                   config_path=None, config_overrides=None):
    """Evaluate a model using expanding-window time-series cross-validation.

    Folds are non-overlapping by construction. Consecutive validation windows
    are separated by `skip_days` (matching the R reference implementation's
    `skip` parameter). The CV window can be capped at `cv_end_date` to prevent
    tuning or evaluation folds from touching the held-out test period defined
    by `run_single_its`.

    Fold layout (train = expanding, test = fixed width):

        |------ min_train_days ------|-- test_days --|-- skip_days --|-- test_days --|...
        fold 1: train [0, T0),         test [T0, T0+test_days)
        fold 2: train [0, T0+test_days+skip_days), test [T0+test_days+skip_days, ...)
        ...

    Parameters
    ----------
    df : pd.DataFrame
        Full time series dataset.
    intervention_date : str or pd.Timestamp
        CV uses only pre-intervention data (or data before cv_end_date if set).
    model_name : str
        Model to evaluate.
    n_folds : int
        Maximum number of CV folds to attempt.
    test_days : int
        Length of each validation window in days.
    min_train_days : int
        Minimum training window for the first fold.
    skip_days : int
        Gap in days between the end of one validation window and the start of
        the next. Set to 0 for adjacent non-overlapping folds. The R reference
        uses skip = "12 months" (365 days for daily data).
    cv_end_date : str or pd.Timestamp, optional
        Upper bound on the data used for CV. Must be <= intervention_date.
        Use intervention_date - test_days to keep CV folds out of the
        held-out evaluation window used by run_single_its.
        Defaults to intervention_date (all pre-intervention data).
    date_col : str, optional
        Date column name. Defaults to config value.
    target_col : str, optional
        Target column name. Defaults to config value.
    covariate_cols : list[str], optional
        Covariate column names.
    config_path : str or Path, optional
    config_overrides : dict, optional

    Returns
    -------
    CVResult
    """
    # Lazy import to avoid circular dependency
    from .pipeline import _get_model

    config = load_config(config_path, config_overrides)
    date_col = date_col or config["data"]["date_col"]
    target_col = target_col or config["data"]["target_col"]
    covariate_cols = (covariate_cols if covariate_cols is not None
                      else config["data"]["covariate_cols"])
    seasonality = config["metrics"]["seasonality"]

    intervention_date = pd.Timestamp(intervention_date)
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Determine upper bound for CV data
    if cv_end_date is not None:
        cv_end_date = pd.Timestamp(cv_end_date)
        if cv_end_date > intervention_date:
            raise ValueError(
                f"cv_end_date ({cv_end_date.date()}) must be <= "
                f"intervention_date ({intervention_date.date()})."
            )
        cv_df = df[df[date_col] < cv_end_date].copy()
    else:
        cv_df = df[df[date_col] < intervention_date].copy()

    n_cv = len(cv_df)

    if n_cv < min_train_days + test_days:
        raise ValueError(
            f"Not enough pre-intervention data for CV. Need at least "
            f"{min_train_days + test_days} rows, have {n_cv}."
        )

    model_params = get_model_config(config, model_name)
    fold_results = []

    # Non-overlapping fold layout: each fold's test window starts at
    # min_train_days + i * (test_days + skip_days), guaranteeing a gap of
    # skip_days between the end of fold i and the start of fold i+1.
    for i in range(n_folds):
        test_start_idx = min_train_days + i * (test_days + skip_days)
        test_end_idx = test_start_idx + test_days

        if test_end_idx > n_cv:
            break

        train_fold = cv_df.iloc[:test_start_idx].copy()
        test_fold = cv_df.iloc[test_start_idx:test_end_idx].copy()

        model = _get_model(model_name, model_params)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(train_fold, target_col=target_col,
                          date_col=date_col, covariate_cols=covariate_cols or None)
                pred = model.predict(test_fold, target_col=target_col,
                                     date_col=date_col,
                                     covariate_cols=covariate_cols or None)
        except Exception as e:
            logger.warning("CV fold %d failed: %s", i + 1, e)
            continue

        metrics = compute_metrics(
            test_fold[target_col].values,
            pred.predicted,
            training_actual=train_fold[target_col].values,
            seasonality=seasonality,
        )

        fold_results.append(CVFoldResult(
            fold=i + 1,
            train_end=train_fold[date_col].iloc[-1],
            test_start=test_fold[date_col].iloc[0],
            test_end=test_fold[date_col].iloc[-1],
            n_train=len(train_fold),
            n_test=len(test_fold),
            metrics=metrics,
        ))

        logger.info(
            "CV fold %d/%d: RMSE=%.4f, MAE=%.4f, R2=%.4f",
            i + 1, n_folds, metrics.rmse, metrics.mae, metrics.r2,
        )

    if not fold_results:
        raise RuntimeError("All CV folds failed. Check data and model.")

    rmses = [f.metrics.rmse for f in fold_results]
    maes = [f.metrics.mae for f in fold_results]
    mapes = [f.metrics.mape for f in fold_results]
    r2s = [f.metrics.r2 for f in fold_results]

    return CVResult(
        model_name=model_name,
        folds=fold_results,
        mean_rmse=float(np.mean(rmses)),
        mean_mae=float(np.mean(maes)),
        mean_mape=float(np.nanmean(mapes)),
        mean_r2=float(np.mean(r2s)),
        std_rmse=float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
        std_mae=float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
    )
