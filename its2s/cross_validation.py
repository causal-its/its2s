# Description: Time-series cross-validation for model evaluation.
# Usage: from its2s.cross_validation import time_series_cv
# Dependencies: pandas, numpy

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data_prep import prepare_splits
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
                   date_col=None, target_col=None, covariate_cols=None,
                   config_path=None, config_overrides=None):
    """Evaluate a model using expanding-window time-series cross-validation.

    Folds are constructed using only the pre-intervention data.
    Each fold trains on an expanding window and tests on the next
    `test_days` days.

    Parameters
    ----------
    df : pd.DataFrame
        Full time series dataset.
    intervention_date : str or pd.Timestamp
        Only pre-intervention data is used for CV.
    model_name : str
        Model to evaluate.
    n_folds : int
        Number of CV folds.
    test_days : int
        Days in each test fold.
    min_train_days : int
        Minimum training days for the first fold.
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

    # Use only pre-intervention data for CV
    pre_df = df[df[date_col] < intervention_date].copy()
    n_pre = len(pre_df)

    if n_pre < min_train_days + test_days:
        raise ValueError(
            f"Not enough pre-intervention data for CV. Need at least "
            f"{min_train_days + test_days} rows, have {n_pre}."
        )

    # Compute fold boundaries
    available_for_testing = n_pre - min_train_days
    step = max(1, available_for_testing // n_folds)
    fold_results = []

    model_params = get_model_config(config, model_name)

    for i in range(n_folds):
        train_end_idx = min_train_days + i * step
        test_end_idx = min(train_end_idx + test_days, n_pre)

        if train_end_idx >= n_pre or test_end_idx <= train_end_idx:
            break

        train_fold = pre_df.iloc[:train_end_idx].copy()
        test_fold = pre_df.iloc[train_end_idx:test_end_idx].copy()

        if len(test_fold) == 0:
            break

        model = _get_model(model_name, model_params)
        try:
            import warnings
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
