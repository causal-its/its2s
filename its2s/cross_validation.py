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


_CV_METHOD_ARGS = {
    "observations": ("test_obs", "min_train_obs", "skip_obs"),
    "percent":      ("test_pct", "min_train_pct", "skip_pct"),
}


def time_series_cv(df, intervention_date, model_name="arima",
                   n_folds=5, test_obs=None, min_train_obs=None,
                   skip_obs=None, cv_end_date=None,
                   split_method="observations",
                   test_pct=None, min_train_pct=None, skip_pct=None,
                   date_col=None, target_col=None, covariate_cols=None,
                   config_path=None, config_overrides=None):
    """Evaluate a model using expanding-window time-series cross-validation.

    All CV windows are sized in OBSERVATIONS (rows of the regular series),
    never calendar days: fold boundaries are positional slices, so on a weekly
    series ``test_obs=52`` spans one year. Calendar-day windows exist only in
    ``prepare_splits``. Folds are non-overlapping by construction. Consecutive
    validation windows are separated by ``skip_obs`` (matching the R reference
    implementation's ``skip`` parameter). The CV window can be capped at
    ``cv_end_date`` to prevent tuning or evaluation folds from touching the
    held-out test period defined by ``run_single_its``.

    Only the window arguments belonging to the chosen ``split_method`` may be
    passed; arguments for the other method raise ValueError rather than being
    silently ignored.

    Fold layout (train = expanding, test = fixed width, all in observations):

        |------ min_train_obs ------|-- test_obs --|-- skip_obs --|-- test_obs --|...
        fold 1: train [0, T0),        test [T0, T0+test_obs)
        fold 2: train [0, T0+test_obs+skip_obs), test [T0+test_obs+skip_obs, ...)
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
    split_method : {"observations", "percent"}
        "observations" (default): size the fold windows as explicit
        observation counts via `test_obs`/`min_train_obs`/`skip_obs`.
        "percent": size them as fractions of the CV data's observation count
        via `test_pct`/`min_train_pct`/`skip_pct`.
    test_obs : int
        Length of each validation window in observations. Defaults to 90.
        Only with ``split_method="observations"``.
    min_train_obs : int
        Minimum training window for the first fold, in observations.
        Defaults to 365. Only with ``split_method="observations"``.
    skip_obs : int
        Gap in observations between the end of one validation window and the
        start of the next. Set to 0 for adjacent non-overlapping folds. The R
        reference uses skip = "12 months" (365 observations for daily data).
        Defaults to 0. Only with ``split_method="observations"``.
    test_pct : float
        Validation window per fold as a fraction of the CV observations.
        Defaults to 0.10. Only with ``split_method="percent"``.
    min_train_pct : float
        Minimum training window as a fraction of the CV observations.
        Defaults to 0.50. Only with ``split_method="percent"``.
    skip_pct : float
        Gap between folds as a fraction of the CV observations. Defaults to
        0.0. Only with ``split_method="percent"``.
    cv_end_date : str or pd.Timestamp, optional
        Upper bound on the data used for CV. Must be <= intervention_date.
        To keep CV folds out of the held-out evaluation window used by
        run_single_its, subtract a calendar span covering that window's
        observations (its size times the series period; a resolved-units
        safe default is tracked in GH #40).
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

    if split_method == "days":
        raise ValueError(
            "CV windows are observation counts; use split_method="
            "'observations' with test_obs/min_train_obs/skip_obs. "
            "Calendar-day windows exist only in prepare_splits."
        )
    if split_method not in _CV_METHOD_ARGS:
        raise ValueError(
            f"split_method must be 'percent' or 'observations', "
            f"got {split_method!r}."
        )

    passed = {"test_obs": test_obs, "min_train_obs": min_train_obs,
              "skip_obs": skip_obs, "test_pct": test_pct,
              "min_train_pct": min_train_pct, "skip_pct": skip_pct}
    allowed = _CV_METHOD_ARGS[split_method]
    foreign = [name for name, value in passed.items()
               if value is not None and name not in allowed]
    if foreign:
        raise ValueError(
            f"Arguments {foreign} do not apply to split_method="
            f"{split_method!r}, which uses {list(allowed)}. Pass the "
            "arguments for the chosen split_method only."
        )

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

    if split_method == "percent":
        test_pct = 0.10 if test_pct is None else test_pct
        min_train_pct = 0.50 if min_train_pct is None else min_train_pct
        skip_pct = 0.0 if skip_pct is None else skip_pct
        budget = min_train_pct + n_folds * test_pct
        if budget > 1.0:
            raise ValueError(
                f"CV percent budget exceeded: min_train_pct ({min_train_pct}) + "
                f"n_folds ({n_folds}) * test_pct ({test_pct}) = {budget:.3f} > 1.0. "
                "Reduce n_folds, test_pct, or min_train_pct."
            )
        if not (0 < test_pct < 1):
            raise ValueError(f"test_pct must be in (0, 1), got {test_pct}.")
        if not (0 < min_train_pct < 1):
            raise ValueError(f"min_train_pct must be in (0, 1), got {min_train_pct}.")
        if skip_pct < 0 or skip_pct >= 1:
            raise ValueError(f"skip_pct must be in [0, 1), got {skip_pct}.")
        test_obs = max(1, int(round(test_pct * n_cv)))
        min_train_obs = max(1, int(round(min_train_pct * n_cv)))
        skip_obs = max(0, int(round(skip_pct * n_cv)))
    else:
        test_obs = 90 if test_obs is None else test_obs
        min_train_obs = 365 if min_train_obs is None else min_train_obs
        skip_obs = 0 if skip_obs is None else skip_obs

    if n_cv < min_train_obs + test_obs:
        raise ValueError(
            f"Not enough pre-intervention data for CV. Need at least "
            f"{min_train_obs + test_obs} rows, have {n_cv}."
        )

    model_params = get_model_config(config, model_name)
    fold_results = []

    # Non-overlapping fold layout: each fold's test window starts at
    # min_train_obs + i * (test_obs + skip_obs), guaranteeing a gap of
    # skip_obs between the end of fold i and the start of fold i+1.
    for i in range(n_folds):
        test_start_idx = min_train_obs + i * (test_obs + skip_obs)
        test_end_idx = test_start_idx + test_obs

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
