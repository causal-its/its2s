# Description: Hyperparameter tuning via Latin hypercube grid search with time-series CV.
# Usage: from its2s.tuning import tune_model, TuningResult
# Dependencies: scipy, joblib, pandas, numpy

import logging
import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import qmc

from .cross_validation import time_series_cv

logger = logging.getLogger(__name__)

# Default trial counts per model, matching the R reference implementation:
# ARIMA=100, NNETAR (-> NeuralProphet)=75, prophet_xgb=100
_DEFAULT_N_TRIALS = {
    "arima":            100,
    "neuralprophet":    75,
    "prophet_xgb":      100,
    "prophet_then_xgb": 100,
}

# Search spaces: param_name -> (low, high, dtype, scale)
#   dtype : "int" or "float"
#   scale : "linear" or "log" (log produces a log-uniform distribution)
#
# Params prefixed "section__" are nested into {"section": {param: value}} in the
# model config dict before injection via config_overrides.
#
# NOTE on ARIMA: the R reference tunes fixed ARIMA orders (p,d,q,P,D,Q). The Python
# ARIMAModel uses pmdarima.auto_arima, which accepts upper bounds (max_p, max_d, etc.)
# and selects the best order within those bounds. Tuning these bounds controls the
# search space available to auto_arima rather than selecting a fixed order -- a
# methodological difference from the R implementation that should be noted in
# any downstream reporting.

_ARIMA_SPACE = {
    "max_p": (1, 5, "int", "linear"),
    "max_d": (0, 2, "int", "linear"),
    "max_q": (1, 5, "int", "linear"),
    "max_P": (0, 3, "int", "linear"),
    "max_D": (0, 1, "int", "linear"),
    "max_Q": (0, 3, "int", "linear"),
}

_NEURALPROPHET_SPACE = {
    "n_lags":          (7,     30,   "int",   "linear"),
    "epochs":          (50,    200,  "int",   "linear"),
    "learning_rate":   (0.001, 0.1,  "float", "log"),
    "batch_size":      (16,    128,  "int",   "linear"),
    "n_hidden_layers": (0,     2,    "int",   "linear"),
}

# Mirrors R phxgb: 3 Prophet params + 7 XGBoost params = 10 total (R had 11,
# including growth=linear and seasonality_yearly=TRUE which are fixed here).
_PROPHET_XGB_SPACE = {
    "prophet__changepoint_prior_scale": (0.01,  0.5,  "float", "log"),
    "prophet__seasonality_prior_scale": (0.1,   10.0, "float", "log"),
    "prophet__changepoint_range":       (0.4,   0.9,  "float", "linear"),
    "xgb__n_estimators":                (50,    500,  "int",   "linear"),
    "xgb__max_depth":                   (4,     20,   "int",   "linear"),
    "xgb__learning_rate":               (0.001, 0.3,  "float", "log"),
    "xgb__min_child_weight":            (1,     12,   "int",   "linear"),
    "xgb__subsample":                   (0.6,   1.0,  "float", "linear"),
    "xgb__colsample_bytree":            (0.5,   1.0,  "float", "linear"),
    "xgb__gamma":                       (0.001, 5.0,  "float", "log"),
}

_SEARCH_SPACES = {
    "arima":            _ARIMA_SPACE,
    "neuralprophet":    _NEURALPROPHET_SPACE,
    "prophet_xgb":      _PROPHET_XGB_SPACE,
    "prophet_then_xgb": _PROPHET_XGB_SPACE,
}


@dataclass
class TuningResult:
    """Result from a hyperparameter tuning run.

    Attributes
    ----------
    model_name : str
    best_params : dict
        Nested param dict ready to pass as config_overrides["models"][model_name].
    best_rmse : float
        Mean CV RMSE of the best parameter combination.
    best_std_rmse : float
        Std dev of CV RMSE across folds for the best combination.
    trials_df : pd.DataFrame
        One row per trial. Columns: trial_id, <param cols>, mean_rmse, std_rmse,
        mean_mae, mean_mape, mean_r2, n_folds_ok.
    n_trials : int
    n_folds : int
    metric : str
        Objective used for selection ("rmse" or "mae").
    seed : int
    """

    model_name: str
    best_params: dict
    best_rmse: float
    best_std_rmse: float
    trials_df: pd.DataFrame
    n_trials: int
    n_folds: int
    metric: str
    seed: int


def _sample_lhs(search_space: dict, n_trials: int, seed: int) -> list[dict]:
    """Return n_trials parameter dicts sampled via Latin hypercube.

    Each parameter is scaled from the unit hypercube to its declared range.
    Log-scale parameters are sampled log-uniformly (uniform in log space).
    Integer parameters are rounded and clamped to [low, high].
    """
    keys = list(search_space.keys())
    sampler = qmc.LatinHypercube(d=len(keys), seed=seed)
    unit_samples = sampler.random(n=n_trials)  # shape (n_trials, n_params)

    trials = []
    for row in unit_samples:
        params = {}
        for j, key in enumerate(keys):
            low, high, dtype, scale = search_space[key]
            u = float(row[j])
            if scale == "log":
                val = math.exp(math.log(low) + u * (math.log(high) - math.log(low)))
            else:
                val = low + u * (high - low)
            if dtype == "int":
                val = int(round(val))
                val = max(int(low), min(int(high), val))
            params[key] = val
        trials.append(params)

    return trials


def _unflatten_params(flat_params: dict) -> dict:
    """Convert double-underscore-prefixed keys into nested dicts.

    {"xgb__max_depth": 8, "prophet__changepoint_prior_scale": 0.05}
    -> {"xgb": {"max_depth": 8}, "prophet": {"changepoint_prior_scale": 0.05}}

    Keys without "__" stay at the top level.
    """
    result = {}
    for key, val in flat_params.items():
        if "__" in key:
            section, param = key.split("__", 1)
            result.setdefault(section, {})[param] = val
        else:
            result[key] = val
    return result


def _evaluate_trial(
    df: pd.DataFrame,
    intervention_date,
    model_name: str,
    params: dict,
    cv_kwargs: dict,
) -> dict:
    """Run time_series_cv for one parameter combination and return metric dict.

    Returns a dict with inf values (not an exception) if the trial fails,
    so a single unstable parameter set cannot abort the full search.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_result = time_series_cv(
                df=df,
                intervention_date=intervention_date,
                model_name=model_name,
                config_overrides={"models": {model_name: params}},
                **cv_kwargs,
            )
        return {
            "mean_rmse":  cv_result.mean_rmse,
            "std_rmse":   cv_result.std_rmse,
            "mean_mae":   cv_result.mean_mae,
            "mean_mape":  cv_result.mean_mape,
            "mean_r2":    cv_result.mean_r2,
            "n_folds_ok": len(cv_result.folds),
        }
    except Exception as exc:
        logger.warning("Trial failed for %s: %s", model_name, exc)
        return {
            "mean_rmse":  float("inf"),
            "std_rmse":   float("nan"),
            "mean_mae":   float("inf"),
            "mean_mape":  float("nan"),
            "mean_r2":    float("nan"),
            "n_folds_ok": 0,
        }


def tune_model(
    df: pd.DataFrame,
    intervention_date,
    model_name: str,
    n_trials: int | None = None,
    n_folds: int = 5,
    test_days: int = 365,
    min_train_days: int = 730,
    skip_days: int = 0,
    cv_end_date=None,
    metric: str = "rmse",
    config_path=None,
    n_jobs: int = 1,
    seed: int = 42,
) -> TuningResult:
    """Tune model hyperparameters via Latin hypercube grid search with time-series CV.

    Mirrors the R reference implementation (Two_Stage_ITS): a one-shot space-filling
    sample of the parameter space is evaluated via expanding-window CV, and the
    combination with the lowest mean CV RMSE (or MAE) is selected.

    R reference CV settings: 5 folds, 12-month validation window, 2-year initial
    training window, 12-month skip between folds. Matching those settings:
        n_folds=5, test_days=365, min_train_days=730, skip_days=365

    To prevent tuning from seeing the held-out evaluation window that
    run_single_its uses, set cv_end_date to intervention_date minus test_days.

    Parameters
    ----------
    df : pd.DataFrame
        Full time series dataset.
    intervention_date : str or pd.Timestamp
        Only pre-intervention data is used for tuning CV.
    model_name : str
        One of "arima", "neuralprophet", "prophet_xgb", "prophet_then_xgb".
    n_trials : int or None
        Number of parameter combinations to evaluate. Defaults to model-specific
        values matching R reference (100 for most models, 75 for neuralprophet).
    n_folds : int
        Number of expanding-window CV folds.
    test_days : int
        Validation window per fold in days.
    min_train_days : int
        Minimum training window for the first fold in days.
    skip_days : int
        Gap in days between consecutive fold validation windows. Set to 365 to
        match the R reference (skip = "12 months"). Defaults to 0 (adjacent folds).
    cv_end_date : str or pd.Timestamp, optional
        Upper bound on data used for CV folds. Must be <= intervention_date.
        Pass intervention_date - pd.Timedelta(days=test_days) to prevent tuning
        folds from overlapping with the held-out evaluation window.
        Defaults to None (use all pre-intervention data).
    metric : str
        Objective for selecting the best parameter set. "rmse" or "mae".
    config_path : str or Path, optional
        Path to a custom base YAML config (merged before tuning overrides).
    n_jobs : int
        Parallel workers for evaluating trials. -1 uses all available cores.
    seed : int
        Random seed for the Latin hypercube sampler.

    Returns
    -------
    TuningResult
        Contains best_params (inject via run_single_its config_overrides),
        trials_df (all evaluated combinations and their CV metrics), and
        summary statistics.

    Examples
    --------
    Tune and apply best params (R-matched CV settings, leakage-free):

        import pandas as pd
        result = tune_model(
            df, "2025-01-07", "prophet_xgb",
            n_trials=100, n_folds=5,
            test_days=365, min_train_days=730, skip_days=365,
            cv_end_date=pd.Timestamp("2025-01-07") - pd.Timedelta(days=365),
        )
        run_single_its(
            df, "2025-01-07",
            model_name="prophet_xgb",
            config_overrides={"models": {"prophet_xgb": result.best_params}},
        )
    """
    if model_name not in _SEARCH_SPACES:
        raise ValueError(
            f"No search space defined for '{model_name}'. "
            f"Available: {list(_SEARCH_SPACES)}"
        )
    if metric not in ("rmse", "mae"):
        raise ValueError(f"metric must be 'rmse' or 'mae', got '{metric}'")

    n_trials = n_trials if n_trials is not None else _DEFAULT_N_TRIALS[model_name]
    search_space = _SEARCH_SPACES[model_name]

    flat_trials = _sample_lhs(search_space, n_trials, seed)
    nested_trials = [_unflatten_params(p) for p in flat_trials]

    cv_kwargs = {
        "n_folds":        n_folds,
        "test_days":      test_days,
        "min_train_days": min_train_days,
        "skip_days":      skip_days,
        "cv_end_date":    cv_end_date,
        "config_path":    config_path,
    }

    logger.info(
        "Tuning %s: %d trials x %d folds (metric=%s, n_jobs=%d)",
        model_name, n_trials, n_folds, metric, n_jobs,
    )

    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_trial)(df, intervention_date, model_name, params, cv_kwargs)
        for params in nested_trials
    )

    rows = []
    for i, (flat_p, metrics) in enumerate(zip(flat_trials, results)):
        row = {"trial_id": i}
        row.update(flat_p)
        row.update(metrics)
        rows.append(row)

    trials_df = pd.DataFrame(rows)

    metric_col = f"mean_{metric}"
    best_idx = int(trials_df[metric_col].idxmin())
    best_row = trials_df.loc[best_idx]
    best_params = nested_trials[best_idx]

    logger.info(
        "Tuning complete: %s | best %s=%.4f +/- %.4f (trial %d/%d)",
        model_name, metric_col,
        best_row[metric_col], best_row["std_rmse"],
        best_idx, n_trials,
    )

    return TuningResult(
        model_name=model_name,
        best_params=best_params,
        best_rmse=float(best_row["mean_rmse"]),
        best_std_rmse=float(best_row["std_rmse"]) if not math.isnan(best_row["std_rmse"]) else 0.0,
        trials_df=trials_df,
        n_trials=n_trials,
        n_folds=n_folds,
        metric=metric,
        seed=seed,
    )
