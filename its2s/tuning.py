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

from .cross_validation import (_CV_METHOD_ARGS, _default_cv_end_date,
                               time_series_cv)
from .settings import _deep_merge, load_config

logger = logging.getLogger(__name__)

# Default trial counts per model, matching the R reference implementation:
# ARIMA=100, NNETAR (-> NeuralProphet)=75, prophet_xgb=100
_DEFAULT_N_TRIALS = {
    "prophet_xgb":      100,
    "neuralprophet":    75,
    "arima":            100,
}

# Search spaces: param_name -> (low, high, dtype, scale)
# These are the tuning ranges used by the Latin hypercube sampler. They are
# independent of params.yaml, which holds default (non-tuning) fallback values.
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
    # n_lags bounds are OBSERVATION counts (7..30 rows), not calendar days.
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
    "prophet_xgb":      _PROPHET_XGB_SPACE,
    "neuralprophet":    _NEURALPROPHET_SPACE,
    "arima":            _ARIMA_SPACE,
}


@dataclass
class TuningResult:
    """Result from a hyperparameter tuning run.

    Attributes
    ----------
    model_name : str
        Name of the tuned model (e.g. ``"arima"``).
    best_params : dict
        Nested param dict ready to pass as ``config_overrides["models"][model_name]``.
    best_rmse : float
        Mean CV RMSE of the best parameter combination.
    best_std_rmse : float
        Std dev of CV RMSE across folds for the best combination.
    trials_df : pd.DataFrame
        One row per trial. Columns: ``trial_id``, ``<param cols>``, ``mean_rmse``,
        ``std_rmse``, ``mean_mae``, ``mean_mape``, ``n_folds_ok``.
    n_trials : int
        Number of parameter combinations evaluated.
    n_folds : int
        Number of expanding-window CV folds used per trial.
    metric : str
        Objective used for selection (``"rmse"`` or ``"mae"``).
    seed : int
        Random seed driving the Latin hypercube sample.
    cv_end_date : pd.Timestamp or None
        Effective CV cap used for every trial; derived from the run's
        held-out test split when not passed explicitly (GH #40).
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
    cv_end_date: pd.Timestamp | None = None


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
    user_overrides: dict | None = None,
) -> dict:
    """Run time_series_cv for one parameter combination and return metric dict.

    Returns a dict with inf values (not an exception) if the trial fails,
    so a single unstable parameter set cannot abort the full search.
    Trial params are merged on top of any user config_overrides, so the
    sampled values always win over user model overrides.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_result = time_series_cv(
                df=df,
                intervention_date=intervention_date,
                model_name=model_name,
                config_overrides=_deep_merge(
                    user_overrides or {}, {"models": {model_name: params}}),
                **cv_kwargs,
            )
        return {
            "mean_rmse":  cv_result.mean_rmse,
            "std_rmse":   cv_result.std_rmse,
            "mean_mae":   cv_result.mean_mae,
            "mean_mape":  cv_result.mean_mape,
            "n_folds_ok": len(cv_result.folds),
        }
    except Exception as exc:
        logger.warning("Trial failed for %s: %s", model_name, exc)
        return {
            "mean_rmse":  float("inf"),
            "std_rmse":   float("nan"),
            "mean_mae":   float("inf"),
            "mean_mape":  float("nan"),
            "n_folds_ok": 0,
        }


def tune_model(
    df: pd.DataFrame,
    intervention_date,
    model_name: str,
    n_trials: int | None = None,
    n_folds: int = 5,
    test_obs: int | None = None,
    min_train_obs: int | None = None,
    skip_obs: int | None = None,
    cv_end_date=None,
    split_method: str = "percent",
    test_pct: float | None = None,
    min_train_pct: float | None = None,
    skip_pct: float | None = None,
    metric: str = "rmse",
    config_path=None,
    config_overrides: dict | None = None,
    n_jobs: int = 1,
    seed: int = 42,
) -> TuningResult:
    """Tune model hyperparameters via Latin hypercube grid search with time-series CV.

    Mirrors the R reference implementation (Two_Stage_ITS): a one-shot space-filling
    sample of the parameter space is evaluated via expanding-window CV, and the
    combination with the lowest mean CV RMSE (or MAE) is selected.

    All CV windows are sized in OBSERVATIONS (rows), never calendar days --
    see time_series_cv. The R reference CV settings are 5 folds, 12-month
    validation window, 2-year initial training window, 12-month skip between
    folds; on DAILY data those translate to:
        split_method="observations", n_folds=5,
        test_obs=365, min_train_obs=730, skip_obs=365
    (on any other frequency, convert months to observation counts first).

    Only the window arguments belonging to the chosen ``split_method`` may be
    passed; arguments for the other method raise ValueError rather than being
    silently ignored.

    By default, tuning folds are capped at the start of the held-out test
    window that run_single_its will evaluate on, derived row-exactly from the
    config's "periods" section (GH #40). Pass the same config_path /
    config_overrides you will run with so the derived boundary matches the
    run; pass cv_end_date=intervention_date to deliberately tune on all
    pre-intervention data.

    Parameters
    ----------
    df : pd.DataFrame
        Full time series dataset.
    intervention_date : str or pd.Timestamp
        Only pre-intervention data is used for tuning CV.
    model_name : str
        One of "arima", "neuralprophet", "prophet_xgb".
    n_trials : int or None
        Number of parameter combinations to evaluate. Defaults to model-specific
        values matching R reference (100 for most models, 75 for neuralprophet).
    n_folds : int
        Number of expanding-window CV folds.
    split_method : {"percent", "observations"}
        "percent" (default): size the fold windows as fractions of the CV
        observations via `test_pct`/`min_train_pct`/`skip_pct`.
        "observations": size them as explicit observation counts via
        `test_obs`/`min_train_obs`/`skip_obs`.
    test_obs : int
        Validation window per fold in observations. Defaults to 365.
        Only with ``split_method="observations"``.
    min_train_obs : int
        Minimum training window for the first fold, in observations.
        Defaults to 730. Only with ``split_method="observations"``.
    skip_obs : int
        Gap in observations between consecutive fold validation windows. Set
        to 365 on daily data to match the R reference (skip = "12 months").
        Defaults to 0 (adjacent folds). Only with
        ``split_method="observations"``.
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
        Upper bound on data used for CV folds. Must be <= intervention_date.
        Defaults to the first date of the held-out test window that
        prepare_splits produces for this df and the loaded config's
        "periods" section, so tuning folds never touch the window
        run_single_its evaluates on (GH #40). The derivation is row-exact
        for every split method and series frequency; tune on the same
        missing-handled DataFrame and periods config you will run with.
        Pass cv_end_date=intervention_date explicitly to tune on all
        pre-intervention data.
    metric : str
        Objective for selecting the best parameter set. "rmse" or "mae".
    config_path : str or Path, optional
        Path to a custom base YAML config (merged before tuning overrides).
    config_overrides : dict, optional
        Runtime config overrides, as in run_single_its. Pass the same
        overrides here that the run will use -- in particular any "periods"
        override -- so the derived cv_end_date matches the run's actual
        test window. Per-trial model params are merged on top and always
        win over model overrides given here.
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
    Tune and apply best params on a DAILY series (R-matched CV settings;
    365 observations = 365 calendar days only because the series is daily).
    Tuning folds stop before the run's held-out test window by default;
    passing the same periods override to both calls keeps the derived
    boundary and the run's actual window identical:

        overrides = {"periods": {"split_method": "days",
                                 "test_days": 365, "holdout_days": 365}}
        result = tune_model(
            df, "2025-01-07", "prophet_xgb",
            n_trials=100, n_folds=5,
            split_method="observations",
            test_obs=365, min_train_obs=730, skip_obs=365,
            config_overrides=overrides,
        )
        run_single_its(
            df, "2025-01-07",
            model_name="prophet_xgb",
            config_overrides={"models": {"prophet_xgb": result.best_params},
                              **overrides},
        )
    """
    if model_name not in _SEARCH_SPACES:
        raise ValueError(
            f"No search space defined for '{model_name}'. "
            f"Available: {list(_SEARCH_SPACES)}"
        )
    if metric not in ("rmse", "mae"):
        raise ValueError(f"metric must be 'rmse' or 'mae', got '{metric}'")

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

    n_trials = n_trials if n_trials is not None else _DEFAULT_N_TRIALS[model_name]

    # M2-3: Lower-bound parameter checks
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}.")
    if n_folds < 2:
        raise ValueError(
            f"n_folds must be >= 2, got {n_folds}. "
            "At least 2 folds are required for meaningful cross-validation."
        )

    # Resolve the CV cap once, upfront, so every trial sees the same concrete
    # date and an invalid explicit value raises here rather than being
    # swallowed into inf metrics by _evaluate_trial (GH #40).
    intervention_ts = pd.Timestamp(intervention_date)
    if cv_end_date is not None:
        cv_end_date = pd.Timestamp(cv_end_date)
        if cv_end_date > intervention_ts:
            raise ValueError(
                f"cv_end_date ({cv_end_date.date()}) must be <= "
                f"intervention_date ({intervention_ts.date()})."
            )
    else:
        config = load_config(config_path, config_overrides)
        cv_end_date = _default_cv_end_date(
            df, intervention_date, config["data"]["date_col"], config)

    cv_kwargs = {
        "n_folds":        n_folds,
        "cv_end_date":    cv_end_date,
        "split_method":   split_method,
        "config_path":    config_path,
    }
    if split_method == "observations":
        test_obs = 365 if test_obs is None else test_obs
        min_train_obs = 730 if min_train_obs is None else min_train_obs
        skip_obs = 0 if skip_obs is None else skip_obs
        if test_obs < 1:
            raise ValueError(f"test_obs must be >= 1, got {test_obs}.")
        cv_kwargs.update(test_obs=test_obs, min_train_obs=min_train_obs,
                         skip_obs=skip_obs)
    else:
        test_pct = 0.10 if test_pct is None else test_pct
        min_train_pct = 0.50 if min_train_pct is None else min_train_pct
        skip_pct = 0.0 if skip_pct is None else skip_pct
        cv_kwargs.update(test_pct=test_pct, min_train_pct=min_train_pct,
                         skip_pct=skip_pct)

    search_space = _SEARCH_SPACES[model_name]

    flat_trials = _sample_lhs(search_space, n_trials, seed)
    nested_trials = [_unflatten_params(p) for p in flat_trials]

    logger.info(
        "Tuning %s: %d trials x %d folds (metric=%s, n_jobs=%d)",
        model_name, n_trials, n_folds, metric, n_jobs,
    )

    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_trial)(df, intervention_date, model_name, params,
                                 cv_kwargs, config_overrides)
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
        cv_end_date=cv_end_date,
    )
