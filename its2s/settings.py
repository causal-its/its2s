# Description: Default configuration for the old path (run_single_its,
#   tune_model, time_series_cv), merged with runtime overrides. Model defaults
#   live in the family specs (its2s.models.families); this module only adds
#   the old path's seasonal "auto" keys, which it turns into a declaration
#   (its2s.seasonality.legacy_seasonality). Deleted with the old path in
#   P11.04.07; the YAML file it replaced retired in P11.04.02.
# Usage: from its2s.settings import load_config, get_model_config
# Dependencies: none (stdlib only)

import copy

from .models.families import FAMILIES


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


# The old path's seasonal keys, resolved per training window by
# legacy_seasonality: Prophet and NeuralProphet "auto" follow the 730-day
# yearly rule and the spacing rule for weekly (GH #60, D-057, D-080); ARIMA
# "auto" derives m from the series frequency with the n_train >= 2m guard
# (GH #59).
_LEGACY_SEASONAL_KEYS = {
    "prophet_xgb": {"prophet": {"yearly_seasonality": "auto",
                                "weekly_seasonality": "auto",
                                "daily_seasonality": False}},
    "neuralprophet": {"yearly_seasonality": "auto", "weekly_seasonality": "auto"},
    "arima": {"seasonal": True, "m": "auto"},
}

# Override precedence: config_overrides passed at runtime, then these.
# After tuning, inject best_params via:
#   run_single_its(..., config_overrides={"models": {model_name: result.best_params}})
_LEGACY_DEFAULTS = {
    "data": {
        "date_col": "ds",
        "target_col": "y",
        "covariate_cols": [],
        "missing_data": "error",    # "error", "drop", or "interpolate"
    },
    # Window sizes default in code per family (prepare_splits): percent
    # test_pct 0.20 / post_event_pct 1.0, days test_days 365 / post_event_days
    # 365, observations required with no defaults. They are NOT set here:
    # only the family matching split_method may appear in config, and keys
    # from another family raise instead of being silently ignored (GH #28,
    # #54), so defaults for every family here would make the non-default
    # methods unreachable.
    "periods": {
        "split_method": "percent",  # "percent" | "days" | "observations"
        "min_test_obs": 30,         # warn below this many test observations (0 disables); GH #29
    },
    "models": {name: _deep_merge(spec.defaults, _LEGACY_SEASONAL_KEYS[name])
               for name, spec in FAMILIES.items()},
    "bootstrap": {
        "method": "mbb",
        "n_sim": 1000,
        # Block length in OBSERVATIONS, not days: 14 = two weeks on daily
        # data but 14 weeks on a weekly series. GH #59
        "block_length": 14,
        "ci_method": "quantile",
        "ci_level": 0.95,
        "n_jobs": 1,
    },
    "metrics": {
        # MASE benchmark period m. "auto" derives the dominant seasonal cycle
        # from the resolved series frequency; an integer is honored or fails
        # hard (GH #62).
        "seasonality": "auto",
    },
    "output": {
        "format": "csv",
        "plot_dpi": 150,
        "plot_figsize": [14, 6],
        # Sequence overrides map to: fit, counterfactual/CI, intervention,
        # post_event. Dict overrides may also include observed.
        "plot_colors": {
            "fit": "#2166AC",
            "counterfactual": "#B2182B",
            "intervention": "#4DAF4A",
            "post_event": "#FEE08B",
            "observed": "#333333",
        },
        "plot_font_sizes": {"title": 22, "axis_label": 20, "tick": 18, "legend": 18},
    },
    "parallel": {"n_jobs": 1},
    # Offsets are unit-explicit: start_offset_days/end_offset_days OR
    # start_offset_obs/end_offset_obs per period; mixing them raises.
    "excess_periods": [],
    # CV window sizes default in code per family (get_tuning_config); only
    # the family matching split_method may appear (GH #55). cv_end_date is
    # derived at runtime from the 'periods' split (GH #40).
    "tuning": {
        "split_method": "percent",
        "n_folds": 5,
        "metric": "rmse",
        "n_jobs": 1,
        "seed": 42,
        "model_defaults": {name: spec.starter_n_trials
                           for name, spec in FAMILIES.items()},
    },
}


def load_config(overrides=None):
    """Return the old path's default config with overrides merged on top.

    Parameters
    ----------
    overrides : dict, optional
        Runtime overrides applied last.

    Returns
    -------
    dict
    """
    config = copy.deepcopy(_LEGACY_DEFAULTS)
    if overrides:
        config = _deep_merge(config, overrides)
    return config


def get_model_config(config, model_name):
    """Extract model-specific config section.

    Parameters
    ----------
    config : dict
    model_name : str

    Returns
    -------
    dict
    """
    return copy.deepcopy(config.get("models", {}).get(model_name, {}))


def get_tuning_config(config):
    """Extract the tuning defaults section.

    Fills in sensible defaults for percent-based and observation-based CV
    controls so callers (e.g. tune_model) can read either path uniformly.
    Legacy day-named keys raise: CV windows are observation counts (GH #39),
    and silently reinterpreting old keys is the failure mode GH #28 showed.
    Keys from the window family not selected by split_method also raise, so
    a config cannot carry settings that would be silently ignored (GH #55).

    Parameters
    ----------
    config : dict

    Returns
    -------
    dict
    """
    tuning = copy.deepcopy(config.get("tuning", {}))
    legacy = [k for k in ("test_days", "min_train_days", "skip_days")
              if k in tuning]
    if legacy:
        raise ValueError(
            f"Legacy tuning keys {legacy} are no longer supported. CV windows "
            "are observation counts: use test_obs/min_train_obs/skip_obs with "
            "split_method 'observations'."
        )
    tuning.setdefault("split_method", "percent")
    tuning.setdefault("n_folds", 5)
    tuning.setdefault("metric", "rmse")
    tuning.setdefault("n_jobs", 1)
    tuning.setdefault("seed", 42)
    families = {"percent":      ("test_pct", "min_train_pct", "skip_pct"),
                "observations": ("test_obs", "min_train_obs", "skip_obs")}
    method = tuning["split_method"]
    if method not in families:
        raise ValueError(
            f"tuning.split_method must be 'percent' or 'observations', "
            f"got {method!r}."
        )
    foreign = [k for fam in families.values() for k in fam
               if k in tuning and k not in families[method]]
    if foreign:
        raise ValueError(
            f"Tuning keys {foreign} do not apply to split_method="
            f"{method!r}, which uses {list(families[method])}. Set the "
            "keys for the chosen split_method only."
        )
    if method == "percent":
        tuning.setdefault("test_pct", 0.10)
        tuning.setdefault("min_train_pct", 0.50)
        tuning.setdefault("skip_pct", 0.0)
    else:
        tuning.setdefault("test_obs", 365)
        tuning.setdefault("min_train_obs", 730)
        tuning.setdefault("skip_obs", 0)
    return tuning
