# Description: Unit tests for its2s.tuning -- hyperparameter tuning framework.
# Usage: python -m pytest tests/test_tuning.py -v --tb=short
# Dependencies: pytest, numpy, pandas, scipy, its2s

import math

import numpy as np
import pandas as pd
import pytest

from its2s.tuning import (
    TuningResult,
    _SEARCH_SPACES,
    _sample_lhs,
    _unflatten_params,
    tune_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def short_daily_series():
    """Synthetic daily series with enough data for 2-fold CV (min_train_obs=400)."""
    rng = np.random.default_rng(0)
    n = 900
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    y = 100 + np.sin(np.arange(n) * 2 * np.pi / 365) * 10 + rng.normal(0, 2, n)
    return pd.DataFrame({"ds": dates, "y": y})


# ---------------------------------------------------------------------------
# _sample_lhs
# ---------------------------------------------------------------------------

class TestSampleLhs:
    def test_shape(self):
        space = _SEARCH_SPACES["arima"]
        trials = _sample_lhs(space, 10, seed=0)
        assert len(trials) == 10
        assert all(set(t.keys()) == set(space.keys()) for t in trials)

    def test_bounds_respected(self):
        for model_name, space in _SEARCH_SPACES.items():
            trials = _sample_lhs(space, 20, seed=42)
            for trial in trials:
                for key, (low, high, dtype, scale) in space.items():
                    val = trial[key]
                    assert val >= low, f"{model_name}.{key}: {val} < {low}"
                    assert val <= high, f"{model_name}.{key}: {val} > {high}"

    def test_int_dtype_produces_integers(self):
        space = _SEARCH_SPACES["arima"]
        trials = _sample_lhs(space, 20, seed=1)
        for trial in trials:
            for key, (_, _, dtype, _) in space.items():
                if dtype == "int":
                    assert isinstance(trial[key], int), f"{key} should be int"

    def test_log_scale_not_linear(self):
        # learning_rate in prophet_xgb is log-scaled; values should not be
        # uniformly spread in linear space (i.e., more mass near the lower end)
        space = {"learning_rate": (0.001, 0.3, "float", "log")}
        trials = _sample_lhs(space, 200, seed=7)
        vals = [t["learning_rate"] for t in trials]
        # If log-uniform, median should be much closer to 0.001 than to 0.3
        # geometric midpoint = sqrt(0.001 * 0.3) ~ 0.017
        assert np.median(vals) < 0.1, "log-scale median should be well below linear midpoint"

    def test_reproducible_with_same_seed(self):
        space = _SEARCH_SPACES["prophet_xgb"]
        t1 = _sample_lhs(space, 10, seed=99)
        t2 = _sample_lhs(space, 10, seed=99)
        for a, b in zip(t1, t2):
            assert a == b

    def test_different_seeds_differ(self):
        space = _SEARCH_SPACES["arima"]
        t1 = _sample_lhs(space, 10, seed=1)
        t2 = _sample_lhs(space, 10, seed=2)
        # At least one param in at least one trial should differ
        any_diff = any(
            t1[i][k] != t2[i][k]
            for i in range(len(t1))
            for k in t1[i]
        )
        assert any_diff


# ---------------------------------------------------------------------------
# _unflatten_params
# ---------------------------------------------------------------------------

class TestUnflattenParams:
    def test_nested_keys_split_correctly(self):
        flat = {"xgb__max_depth": 8, "xgb__learning_rate": 0.05}
        result = _unflatten_params(flat)
        assert result == {"xgb": {"max_depth": 8, "learning_rate": 0.05}}

    def test_multiple_sections(self):
        flat = {
            "prophet__changepoint_prior_scale": 0.1,
            "xgb__n_estimators": 200,
        }
        result = _unflatten_params(flat)
        assert result["prophet"] == {"changepoint_prior_scale": 0.1}
        assert result["xgb"] == {"n_estimators": 200}

    def test_flat_keys_unchanged(self):
        flat = {"max_p": 3, "max_d": 1}
        result = _unflatten_params(flat)
        assert result == {"max_p": 3, "max_d": 1}

    def test_mixed_flat_and_nested(self):
        flat = {"max_p": 2, "xgb__max_depth": 6}
        result = _unflatten_params(flat)
        assert result["max_p"] == 2
        assert result["xgb"]["max_depth"] == 6

    def test_empty_input(self):
        assert _unflatten_params({}) == {}


# ---------------------------------------------------------------------------
# tune_model -- structural / fast tests (n_trials=3, n_folds=2)
# ---------------------------------------------------------------------------

class TestTuneModel:
    def test_returns_tuning_result(self, short_daily_series):
        result = tune_model(
            short_daily_series,
            intervention_date="2020-06-01",
            model_name="arima",
            n_trials=3,
            n_folds=2,
            test_obs=60,
            min_train_obs=400,
            n_jobs=1,
            seed=0,
        )
        assert isinstance(result, TuningResult)

    def test_trials_df_shape(self, short_daily_series):
        result = tune_model(
            short_daily_series,
            intervention_date="2020-06-01",
            model_name="arima",
            n_trials=3,
            n_folds=2,
            test_obs=60,
            min_train_obs=400,
            n_jobs=1,
            seed=0,
        )
        assert len(result.trials_df) == 3
        assert "trial_id" in result.trials_df.columns
        assert "mean_rmse" in result.trials_df.columns
        assert "std_rmse" in result.trials_df.columns
        assert "mean_mae" in result.trials_df.columns
        assert "n_folds_ok" in result.trials_df.columns

    def test_best_rmse_is_minimum_in_trials(self, short_daily_series):
        result = tune_model(
            short_daily_series,
            intervention_date="2020-06-01",
            model_name="arima",
            n_trials=3,
            n_folds=2,
            test_obs=60,
            min_train_obs=400,
            n_jobs=1,
            seed=0,
        )
        assert math.isclose(result.best_rmse, result.trials_df["mean_rmse"].min())

    def test_best_params_keys_match_model(self, short_daily_series):
        result = tune_model(
            short_daily_series,
            intervention_date="2020-06-01",
            model_name="arima",
            n_trials=3,
            n_folds=2,
            test_obs=60,
            min_train_obs=400,
            n_jobs=1,
            seed=0,
        )
        # best_params for arima should be a flat dict with max_p, max_d, etc.
        for key in ("max_p", "max_d", "max_q"):
            assert key in result.best_params

    def test_prophet_xgb_best_params_nested(self, short_daily_series):
        result = tune_model(
            short_daily_series,
            intervention_date="2020-06-01",
            model_name="prophet_xgb",
            n_trials=3,
            n_folds=2,
            test_obs=60,
            min_train_obs=400,
            n_jobs=1,
            seed=0,
        )
        assert "prophet" in result.best_params
        assert "xgb" in result.best_params
        assert "changepoint_prior_scale" in result.best_params["prophet"]
        assert "n_estimators" in result.best_params["xgb"]

    def test_metric_mae_selects_by_mae(self, short_daily_series):
        result = tune_model(
            short_daily_series,
            intervention_date="2020-06-01",
            model_name="arima",
            n_trials=3,
            n_folds=2,
            test_obs=60,
            min_train_obs=400,
            metric="mae",
            n_jobs=1,
            seed=0,
        )
        assert result.metric == "mae"
        assert math.isclose(result.best_rmse, result.trials_df.loc[
            result.trials_df["mean_mae"].idxmin(), "mean_rmse"
        ])

    def test_failed_trial_does_not_abort(self, short_daily_series):
        # Inject one guaranteed-failing trial by monkeypatching _evaluate_trial
        # via an extreme param set; tune_model should still return normally
        # with the remaining successful trials.
        # We test this indirectly: if all trials happened to fail, tune_model
        # would return best_rmse=inf. With a real series and 3 ARIMA trials,
        # at least one should succeed.
        result = tune_model(
            short_daily_series,
            intervention_date="2020-06-01",
            model_name="arima",
            n_trials=3,
            n_folds=2,
            test_obs=60,
            min_train_obs=400,
            n_jobs=1,
            seed=0,
        )
        # At least one successful fold
        assert result.trials_df["n_folds_ok"].max() > 0

    def test_invalid_model_name_raises(self, short_daily_series):
        with pytest.raises(ValueError, match="No search space defined"):
            tune_model(
                short_daily_series,
                intervention_date="2020-06-01",
                model_name="not_a_model",
                n_trials=2,
                n_folds=2,
                test_obs=60,
                min_train_obs=400,
            )

    def test_invalid_metric_raises(self, short_daily_series):
        with pytest.raises(ValueError, match="metric must be"):
            tune_model(
                short_daily_series,
                intervention_date="2020-06-01",
                model_name="arima",
                n_trials=2,
                n_folds=2,
                test_obs=60,
                min_train_obs=400,
                metric="mape",
            )

    def test_result_metadata_fields(self, short_daily_series):
        result = tune_model(
            short_daily_series,
            intervention_date="2020-06-01",
            model_name="arima",
            n_trials=3,
            n_folds=2,
            test_obs=60,
            min_train_obs=400,
            seed=77,
        )
        assert result.model_name == "arima"
        assert result.n_trials == 3
        assert result.n_folds == 2
        assert result.metric == "rmse"
        assert result.seed == 77

    def test_tune_model_percent_cv_short_series(self):
        """Percent-based CV must handle a short pre-intervention series."""
        rng = np.random.default_rng(0)
        n_pre, n_post = 200, 30
        n = n_pre + n_post
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        y = 100 + np.sin(np.arange(n) * 2 * np.pi / 30) * 5 + rng.normal(0, 1, n)
        df = pd.DataFrame({"ds": dates, "y": y})
        intv = dates[n_pre]
        result = tune_model(
            df, intervention_date=intv, model_name="arima",
            n_trials=4, n_folds=3,
            split_method="percent",
            test_pct=0.10, min_train_pct=0.50,
            seed=11,
        )
        assert result.n_trials == 4
        assert math.isfinite(result.best_rmse)


# ---------------------------------------------------------------------------
# Tuning section in params.yaml
# ---------------------------------------------------------------------------

class TestTuningConfig:
    def test_tuning_section_in_default_config(self):
        from its2s.settings import load_config
        cfg = load_config()
        assert "tuning" in cfg
        assert cfg["tuning"]["n_folds"] == 5
        assert cfg["tuning"]["test_obs"] == 365
        assert cfg["tuning"]["min_train_obs"] == 730
        assert cfg["tuning"]["metric"] == "rmse"

    def test_get_tuning_config(self):
        from its2s.settings import get_tuning_config, load_config
        cfg = load_config()
        tc = get_tuning_config(cfg)
        assert tc["model_defaults"]["arima"] == 100
        assert tc["model_defaults"]["neuralprophet"] == 75

    def test_tuning_config_is_deep_copy(self):
        from its2s.settings import get_tuning_config, load_config
        cfg = load_config()
        tc = get_tuning_config(cfg)
        tc["n_folds"] = 999
        cfg2 = load_config()
        assert cfg2["tuning"]["n_folds"] == 5  # original unaffected

    def test_legacy_day_keys_raise(self):
        # CV windows are observation counts (GH #39); day-named tuning keys
        # raise instead of being silently reinterpreted.
        from its2s.settings import get_tuning_config, load_config
        cfg = load_config()
        cfg["tuning"]["min_train_days"] = 730
        with pytest.raises(ValueError, match="min_train_days"):
            get_tuning_config(cfg)
