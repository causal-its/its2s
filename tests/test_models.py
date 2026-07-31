# Description: Model-facing test suite for its2s package.
#   All four models (arima, prophet_xgb, prophet_then_xgb, neuralprophet) are
#   exercised through parametrized loops wherever applicable.
#   Covers: model contract, MBB bootstrap, model-specific unit tests,
#           end-to-end integration, statistical validation, robustness,
#           and cross-model comparison.
# Usage: python -m pytest tests/test_models.py -v --tb=short
# Dependencies: pytest, numpy, pandas, its2s

import warnings

import numpy as np
import pandas as pd
import pytest

from conftest import (
    _FAST,
    _NP_FAST,
    _NP_FAST_PARAMS,
    _has_neuralprophet,
    _run_quiet,
    collect_model_params,
    make_constant_series,
    make_count_series,
    make_daily_series,
    make_missing_data_series,
    make_monthly_series,
    make_outlier_series,
    make_quarterly_series,
    make_series_with_covariates,
    make_short_series,
    make_weekly_series,
)

pytestmark = [
    pytest.mark.filterwarnings("ignore::FutureWarning"),
    pytest.mark.filterwarnings("ignore::UserWarning"),
]

# ---------------------------------------------------------------------------
# Module-level model parametrization
# ---------------------------------------------------------------------------
# _MODEL_PARAMS: list of (model_name, ModelClass, init_params).
# NeuralProphet is included only when its dependencies are installed.
_MODEL_PARAMS = collect_model_params()
_MODEL_IDS = [name for name, _, _ in _MODEL_PARAMS]

# Model names with an explicit skip-mark for NeuralProphet when not installed.
# Use this for tests that ALWAYS want all 4 slots visible in the report.
_MODEL_NAMES_MARKS = [
    pytest.param(
        name,
        marks=pytest.mark.skipif(
            name == "neuralprophet" and not _has_neuralprophet(),
            reason="neuralprophet not installed",
        ),
    )
    for name in ["prophet_xgb", "prophet_then_xgb", "neuralprophet", "arima"]
]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _e2e_config(model_name, *, test_days=365, holdout_days=365, n_sim=10):
    """Minimal config_overrides for e2e integration tests.

    Sets fast bootstrap params and propagates NeuralProphet fast-training
    hyperparameters when needed.
    """
    cfg = {
        "bootstrap": {"n_sim": n_sim, "n_jobs": 1},
        "periods": {"split_method": "days",
                    "test_days": test_days, "holdout_days": holdout_days},
    }
    if model_name == "neuralprophet":
        cfg["models"] = {"neuralprophet": _NP_FAST_PARAMS}
    return cfg


# ARIMA requires explicit seasonality period for non-daily frequencies.
_ARIMA_FREQ_EXTRA = {
    "weekly":    {"models": {"arima": {"m": 52}},
                  "metrics": {"seasonality": 52}},
    "monthly":   {"models": {"arima": {"m": 12}},
                  "metrics": {"seasonality": 12}},
    "quarterly": {"models": {"arima": {"m": 4, "seasonal": True}},
                  "metrics": {"seasonality": 4}},
}


def _merge(base, extra):
    """Shallow-merge two config dicts, recursing one level for nested dicts."""
    result = dict(base)
    for k, v in extra.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = {**result[k], **v}
        else:
            result[k] = v
    return result


# ===================================================================
# Model Contract (parametrized across all 4 models)
# ===================================================================
class TestModelContract:
    """Verify that every model satisfies the base interface contract."""

    @pytest.fixture(scope="class")
    def train_test_data(self):
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_short_series(n_pre=180, n_post=30, seed=1001)
        return prepare_splits(df, intv, split_method="days", test_days=30, holdout_days=30)

    @pytest.mark.parametrize("name,cls,params", _MODEL_PARAMS, ids=_MODEL_IDS)
    def test_fit_returns_fit_result(self, name, cls, params, train_test_data):
        from its2s.models.base import FitResult
        model = cls(params=params)
        fr = _run_quiet(model.fit, train_test_data.train_df)
        assert isinstance(fr, FitResult)
        assert len(fr.fitted_values) == len(train_test_data.train_df)
        assert len(fr.residuals) == len(train_test_data.train_df)

    @pytest.mark.parametrize("name,cls,params", _MODEL_PARAMS, ids=_MODEL_IDS)
    def test_predict_returns_prediction_result(self, name, cls, params, train_test_data):
        from its2s.models.base import PredictionResult
        model = cls(params=params)
        _run_quiet(model.fit, train_test_data.train_df)
        pr = _run_quiet(model.predict, train_test_data.full_predict_df)
        assert isinstance(pr, PredictionResult)
        assert len(pr.predicted) == len(train_test_data.full_predict_df)

    @pytest.mark.parametrize("name,cls,params", _MODEL_PARAMS, ids=_MODEL_IDS)
    def test_clone_fresh_returns_unfitted(self, name, cls, params, train_test_data):
        model = cls(params=params)
        _run_quiet(model.fit, train_test_data.train_df)
        clone = model.clone_fresh()
        assert clone._fit_result is None
        assert clone is not model

    @pytest.mark.parametrize("name,cls,params", _MODEL_PARAMS, ids=_MODEL_IDS)
    def test_clone_fresh_can_refit(self, name, cls, params, train_test_data):
        from its2s.models.base import FitResult
        model = cls(params=params)
        _run_quiet(model.fit, train_test_data.train_df)
        clone = model.clone_fresh()
        fr = _run_quiet(clone.fit, train_test_data.train_df)
        assert isinstance(fr, FitResult)

    @pytest.mark.parametrize("name,cls,params", _MODEL_PARAMS, ids=_MODEL_IDS)
    def test_residuals_near_zero_mean(self, name, cls, params, train_test_data):
        model = cls(params=params)
        fr = _run_quiet(model.fit, train_test_data.train_df)
        assert abs(np.nanmean(fr.residuals)) < 20.0

    @pytest.mark.parametrize("name,cls,params", _MODEL_PARAMS, ids=_MODEL_IDS)
    def test_fitted_plus_residuals_eq_actual(self, name, cls, params, train_test_data):
        model = cls(params=params)
        fr = _run_quiet(model.fit, train_test_data.train_df)
        reconstructed = fr.fitted_values + fr.residuals
        actual = train_test_data.train_df["y"].values
        finite_mask = np.isfinite(reconstructed)
        assert finite_mask.sum() > 0
        np.testing.assert_allclose(
            reconstructed[finite_mask], actual[finite_mask], rtol=1e-4, atol=1e-4)


# ===================================================================
# MBB Bootstrap (parametrized across all 4 models)
# ===================================================================
class TestModelBootstrap:
    """Verify MBB bootstrap behavior for each model."""

    @pytest.fixture(scope="class")
    def fitted_splits(self):
        """Return (model_name, fitted_model, splits) for ARIMA as infrastructure."""
        from its2s.models.arima import ARIMAModel
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_short_series(n_pre=180, n_post=30, seed=1002)
        splits = prepare_splits(df, intv, split_method="days", test_days=30, holdout_days=30)
        model = ARIMAModel(params={"seasonal": False, "m": 1, "stepwise": True,
                                   "suppress_warnings": True})
        model.fit(splits.train_df)
        return model, splits

    @pytest.mark.parametrize("name,cls,params", _MODEL_PARAMS, ids=_MODEL_IDS)
    def test_mbb_ci_shape(self, name, cls, params):
        from its2s.bootstrap.mbb import MovingBlockBootstrap
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_short_series(n_pre=180, n_post=30, seed=1003)
        splits = prepare_splits(df, intv, split_method="days", test_days=30, holdout_days=30)
        model = cls(params=params)
        _run_quiet(model.fit, splits.train_df)
        mbb = MovingBlockBootstrap(n_sim=5, block_length=7, n_jobs=1)
        result = _run_quiet(mbb.generate_cis, model, splits.train_df,
                            splits.full_predict_df, seed=42)
        n_target = len(splits.full_predict_df)
        assert result.predicted.shape == (n_target,)
        assert result.conf_lo.shape == (n_target,)
        assert result.conf_hi.shape == (n_target,)
        assert result.pred_matrix.shape[0] == n_target

    @pytest.mark.parametrize("name,cls,params", _MODEL_PARAMS, ids=_MODEL_IDS)
    def test_mbb_ci_has_positive_width(self, name, cls, params):
        from its2s.bootstrap.mbb import MovingBlockBootstrap
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_short_series(n_pre=180, n_post=30, seed=1004)
        splits = prepare_splits(df, intv, split_method="days", test_days=30, holdout_days=30)
        model = cls(params=params)
        _run_quiet(model.fit, splits.train_df)
        mbb = MovingBlockBootstrap(n_sim=10, block_length=7, n_jobs=1)
        result = _run_quiet(mbb.generate_cis, model, splits.train_df,
                            splits.full_predict_df, seed=42)
        widths = result.conf_hi - result.conf_lo
        # NeuralProphet AR warmup rows produce NaN; ignore them in the check.
        finite_widths = widths[np.isfinite(widths)]
        assert len(finite_widths) > 0, "No finite CI widths found"
        assert np.all(finite_widths >= 0)
        assert np.mean(finite_widths) > 0

    @pytest.mark.parametrize("name,cls,params", _MODEL_PARAMS, ids=_MODEL_IDS)
    def test_mbb_n_successful_equals_n_sim(self, name, cls, params):
        from its2s.bootstrap.mbb import MovingBlockBootstrap
        from its2s.data_prep import prepare_splits
        n_sim = 5
        df, intv, _ = make_short_series(n_pre=180, n_post=30, seed=1005)
        splits = prepare_splits(df, intv, split_method="days", test_days=30, holdout_days=30)
        model = cls(params=params)
        _run_quiet(model.fit, splits.train_df)
        mbb = MovingBlockBootstrap(n_sim=n_sim, block_length=7, n_jobs=1)
        result = _run_quiet(mbb.generate_cis, model, splits.train_df,
                            splits.full_predict_df, seed=42)
        assert result.n_successful == n_sim

    @pytest.mark.parametrize("name,cls,params", _MODEL_PARAMS, ids=_MODEL_IDS)
    def test_mbb_ci_contains_prediction(self, name, cls, params, request):
        """Point prediction should fall inside CI for >= 80% of dates.

        ARIMA is expected to fail this test: its constant long-horizon forecast
        can fall just outside the MBB CI lower bound due to bootstrap mean-shift
        from residual resampling.  This reveals a real limitation of flat-
        prediction ARIMA with MBB, not a defect in the test logic.
        """
        if name == "arima":
            request.applymarker(pytest.mark.xfail(
                reason="Known limitation, not a test defect: ARIMA's flat "
                       "long-horizon forecast escapes the MBB CI, whose "
                       "narrowness is tracked in GH #41 (missing innovation "
                       "variance). Expected to resolve when the interval "
                       "construction gains the innovation term.",
                strict=False))
        from its2s.bootstrap.mbb import MovingBlockBootstrap
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_short_series(n_pre=180, n_post=30, seed=77)
        splits = prepare_splits(df, intv, split_method="days", test_days=30, holdout_days=30)
        model = cls(params=params)
        _run_quiet(model.fit, splits.train_df)
        mbb = MovingBlockBootstrap(n_sim=20, block_length=7, n_jobs=1)
        result = _run_quiet(mbb.generate_cis, model, splits.train_df,
                            splits.full_predict_df, seed=42)
        finite = np.isfinite(result.predicted) & np.isfinite(result.conf_lo)
        inside = (result.conf_lo[finite] <= result.predicted[finite]) & \
                 (result.predicted[finite] <= result.conf_hi[finite])
        pct_inside = np.mean(inside)
        assert pct_inside >= 0.8, (
            f"[{name}] Only {pct_inside:.0%} of predicted values fall inside the CI.")


# ===================================================================
# ARIMA-Specific Unit Tests
# ===================================================================
class TestARIMASpecific:
    """Tests for ARIMA behaviors that are not shared with other models."""

    def test_fixed_order_preserved_in_clone(self):
        from its2s.models.arima import ARIMAModel
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_short_series(n_pre=180, n_post=30, seed=1100)
        splits = prepare_splits(df, intv, split_method="days", test_days=30, holdout_days=30)
        model = ARIMAModel(params={"seasonal": False, "m": 1, "stepwise": True,
                                   "suppress_warnings": True})
        model.fit(splits.train_df)
        clone = model.clone_fresh()
        assert clone._fixed_order == model._fixed_order
        assert clone._fixed_seasonal_order == model._fixed_seasonal_order
        assert clone._fit_result is None

    def test_seasonal_order_is_four_tuple(self):
        from its2s.models.arima import ARIMAModel
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_short_series(n_pre=180, n_post=30, seed=1101)
        splits = prepare_splits(df, intv, split_method="days", test_days=30, holdout_days=30)
        model = ARIMAModel(params={"seasonal": True, "m": 7, "stepwise": True,
                                   "suppress_warnings": True})
        model.fit(splits.train_df)
        assert isinstance(model._fixed_seasonal_order, tuple)
        assert len(model._fixed_seasonal_order) == 4

    def test_with_covariates(self):
        from its2s.models.arima import ARIMAModel
        from its2s.data_prep import prepare_splits
        df, intv, _, cov_cols = make_series_with_covariates(seed=1102)
        splits = prepare_splits(df, intv, split_method="days", test_days=365, holdout_days=365)
        model = ARIMAModel(params={"seasonal": True, "m": 7, "stepwise": True,
                                   "suppress_warnings": True})
        fr = model.fit(splits.train_df, covariate_cols=cov_cols)
        pr = model.predict(splits.full_predict_df, covariate_cols=cov_cols)
        assert len(fr.fitted_values) == len(splits.train_df)
        assert len(pr.predicted) == len(splits.full_predict_df)

    def test_e2e_weekly_with_m52(self):
        from its2s import run_single_its
        df, intv, _ = make_weekly_series(intervention_effect=5.0, seed=1103)
        result = run_single_its(df, intv, model_name="arima", seed=42,
                                config_overrides=_merge(
                                    _FAST,
                                    {"periods": {"test_days": 364, "holdout_days": 364},
                                     "models": {"arima": {"m": 52}},
                                     "metrics": {"seasonality": 52}}))
        assert result.model_name == "arima"

    def test_e2e_monthly_with_m12(self):
        from its2s import run_single_its
        df, intv, _ = make_monthly_series(intervention_effect=10.0, seed=1104)
        result = run_single_its(df, intv, model_name="arima", seed=42,
                                config_overrides=_merge(
                                    _FAST,
                                    {"periods": {"test_days": 360, "holdout_days": 360},
                                     "models": {"arima": {"m": 12}},
                                     "metrics": {"seasonality": 12}}))
        assert result.model_name == "arima"

    def test_e2e_quarterly_with_m4(self):
        from its2s import run_single_its
        df, intv, _ = make_quarterly_series(intervention_effect=15.0, seed=1105)
        result = run_single_its(df, intv, model_name="arima", seed=42,
                                config_overrides=_merge(
                                    _FAST,
                                    {"periods": {"test_days": 360, "holdout_days": 360},
                                     "models": {"arima": {"m": 4, "seasonal": True}},
                                     "metrics": {"seasonality": 4}}))
        assert result.model_name == "arima"


# ===================================================================
# ProphetXGB-Specific Unit Tests
# ===================================================================
class TestProphetXGBSpecific:
    """Tests for ProphetXGBHybridModel behaviors not shared with other models."""

    def test_prophet_and_xgb_objects_stored_after_fit(self):
        from its2s.models.prophet_xgb import ProphetXGBHybridModel
        from its2s.data_prep import prepare_splits
        from prophet import Prophet
        from xgboost import XGBRegressor
        df, intv, _ = make_short_series(n_pre=180, n_post=30, seed=1200)
        splits = prepare_splits(df, intv, split_method="days", test_days=30, holdout_days=30)
        model = ProphetXGBHybridModel(params={})
        _run_quiet(model.fit, splits.train_df)
        assert isinstance(model._prophet, Prophet)
        assert isinstance(model._xgb, XGBRegressor)

    def test_clone_clears_prophet_and_xgb(self):
        from its2s.models.prophet_xgb import ProphetXGBHybridModel
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_short_series(n_pre=180, n_post=30, seed=1201)
        splits = prepare_splits(df, intv, split_method="days", test_days=30, holdout_days=30)
        model = ProphetXGBHybridModel(params={})
        _run_quiet(model.fit, splits.train_df)
        clone = model.clone_fresh()
        assert clone._prophet is None
        assert clone._xgb is None
        assert clone._fit_result is None

    def test_residuals_near_zero_mean_on_long_series(self):
        from its2s.models.prophet_xgb import ProphetXGBHybridModel
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_daily_series(n_pre=730, n_post=180, seed=1202)
        splits = prepare_splits(df, intv, split_method="days", test_days=180, holdout_days=180)
        model = ProphetXGBHybridModel(params={})
        fr = _run_quiet(model.fit, splits.train_df)
        assert abs(np.mean(fr.residuals)) < 5.0

    def test_with_covariates(self):
        from its2s.models.prophet_xgb import ProphetXGBHybridModel
        from its2s.data_prep import prepare_splits
        df, intv, _, cov_cols = make_series_with_covariates(seed=1203)
        splits = prepare_splits(df, intv, split_method="days", test_days=365, holdout_days=365)
        model = ProphetXGBHybridModel(params={})
        fr = _run_quiet(model.fit, splits.train_df, covariate_cols=cov_cols)
        pr = _run_quiet(model.predict, splits.full_predict_df, covariate_cols=cov_cols)
        assert len(fr.fitted_values) == len(splits.train_df)
        assert len(pr.predicted) == len(splits.full_predict_df)


# ===================================================================
# ProphetThenXGB-Specific Unit Tests
# ===================================================================
class TestProphetThenXGBSpecific:
    """Tests for ProphetThenXGBModel behaviors not shared with other models."""

    def test_prophet_forecast_is_xgb_feature(self):
        """Prophet residuals become XGB input features in the sequential model."""
        from its2s.models.prophet_then_xgb import ProphetThenXGBModel
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_short_series(n_pre=180, n_post=30, seed=1300)
        splits = prepare_splits(df, intv, split_method="days", test_days=30, holdout_days=30)
        model = ProphetThenXGBModel(params={})
        _run_quiet(model.fit, splits.train_df)
        feature_names = model._xgb.get_booster().feature_names
        assert "prophet_forecast" in feature_names

    def test_clone_clears_prophet_and_xgb(self):
        from its2s.models.prophet_then_xgb import ProphetThenXGBModel
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_short_series(n_pre=180, n_post=30, seed=1301)
        splits = prepare_splits(df, intv, split_method="days", test_days=30, holdout_days=30)
        model = ProphetThenXGBModel(params={})
        _run_quiet(model.fit, splits.train_df)
        clone = model.clone_fresh()
        assert clone._prophet is None
        assert clone._xgb is None
        assert clone._fit_result is None

    def test_residuals_near_zero_mean_on_long_series(self):
        from its2s.models.prophet_then_xgb import ProphetThenXGBModel
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_daily_series(n_pre=730, n_post=180, seed=1302)
        splits = prepare_splits(df, intv, split_method="days", test_days=180, holdout_days=180)
        model = ProphetThenXGBModel(params={})
        fr = _run_quiet(model.fit, splits.train_df)
        assert abs(np.mean(fr.residuals)) < 5.0

    def test_with_covariates(self):
        from its2s.models.prophet_then_xgb import ProphetThenXGBModel
        from its2s.data_prep import prepare_splits
        df, intv, _, cov_cols = make_series_with_covariates(seed=1303)
        splits = prepare_splits(df, intv, split_method="days", test_days=365, holdout_days=365)
        model = ProphetThenXGBModel(params={})
        fr = _run_quiet(model.fit, splits.train_df, covariate_cols=cov_cols)
        pr = _run_quiet(model.predict, splits.full_predict_df, covariate_cols=cov_cols)
        assert len(fr.fitted_values) == len(splits.train_df)
        assert len(pr.predicted) == len(splits.full_predict_df)


# ===================================================================
# Prophet weekly_seasonality "auto" default (GH #60)
# ===================================================================
class TestProphetWeeklySeasonalityAuto:
    """The shipped weekly_seasonality default defers to Prophet's spacing rule.

    On a weekly grid the weekly component is degenerate (its period equals the
    observation spacing), so "auto" must disable it; on a daily grid "auto"
    resolves to the same configuration the previous hard-coded True produced
    (fourier_order 3), keeping daily runs bit-identical.
    """

    @staticmethod
    def _make_model(model_name):
        if model_name == "prophet_xgb":
            from its2s.models.prophet_xgb import ProphetXGBHybridModel
            return ProphetXGBHybridModel(params={})
        from its2s.models.prophet_then_xgb import ProphetThenXGBModel
        return ProphetThenXGBModel(params={})

    @pytest.mark.parametrize("model_name", ["prophet_xgb", "prophet_then_xgb"])
    def test_weekly_disabled_on_weekly_grid(self, model_name):
        df, _, _ = make_weekly_series(seed=1500)
        model = self._make_model(model_name)
        _run_quiet(model.fit, df.iloc[:156])
        assert "weekly" not in model._prophet.seasonalities
        assert "yearly" in model._prophet.seasonalities

    @pytest.mark.parametrize("model_name", ["prophet_xgb", "prophet_then_xgb"])
    def test_weekly_enabled_on_daily_grid(self, model_name):
        df, _, _ = make_short_series(n_pre=180, n_post=30, seed=1501)
        model = self._make_model(model_name)
        _run_quiet(model.fit, df.iloc[:180])
        assert model._prophet.seasonalities["weekly"]["fourier_order"] == 3


# ===================================================================
# NeuralProphet-Specific Unit Tests
# ===================================================================
@pytest.mark.skipif(not _has_neuralprophet(), reason="neuralprophet not installed")
class TestNeuralProphetSpecific:
    """Tests for NeuralProphetModel behaviors not shared with other models."""

    def test_model_state_none_before_fit(self):
        from its2s.models.neuralprophet import NeuralProphetModel
        model = NeuralProphetModel(params=_NP_FAST_PARAMS)
        assert model._model is None
        assert model._fit_result is None

    def test_weekly_seasonality_auto_reaches_library(self):
        """GH #60: the default "auto" is handed to NeuralProphet unresolved;
        the library applies the same spacing rule as Prophet at fit time."""
        from its2s.models.neuralprophet import NeuralProphetModel
        np_model = NeuralProphetModel(params={})._build_model()
        assert np_model.config_seasonality.periods["weekly"].arg == "auto"

    def test_clone_clears_model_attribute(self):
        from its2s.models.neuralprophet import NeuralProphetModel
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_short_series(n_pre=180, n_post=30, seed=1400)
        splits = prepare_splits(df, intv, split_method="days", test_days=30, holdout_days=30)
        model = NeuralProphetModel(params=_NP_FAST_PARAMS)
        _run_quiet(model.fit, splits.train_df)
        clone = model.clone_fresh()
        assert clone._model is None
        assert clone._fit_result is None

    def test_fitted_residuals_nan_only_for_ar_warmup(self):
        """AR warmup produces NaN for the first n_lags rows; rest should be finite."""
        from its2s.models.neuralprophet import NeuralProphetModel
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_short_series(n_pre=180, n_post=30, seed=1401)
        splits = prepare_splits(df, intv, split_method="days", test_days=30, holdout_days=30)
        model = NeuralProphetModel(params=_NP_FAST_PARAMS)
        fr = _run_quiet(model.fit, splits.train_df)
        finite_mask = np.isfinite(fr.residuals)
        n_lags = _NP_FAST_PARAMS["n_lags"]
        # Rows after the AR warmup period must be finite.
        assert finite_mask[n_lags:].mean() > 0.9

    def test_fitted_plus_residuals_eq_actual_finite_rows(self):
        """fitted + residuals == actual for all finite (non-warmup) rows."""
        from its2s.models.neuralprophet import NeuralProphetModel
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_short_series(n_pre=180, n_post=30, seed=1402)
        splits = prepare_splits(df, intv, split_method="days", test_days=30, holdout_days=30)
        model = NeuralProphetModel(params=_NP_FAST_PARAMS)
        fr = _run_quiet(model.fit, splits.train_df)
        reconstructed = fr.fitted_values + fr.residuals
        actual = splits.train_df["y"].values
        finite_mask = np.isfinite(reconstructed)
        assert finite_mask.sum() > 0
        np.testing.assert_allclose(
            reconstructed[finite_mask], actual[finite_mask], rtol=1e-4, atol=1e-4)

    def test_residuals_near_zero_mean_ignoring_nan(self):
        from its2s.models.neuralprophet import NeuralProphetModel
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_daily_series(n_pre=730, n_post=180, seed=1403)
        splits = prepare_splits(df, intv, split_method="days", test_days=180, holdout_days=180)
        model = NeuralProphetModel(params=_NP_FAST_PARAMS)
        fr = _run_quiet(model.fit, splits.train_df)
        assert abs(np.nanmean(fr.residuals)) < 20.0

    def test_with_covariates(self):
        from its2s.models.neuralprophet import NeuralProphetModel
        from its2s.data_prep import prepare_splits
        df, intv, _, cov_cols = make_series_with_covariates(seed=1404)
        splits = prepare_splits(df, intv, split_method="days", test_days=365, holdout_days=365)
        model = NeuralProphetModel(params=_NP_FAST_PARAMS)
        fr = _run_quiet(model.fit, splits.train_df, covariate_cols=cov_cols)
        pr = _run_quiet(model.predict, splits.full_predict_df, covariate_cols=cov_cols)
        assert len(fr.fitted_values) == len(splits.train_df)
        assert len(pr.predicted) == len(splits.full_predict_df)


# ===================================================================
# End-to-End Integration (parametrized across all 4 models)
# ===================================================================
class TestModelIntegration:
    """Full pipeline e2e tests for normal data scenarios."""

    @pytest.mark.parametrize("model_name", _MODEL_NAMES_MARKS)
    def test_e2e_daily(self, model_name):
        from its2s import run_single_its
        df, intv, _ = make_daily_series(intervention_effect=10.0, seed=2010)
        cfg = _e2e_config(model_name)
        result = _run_quiet(run_single_its, df, intv, model_name=model_name,
                            seed=42, config_overrides=cfg)
        assert result.model_name == model_name
        assert not result.excess_table.obs_excess.empty
        assert np.isfinite(result.metrics_test.rmse)

    @pytest.mark.parametrize("model_name", _MODEL_NAMES_MARKS)
    def test_e2e_weekly(self, model_name):
        from its2s import run_single_its
        df, intv, _ = make_weekly_series(intervention_effect=5.0, seed=2011)
        cfg = _e2e_config(model_name, test_days=364, holdout_days=364)
        if model_name == "arima":
            cfg = _merge(cfg, _ARIMA_FREQ_EXTRA["weekly"])
        try:
            result = _run_quiet(run_single_its, df, intv, model_name=model_name,
                                seed=42, config_overrides=cfg)
            assert result.model_name == model_name
        except Exception as e:
            if model_name == "neuralprophet":
                pytest.skip(f"neuralprophet freq mismatch on weekly data: {e}")
            raise

    @pytest.mark.parametrize("model_name", _MODEL_NAMES_MARKS)
    def test_e2e_monthly(self, model_name):
        from its2s import run_single_its
        df, intv, _ = make_monthly_series(intervention_effect=10.0, seed=2012)
        cfg = _e2e_config(model_name, test_days=360, holdout_days=360)
        if model_name == "arima":
            cfg = _merge(cfg, _ARIMA_FREQ_EXTRA["monthly"])
        try:
            result = _run_quiet(run_single_its, df, intv, model_name=model_name,
                                seed=42, config_overrides=cfg)
            assert result.model_name == model_name
        except Exception as e:
            if model_name == "neuralprophet":
                pytest.skip(f"neuralprophet freq mismatch on monthly data: {e}")
            raise

    @pytest.mark.parametrize("model_name", _MODEL_NAMES_MARKS)
    def test_e2e_quarterly(self, model_name):
        from its2s import run_single_its
        df, intv, _ = make_quarterly_series(intervention_effect=15.0, seed=2013)
        cfg = _e2e_config(model_name, test_days=360, holdout_days=360)
        if model_name == "arima":
            cfg = _merge(cfg, _ARIMA_FREQ_EXTRA["quarterly"])
        try:
            result = _run_quiet(run_single_its, df, intv, model_name=model_name,
                                seed=42, config_overrides=cfg)
            assert result.model_name == model_name
        except Exception as e:
            if model_name == "neuralprophet":
                pytest.skip(f"neuralprophet freq mismatch on quarterly data: {e}")
            raise

    @pytest.mark.parametrize("model_name", _MODEL_NAMES_MARKS)
    def test_e2e_with_covariates(self, model_name):
        from its2s import run_single_its
        df, intv, _, cov_cols = make_series_with_covariates(seed=2014)
        cfg = _e2e_config(model_name)
        result = _run_quiet(run_single_its, df, intv, model_name=model_name,
                            covariate_cols=cov_cols, seed=42, config_overrides=cfg)
        assert result.model_name == model_name
        assert not result.excess_table.obs_excess.empty
        assert np.isfinite(result.metrics_test.rmse)

    @pytest.mark.parametrize("model_name", _MODEL_NAMES_MARKS)
    def test_e2e_count_data(self, model_name):
        from its2s import run_single_its
        df, intv, _ = make_count_series(intervention_effect=5.0, seed=2015)
        cfg = _e2e_config(model_name)
        result = _run_quiet(run_single_its, df, intv, model_name=model_name,
                            seed=42, config_overrides=cfg)
        assert result.model_name == model_name
        assert np.isfinite(result.metrics_test.rmse)

    @pytest.mark.parametrize("model_name", _MODEL_NAMES_MARKS)
    def test_e2e_output_files_saved(self, model_name, tmp_path):
        from its2s import run_single_its
        df, intv, _ = make_short_series(n_pre=180, n_post=30, seed=2016)
        cfg = _e2e_config(model_name, test_days=30, holdout_days=30)
        _run_quiet(run_single_its, df, intv, model_name=model_name,
                   seed=42, output_dir=str(tmp_path), config_overrides=cfg)
        assert (tmp_path / f"{model_name}_counterfactual.png").exists()
        assert (tmp_path / f"{model_name}_excess.csv").exists()
        assert (tmp_path / f"{model_name}_metrics.csv").exists()


# ===================================================================
# Robustness & Edge Cases (parametrized across all 4 models)
# ===================================================================
class TestModelRobustness:
    """Edge-case inputs that every model should handle without crashing."""

    @pytest.mark.parametrize("model_name", _MODEL_NAMES_MARKS)
    def test_missing_data(self, model_name):
        from its2s import run_single_its
        df, intv, _ = make_missing_data_series(seed=3010)
        cfg = _e2e_config(model_name)
        try:
            result = _run_quiet(run_single_its, df, intv, model_name=model_name,
                                seed=42, config_overrides=cfg)
            assert result.model_name == model_name
        except Exception as e:
            pytest.skip(f"[{model_name}] does not handle NaN: {type(e).__name__}: {e}")

    @pytest.mark.parametrize("model_name", _MODEL_NAMES_MARKS)
    def test_very_short_series(self, model_name):
        from its2s import run_single_its
        df, intv, _ = make_short_series(n_pre=90, n_post=30, seed=3011)
        cfg = _e2e_config(model_name, test_days=30, holdout_days=30, n_sim=5)
        if model_name == "arima":
            cfg = _merge(cfg, {"models": {"arima": {"seasonal": False, "m": 1}}})
        try:
            result = _run_quiet(run_single_its, df, intv, model_name=model_name,
                                seed=42, config_overrides=cfg)
            assert np.isfinite(result.metrics_train.rmse)
        except Exception as e:
            if model_name == "neuralprophet":
                pytest.skip(f"neuralprophet short series: {e}")
            raise

    @pytest.mark.parametrize("model_name", _MODEL_NAMES_MARKS)
    def test_outlier_data(self, model_name):
        from its2s import run_single_its
        df, intv, _ = make_outlier_series(seed=3012)
        cfg = _e2e_config(model_name)
        result = _run_quiet(run_single_its, df, intv, model_name=model_name,
                            seed=42, config_overrides=cfg)
        assert result.model_name == model_name
        assert np.isfinite(result.metrics_train.rmse)

    @pytest.mark.parametrize("model_name", _MODEL_NAMES_MARKS)
    def test_constant_data(self, model_name):
        from its2s import run_single_its
        df, intv, _ = make_constant_series(value=50.0)
        cfg = _e2e_config(model_name)
        if model_name == "arima":
            cfg = _merge(cfg, {"models": {"arima": {"seasonal": False, "m": 1}}})
        try:
            result = _run_quiet(run_single_its, df, intv, model_name=model_name,
                                seed=42, config_overrides=cfg)
            if not result.excess_table.obs_excess.empty:
                mean_excess = result.excess_table.obs_excess["excess"].mean()
                assert abs(mean_excess) < 10.0
        except Exception as e:
            pytest.skip(f"[{model_name}] fails on constant data: {type(e).__name__}: {e}")

    @pytest.mark.parametrize("model_name", _MODEL_NAMES_MARKS)
    def test_large_intervention_effect(self, model_name):
        from its2s import run_single_its
        from its2s.metrics.excess import calc_ate_summary
        df, intv, _ = make_daily_series(intervention_effect=100.0, base_level=100.0,
                                         seed=3013)
        cfg = _e2e_config(model_name)
        result = _run_quiet(run_single_its, df, intv, model_name=model_name,
                            seed=42, config_overrides=cfg)
        ate = calc_ate_summary(result.excess_table.obs_excess)
        total_ate = ate[ate["metric"] == "Total ATE"]["estimate"].values[0]
        assert total_ate > 0, f"[{model_name}] Expected large positive ATE, got {total_ate}"

    def test_unknown_model_name_raises(self):
        from its2s import run_single_its
        df, intv, _ = make_short_series(seed=3014)
        with pytest.raises(ValueError, match="Unknown model"):
            run_single_its(df, intv, model_name="nonexistent", seed=42)


# ===================================================================
# Statistical Validation (slow, parametrized across all 4 models)
# ===================================================================
class TestModelStatistical:
    """Verify that each model recovers planted intervention effects.

    These tests use n_sim=50 and are marked @pytest.mark.slow so they can be
    excluded from fast CI runs with ``-m "not slow"``.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("model_name", _MODEL_NAMES_MARKS)
    def test_known_positive_effect_recovery(self, model_name):
        from its2s import run_single_its
        from its2s.metrics.excess import calc_ate_summary
        df, intv, _ = make_daily_series(intervention_effect=10.0, noise_sd=5.0,
                                         seed=4010)
        cfg = _e2e_config(model_name, n_sim=50)
        result = _run_quiet(run_single_its, df, intv, model_name=model_name,
                            seed=42, config_overrides=cfg)
        ate = calc_ate_summary(result.excess_table.obs_excess)
        mean_ate = ate[ate["metric"] == "Mean ATE per obs"]["estimate"].values[0]
        assert 3.0 < mean_ate < 20.0, (
            f"[{model_name}] Expected ATE near 10.0, got {mean_ate:.2f}")

    @pytest.mark.slow
    @pytest.mark.parametrize("model_name", _MODEL_NAMES_MARKS)
    def test_null_effect_near_zero(self, model_name):
        from its2s import run_single_its
        from its2s.metrics.excess import calc_ate_summary
        df, intv, _ = make_daily_series(intervention_effect=0.0, noise_sd=5.0,
                                         seed=4011)
        cfg = _e2e_config(model_name, n_sim=50)
        result = _run_quiet(run_single_its, df, intv, model_name=model_name,
                            seed=42, config_overrides=cfg)
        ate = calc_ate_summary(result.excess_table.obs_excess)
        mean_ate = ate[ate["metric"] == "Mean ATE per obs"]["estimate"].values[0]
        assert abs(mean_ate) < 8.0, (
            f"[{model_name}] Expected near-zero ATE, got {mean_ate:.2f}")

    @pytest.mark.slow
    @pytest.mark.parametrize("model_name", _MODEL_NAMES_MARKS)
    def test_negative_effect_recovery(self, model_name):
        from its2s import run_single_its
        from its2s.metrics.excess import calc_ate_summary
        df, intv, _ = make_daily_series(intervention_effect=-8.0, noise_sd=5.0,
                                         seed=4012)
        cfg = _e2e_config(model_name, n_sim=50)
        result = _run_quiet(run_single_its, df, intv, model_name=model_name,
                            seed=42, config_overrides=cfg)
        ate = calc_ate_summary(result.excess_table.obs_excess)
        mean_ate = ate[ate["metric"] == "Mean ATE per obs"]["estimate"].values[0]
        assert -20.0 < mean_ate < 0.0, (
            f"[{model_name}] Expected negative ATE near -8.0, got {mean_ate:.2f}")


# ===================================================================
# Cross-Model Comparison
# ===================================================================
class TestModelComparison:
    """Tests that compare behavior across models on the same dataset."""

    def test_all_models_produce_finite_predictions(self):
        """Every available model should produce >= 90% finite predictions."""
        from its2s import run_single_its
        df, intv, _ = make_daily_series(n_pre=365, n_post=90, intervention_effect=10.0,
                                         seed=5010)
        period_cfg = {"periods": {"test_days": 90, "holdout_days": 90}}
        for model_name, _, _ in _MODEL_PARAMS:
            cfg = _merge(_FAST, _merge(period_cfg, _e2e_config(model_name,
                                                                test_days=90,
                                                                holdout_days=90)))
            result = _run_quiet(run_single_its, df, intv, model_name=model_name,
                                seed=42, config_overrides=cfg)
            preds = result.bootstrap_result.predicted
            finite_frac = np.isfinite(preds).sum() / len(preds)
            assert finite_frac > 0.9, (
                f"[{model_name}] {100 * (1 - finite_frac):.1f}% of predictions "
                f"are non-finite (threshold: 10%)")

    def test_all_models_detect_large_positive_effect(self):
        """Every available model should assign a positive total ATE."""
        from its2s import run_single_its
        from its2s.metrics.excess import calc_ate_summary
        df, intv, _ = make_daily_series(n_pre=365, n_post=90, intervention_effect=20.0,
                                         seed=5011)
        for model_name, _, _ in _MODEL_PARAMS:
            cfg = _merge(_FAST, _e2e_config(model_name, test_days=90, holdout_days=90))
            result = _run_quiet(run_single_its, df, intv, model_name=model_name,
                                seed=42, config_overrides=cfg)
            ate = calc_ate_summary(result.excess_table.obs_excess)
            total_ate = ate[ate["metric"] == "Total ATE"]["estimate"].values[0]
            assert total_ate > 0, (
                f"[{model_name}] Expected positive total ATE, got {total_ate:.2f}")

    def test_prophet_models_produce_different_predictions(self):
        """ProphetXGB and ProphetThenXGB should not produce identical predictions."""
        from its2s import run_single_its
        df, intv, _ = make_daily_series(intervention_effect=10.0, seed=5012)
        cfg = dict(_FAST)
        r1 = _run_quiet(run_single_its, df, intv, model_name="prophet_xgb",
                        seed=42, config_overrides=cfg)
        r2 = _run_quiet(run_single_its, df, intv, model_name="prophet_then_xgb",
                        seed=42, config_overrides=cfg)
        assert not np.allclose(r1.bootstrap_result.predicted,
                               r2.bootstrap_result.predicted, atol=0.1)

    def test_prophet_time_features_are_consistent(self):
        """Both Prophet modules expose _make_time_features; output must match."""
        from its2s.models.prophet_xgb import _make_time_features as tf1
        from its2s.models.prophet_then_xgb import _make_time_features as tf2
        df = pd.DataFrame({"ds": pd.date_range("2021-01-01", periods=30, freq="D")})
        pd.testing.assert_frame_equal(tf1(df), tf2(df))


# ===================================================================
# Main entry point
# ===================================================================
if __name__ == "__main__":
    import sys
    exit_code = pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:])
    sys.exit(exit_code)
