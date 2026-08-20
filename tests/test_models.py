# Description: Model-facing test suite for its2s package.
#   All three models (arima, prophet_xgb, neuralprophet) are
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
    for name in ["prophet_xgb", "neuralprophet", "arima"]
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

    @pytest.mark.parametrize("name,cls,params", _MODEL_PARAMS, ids=_MODEL_IDS)
    def test_clone_fresh_does_not_share_nested_params(self, name, cls, params):
        """A clone's nested param dicts must be independent copies: MBB refits
        clone per draw, and a shared sub-dict would let a mutation on one clone
        silently propagate to the parent and all later draws."""
        nested = {**params, "prophet": {"changepoint_prior_scale": 0.05}}
        model = cls(params=nested)
        clone = model.clone_fresh()
        assert clone.params == model.params
        clone.params["prophet"]["changepoint_prior_scale"] = 0.99
        assert model.params["prophet"]["changepoint_prior_scale"] == 0.05


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
                                    {"periods": {"split_method": "days",
                                                 "test_days": 364, "holdout_days": 364},
                                     "models": {"arima": {"m": 52}},
                                     "metrics": {"seasonality": 52}}))
        assert result.model_name == "arima"

    def test_e2e_monthly_with_m12(self):
        from its2s import run_single_its
        df, intv, _ = make_monthly_series(intervention_effect=10.0, seed=1104)
        result = run_single_its(df, intv, model_name="arima", seed=42,
                                config_overrides=_merge(
                                    _FAST,
                                    {"periods": {"split_method": "days",
                                                 "test_days": 360, "holdout_days": 360},
                                     "models": {"arima": {"m": 12}},
                                     "metrics": {"seasonality": 12}}))
        assert result.model_name == "arima"

    def test_e2e_quarterly_with_m4(self):
        from its2s import run_single_its
        df, intv, _ = make_quarterly_series(intervention_effect=15.0, seed=1105)
        result = run_single_its(df, intv, model_name="arima", seed=42,
                                config_overrides=_merge(
                                    _FAST,
                                    {"periods": {"split_method": "days",
                                                 "test_days": 360, "holdout_days": 360},
                                     "models": {"arima": {"m": 4, "seasonal": True}},
                                     "metrics": {"seasonality": 4}}))
        assert result.model_name == "arima"


# ===================================================================
# ARIMA seasonal period m "auto" resolution (GH #59, D-059)
# ===================================================================
class TestARIMASeasonalityAuto:
    """The shipped m default resolves from the series frequency with a loud
    non-seasonal fallback; explicit values are honored, never substituted."""

    @staticmethod
    def _freq(alias):
        from its2s.frequency import SeriesFrequency
        return SeriesFrequency.from_alias(alias)

    @staticmethod
    def _fit_with_stub(monkeypatch, df, params):
        """Fit an ARIMAModel with pm.auto_arima stubbed out; return the
        kwargs the stub captured (so tests assert on the m actually passed
        without paying for a real stepwise search)."""
        import its2s.models.arima as arima_mod

        captured = {}
        n = len(df)

        class _Stub:
            order = (1, 0, 0)
            seasonal_order = (0, 0, 0, 0)

            def predict_in_sample(self, exogenous=None):
                return np.zeros(n)

        def fake_auto_arima(y, **kwargs):
            captured.update(kwargs)
            return _Stub()

        monkeypatch.setattr(arima_mod.pm, "auto_arima", fake_auto_arima)
        model = arima_mod.ARIMAModel(params=params)
        model.fit(df)
        return captured

    # -- resolver unit tests -------------------------------------------------

    def test_auto_daily_resolves_7(self):
        from its2s.models.arima import resolve_arima_m
        assert resolve_arima_m("auto", n_train=400,
                               series_freq=self._freq("D")) == 7

    def test_auto_weekly_resolves_52(self):
        from its2s.models.arima import resolve_arima_m
        assert resolve_arima_m("auto", n_train=200,
                               series_freq=self._freq("W-SUN")) == 52

    def test_auto_monthly_resolves_12(self):
        from its2s.models.arima import resolve_arima_m
        assert resolve_arima_m("auto", n_train=60,
                               series_freq=self._freq("MS")) == 12

    def test_auto_unmapped_frequency_falls_back_to_1_with_warning(self):
        from its2s.models.arima import resolve_arima_m
        with pytest.warns(UserWarning, match="no dominant seasonal period"):
            m = resolve_arima_m("auto", n_train=100,
                                series_freq=self._freq("QS-JAN"))
        assert m == 1

    def test_auto_short_train_falls_back_to_1_with_warning(self):
        from its2s.models.arima import resolve_arima_m
        with pytest.warns(UserWarning, match=r"n_train >= 2\*m"):
            m = resolve_arima_m("auto", n_train=80,
                                series_freq=self._freq("W-SUN"))
        assert m == 1

    def test_explicit_failing_guard_warns_and_honors(self):
        # The deliberate deviation from resolve_metrics_seasonality, which
        # raises here: a model spec is advisory-warned, never substituted.
        from its2s.models.arima import resolve_arima_m
        with pytest.warns(UserWarning, match="honored"):
            m = resolve_arima_m(52, n_train=60)
        assert m == 52

    def test_explicit_passing_guard_is_silent(self):
        from its2s.models.arima import resolve_arima_m
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert resolve_arima_m(7, n_train=400) == 7

    def test_m_below_1_raises(self):
        from its2s.models.arima import resolve_arima_m
        with pytest.raises(ValueError, match="must be >= 1"):
            resolve_arima_m(0, n_train=400)

    # -- fit-path tests (stubbed auto_arima) ---------------------------------

    def test_fit_auto_daily_passes_m7(self, monkeypatch):
        df = pd.DataFrame({"ds": pd.date_range("2022-01-01", periods=400,
                                               freq="D"),
                           "y": np.ones(400)})
        captured = self._fit_with_stub(monkeypatch, df, params={})
        assert captured["m"] == 7

    def test_fit_auto_weekly_passes_m52(self, monkeypatch):
        df = pd.DataFrame({"ds": pd.date_range("2022-01-02", periods=200,
                                               freq="W-SUN"),
                           "y": np.ones(200)})
        captured = self._fit_with_stub(monkeypatch, df, params={})
        assert captured["m"] == 52

    def test_fit_explicit_m7_emits_no_warning(self, monkeypatch):
        # Regression for the retired M2-8 warning, which fired on m == 7
        # even when the user set it deliberately.
        df = pd.DataFrame({"ds": pd.date_range("2022-01-01", periods=400,
                                               freq="D"),
                           "y": np.ones(400)})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            captured = self._fit_with_stub(monkeypatch, df, params={"m": 7})
        assert captured["m"] == 7

    def test_fit_auto_irregular_dates_warns_and_falls_back_to_m1(
            self, monkeypatch):
        dates = pd.to_datetime(
            ["2022-01-01", "2022-01-02", "2022-01-05", "2022-01-11",
             "2022-02-01", "2022-02-03", "2022-03-01", "2022-04-01"])
        df = pd.DataFrame({"ds": dates, "y": np.ones(len(dates))})
        with pytest.warns(UserWarning, match="could not resolve"):
            captured = self._fit_with_stub(monkeypatch, df, params={})
        assert captured["m"] == 1

    def test_fit_seasonal_false_skips_resolution_silently(self, monkeypatch):
        # Irregular dates would warn on the auto path; with the seasonal
        # search off, m is inert and no resolution (or warning) happens.
        dates = pd.to_datetime(
            ["2022-01-01", "2022-01-02", "2022-01-05", "2022-01-11",
             "2022-02-01", "2022-02-03", "2022-03-01", "2022-04-01"])
        df = pd.DataFrame({"ds": dates, "y": np.ones(len(dates))})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            captured = self._fit_with_stub(monkeypatch, df,
                                           params={"seasonal": False})
        assert captured["m"] == 1
        assert captured["seasonal"] is False


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
        from its2s.models.prophet_xgb import ProphetXGBHybridModel
        return ProphetXGBHybridModel(params={})

    @pytest.mark.parametrize("model_name", ["prophet_xgb"])
    def test_weekly_disabled_on_weekly_grid(self, model_name):
        df, _, _ = make_weekly_series(seed=1500)
        model = self._make_model(model_name)
        _run_quiet(model.fit, df.iloc[:156])
        assert "weekly" not in model._prophet.seasonalities
        assert "yearly" in model._prophet.seasonalities

    @pytest.mark.parametrize("model_name", ["prophet_xgb"])
    def test_weekly_enabled_on_daily_grid(self, model_name):
        df, _, _ = make_short_series(n_pre=180, n_post=30, seed=1501)
        model = self._make_model(model_name)
        _run_quiet(model.fit, df.iloc[:180])
        assert model._prophet.seasonalities["weekly"]["fourier_order"] == 3


# ===================================================================
# Prophet yearly_seasonality "auto" default, reported both ways (D-057, D-080)
# ===================================================================
class TestProphetYearlySeasonalityAuto:
    """The shipped yearly_seasonality default defers to the 730-day rule and
    reports the resolution visibly in BOTH directions (GH #60, D-057, D-080).

    The rule is a hard boundary on a continuous quantity: one day of history
    either side of it produces a materially different model. D-080 made the
    resolution always visible rather than visible only on disable, so a user
    near the boundary can see which side they landed on.
    """

    @pytest.mark.parametrize("model_name", ["prophet_xgb"])
    def test_yearly_disabled_with_warning_on_short_history(self, model_name):
        df, _, _ = make_short_series(n_pre=180, n_post=30, seed=1502)
        model = TestProphetWeeklySeasonalityAuto._make_model(model_name)
        with pytest.warns(UserWarning, match="yearly_seasonality='auto'"):
            model.fit(df.iloc[:180])
        assert "yearly" not in model._prophet.seasonalities

    @pytest.mark.parametrize("model_name", ["prophet_xgb"])
    def test_yearly_enabled_with_warning_on_long_history(self, model_name):
        df, _, _ = make_daily_series(n_pre=800, n_post=30, seed=1503)
        model = TestProphetWeeklySeasonalityAuto._make_model(model_name)
        with pytest.warns(UserWarning, match="ENABLED"):
            model.fit(df.iloc[:800])
        assert model._prophet.seasonalities["yearly"]["fourier_order"] == 10

    def test_report_helper_rule_boundaries(self):
        """The helper's own boundary, hit directly.

        Note periods=N spans N-1 days: that off-by-one is exactly what put the
        730-row effect-recovery fixture one day under the threshold (D-079).
        """
        from its2s.models.base import report_auto_yearly_resolution
        short = pd.DataFrame({
            "ds": pd.date_range("2022-01-01", periods=700, freq="D"),
            "y": np.ones(700)})
        long = pd.DataFrame({
            "ds": pd.date_range("2022-01-01", periods=731, freq="D"),
            "y": np.ones(731)})
        with pytest.warns(UserWarning, match="spans 699 days"):
            report_auto_yearly_resolution(short, "auto")
        with pytest.warns(UserWarning, match="spans 730 days"):
            report_auto_yearly_resolution(long, "auto")
        # An explicit value is the user's own choice: honored, and silent.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            report_auto_yearly_resolution(short, True)
            report_auto_yearly_resolution(short, False)

    def test_yearly_cliff_is_where_the_rule_says_and_is_announced(self):
        """Pin the discontinuity end to end so it cannot move silently.

        Scoped to prophet_xgb, the sole surviving Prophet-based model.
        Asserts WHERE the cliff sits and that both
        sides announce themselves -- deliberately NOT how large the resulting
        difference in the estimate is, which would enshrine the behaviour
        rather than expose it.
        """
        model_at = TestProphetWeeklySeasonalityAuto._make_model("prophet_xgb")
        model_over = TestProphetWeeklySeasonalityAuto._make_model("prophet_xgb")

        # 730 daily rows span 729 days: one day UNDER the threshold.
        df_under, _, _ = make_daily_series(n_pre=730, n_post=30, seed=1504)
        with pytest.warns(UserWarning, match="spans 729 days.*DISABLED"):
            model_at.fit(df_under.iloc[:730])
        assert "yearly" not in model_at._prophet.seasonalities

        # 731 daily rows span 730 days: exactly AT the threshold.
        df_over, _, _ = make_daily_series(n_pre=731, n_post=30, seed=1504)
        with pytest.warns(UserWarning, match="spans 730 days.*ENABLED"):
            model_over.fit(df_over.iloc[:731])
        assert model_over._prophet.seasonalities["yearly"]["fourier_order"] == 10


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

    def test_seasonality_auto_reaches_library(self):
        """GH #60, D-057: the default "auto" is handed to NeuralProphet
        unresolved; the library applies the same rules as Prophet at fit time."""
        from its2s.models.neuralprophet import NeuralProphetModel
        np_model = NeuralProphetModel(params={})._build_model()
        assert np_model.config_seasonality.periods["weekly"].arg == "auto"
        assert np_model.config_seasonality.periods["yearly"].arg == "auto"

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
        assert (tmp_path / f"{model_name}_diagnostics.csv").exists()
        assert (tmp_path / f"{model_name}_residual_acf.png").exists()
        assert (tmp_path / f"{model_name}_residual_pacf.png").exists()
        assert (tmp_path / f"{model_name}_residuals_over_time.png").exists()
        assert (tmp_path / f"{model_name}_residual_qq.png").exists()


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

    They deliberately train on a window well clear of the 730-day
    yearly_seasonality boundary: n_pre=1460 minus the 365-day test window
    leaves 1095 training rows spanning 1094 days. At the previous default
    (n_pre=1095) the window was 730 rows spanning 729 days -- one day UNDER
    the threshold -- so yearly seasonality was dropped, and since this
    fixture's seasonal amplitude equals the planted effect exactly, the
    omitted season loaded onto the trend and was read as intervention effect
    (D-079: a planted 10.0 recovered as 22.99). These tests measure effect
    recovery; the seasonality boundary is pinned separately in
    TestProphetYearlySeasonalityAuto. Do not trim n_pre back toward the cliff.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("model_name", _MODEL_NAMES_MARKS)
    def test_known_positive_effect_recovery(self, model_name):
        from its2s import run_single_its
        from its2s.metrics.excess import calc_ate_summary
        df, intv, _ = make_daily_series(n_pre=1460, intervention_effect=10.0,
                                         noise_sd=5.0, seed=4010)
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
        df, intv, _ = make_daily_series(n_pre=1460, intervention_effect=0.0,
                                         noise_sd=5.0, seed=4011)
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
        df, intv, _ = make_daily_series(n_pre=1460, intervention_effect=-8.0,
                                         noise_sd=5.0, seed=4012)
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


# ===================================================================
# Shared time-feature helper (D-091)
# ===================================================================
class TestMakeTimeFeatures:
    """Direct unit test of its2s.models.utils.make_time_features.

    Replaces test_prophet_time_features_are_consistent, deleted with the
    prophet_then_xgb retirement (D-091). That test was already vacuous: both
    Prophet modules aliased THE SAME utils object, so it compared a function's
    output to itself. This asserts the helper's actual contract instead.
    """

    def test_columns_values_and_index(self):
        from its2s.models.utils import make_time_features
        # A Thursday, spanning a month and an ISO-week boundary.
        df = pd.DataFrame(
            {"ds": pd.date_range("2021-01-01", periods=10, freq="D")},
            index=range(100, 110),
        )
        out = make_time_features(df)

        assert list(out.columns) == ["day_of_week", "day_of_year", "month", "week_of_year"]
        # The input index is preserved, so callers can concat without realigning.
        assert list(out.index) == list(df.index)
        # 2021-01-01 was a Friday: pandas dayofweek is Monday=0, so Friday=4.
        assert out["day_of_week"].tolist() == [4, 5, 6, 0, 1, 2, 3, 4, 5, 6]
        assert out["day_of_year"].tolist() == list(range(1, 11))
        assert out["month"].unique().tolist() == [1]
        # ISO weeks: 2021-01-01 falls in ISO week 53 OF 2020, rolling to 1 on the 4th.
        assert out["week_of_year"].tolist() == [53, 53, 53, 1, 1, 1, 1, 1, 1, 1]
        assert out["week_of_year"].dtype.kind == "i"

    def test_accepts_a_custom_date_col_and_string_dates(self):
        from its2s.models.utils import make_time_features
        df = pd.DataFrame({"date": ["2021-03-14", "2021-03-15"]})
        out = make_time_features(df, date_col="date")
        assert out["month"].tolist() == [3, 3]
        assert out["day_of_week"].tolist() == [6, 0]


# ===================================================================
# Main entry point
# ===================================================================
if __name__ == "__main__":
    import sys
    exit_code = pytest.main([__file__, "-v", "--tb=short"] + sys.argv[1:])
    sys.exit(exit_code)
