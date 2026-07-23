# Description: Infrastructure test suite for its2s package.
#   Covers: settings, data_prep, validation, error_metrics, bootstrap internals,
#           block_length, excess/ATE, diagnostics, outputs, batch,
#           cross_validation, and model comparison.
#   Model-facing tests (contract, integration, statistical) live in test_models.py.
# Usage: python -m pytest tests/test_its2s.py -v --tb=short
# Dependencies: pytest, numpy, pandas, its2s

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from conftest import (
    _FAST,
    _run_quiet,
    make_constant_series,
    make_count_series,
    make_daily_series,
    make_intervention_at_boundary,
    make_missing_data_series,
    make_mock_bootstrap_result,
    make_monthly_series,
    make_outlier_series,
    make_quarterly_series,
    make_series_with_covariates,
    make_short_series,
    make_weekly_series,
)

pytestmark = [pytest.mark.filterwarnings("ignore::FutureWarning")]


# ===================================================================
# Settings & Config
# ===================================================================
class TestSettings:
    def test_load_default_config(self, default_config):
        expected_keys = {"data", "periods", "models", "bootstrap", "metrics",
                         "output", "parallel", "excess_periods"}
        assert expected_keys.issubset(set(default_config.keys()))

    def test_default_config_values(self, default_config):
        assert default_config["periods"]["test_days"] == 365
        assert default_config["periods"]["holdout_days"] == 365
        assert default_config["bootstrap"]["n_sim"] == 1000
        assert default_config["bootstrap"]["block_length"] == 14
        assert default_config["bootstrap"]["ci_level"] == 0.95
        assert default_config["data"]["target_col"] == "y"
        assert default_config["data"]["date_col"] == "ds"

    def test_config_override_shallow(self):
        from its2s.settings import load_config
        cfg = load_config(overrides={"bootstrap": {"n_sim": 50}})
        assert cfg["bootstrap"]["n_sim"] == 50
        assert cfg["bootstrap"]["block_length"] == 14  # unchanged

    def test_config_override_deep_merge(self):
        from its2s.settings import load_config
        cfg = load_config(overrides={"models": {"arima": {"max_p": 3}}})
        assert cfg["models"]["arima"]["max_p"] == 3
        assert cfg["models"]["arima"]["max_d"] == 2  # preserved

    def test_config_override_adds_new_key(self):
        from its2s.settings import load_config
        cfg = load_config(overrides={"custom_key": "custom_value"})
        assert cfg["custom_key"] == "custom_value"

    def test_get_model_config_existing(self, default_config):
        from its2s.settings import get_model_config
        arima_cfg = get_model_config(default_config, "arima")
        assert "max_p" in arima_cfg
        assert "max_d" in arima_cfg

    def test_get_model_config_missing(self, default_config):
        from its2s.settings import get_model_config
        result = get_model_config(default_config, "nonexistent_model")
        assert result == {}


# ===================================================================
# Data Prep
# ===================================================================
class TestDataPrep:
    def test_prepare_splits_basic(self):
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_daily_series(n_pre=1095, n_post=365)
        splits = prepare_splits(df, intv, split_method="days", test_days=365, holdout_days=365)
        train_max = splits.train_df["ds"].max()
        test_min = splits.test_df["ds"].min()
        test_max = splits.test_df["ds"].max()
        holdout_min = splits.holdout_df["ds"].min()
        assert train_max < test_min, "Train and test should not overlap"
        assert test_max < holdout_min, "Test and holdout should not overlap"

    def test_prepare_splits_lengths(self):
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_daily_series(n_pre=1095, n_post=365)
        splits = prepare_splits(df, intv, split_method="days", test_days=365, holdout_days=365)
        assert len(splits.train_df) == 730  # 1095 - 365
        assert len(splits.test_df) == 365
        assert len(splits.holdout_df) == 365

    def test_prepare_splits_full_predict(self):
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_daily_series(n_pre=1095, n_post=365)
        splits = prepare_splits(df, intv, split_method="days", test_days=365, holdout_days=365)
        expected_len = len(splits.test_df) + len(splits.holdout_df)
        assert len(splits.full_predict_df) == expected_len

    def test_prepare_splits_string_date(self):
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_daily_series(n_pre=1095, n_post=365)
        splits = prepare_splits(df, str(intv), split_method="days", test_days=365, holdout_days=365)
        assert isinstance(splits.intervention_date, pd.Timestamp)

    def test_prepare_splits_custom_date_col(self):
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_daily_series(n_pre=1095, n_post=365)
        df = df.rename(columns={"ds": "date"})
        splits = prepare_splits(df, intv, date_col="date",
                                split_method="days",
                                test_days=365, holdout_days=365)
        assert "date" in splits.train_df.columns

    def test_prepare_splits_short_series(self):
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_short_series(n_pre=30, n_post=30)
        splits = prepare_splits(df, intv, split_method="days", test_days=365, holdout_days=365)
        assert len(splits.train_df) == 0

    def test_prepare_splits_intervention_at_start(self):
        from its2s.data_prep import prepare_splits
        df, intv = make_intervention_at_boundary("start")
        splits = prepare_splits(df, intv, split_method="days", test_days=30, holdout_days=30)
        assert len(splits.train_df) == 0 or len(splits.test_df) == 0

    def test_prepare_splits_intervention_at_end(self):
        from its2s.data_prep import prepare_splits
        df, intv = make_intervention_at_boundary("end")
        splits = prepare_splits(df, intv, split_method="days", test_days=30, holdout_days=30)
        assert len(splits.holdout_df) <= 1


# ===================================================================
# Input Validation
# ===================================================================
class TestValidation:
    def test_missing_date_col_raises(self):
        from its2s.validation import validate_inputs
        df = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
        with pytest.raises(ValueError, match="ds"):
            validate_inputs(df, "2021-01-02", "ds", "y", None, "arima")

    def test_missing_target_col_raises(self):
        from its2s.validation import validate_inputs
        df = pd.DataFrame({"ds": pd.date_range("2021-01-01", periods=3)})
        with pytest.raises(ValueError, match="y"):
            validate_inputs(df, "2021-01-02", "ds", "y", None, "arima")

    def test_empty_dataframe_raises(self):
        from its2s.validation import validate_inputs
        df = pd.DataFrame({"ds": [], "y": []})
        with pytest.raises(ValueError, match="empty"):
            validate_inputs(df, "2021-01-01", "ds", "y", None, "arima")

    def test_zero_variance_target_raises(self):
        from its2s.validation import validate_inputs
        df = pd.DataFrame({
            "ds": pd.date_range("2021-01-01", periods=10),
            "y": np.full(10, 5.0),
        })
        with pytest.raises(ValueError, match="zero variance"):
            validate_inputs(df, "2021-01-05", "ds", "y", None, "arima")

    def test_missing_covariate_col_raises(self):
        from its2s.validation import validate_inputs
        df, intv, _ = make_short_series(n_pre=30, n_post=10, seed=42)
        with pytest.raises(ValueError, match="nonexistent"):
            validate_inputs(df, intv, "ds", "y", ["nonexistent"], "arima")

    def test_valid_inputs_does_not_raise(self):
        from its2s.validation import validate_inputs
        df, intv, _ = make_daily_series(n_pre=400, n_post=100, seed=42)
        # Should complete without raising.
        validate_inputs(df, intv, "ds", "y", None, "arima")

    def test_intervention_outside_range_does_not_raise(self):
        import warnings
        from its2s.validation import validate_inputs
        df, intv, _ = make_short_series(n_pre=30, n_post=10, seed=42)
        far_future = pd.Timestamp("2050-01-01")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_inputs(df, far_future, "ds", "y", None, "arima")
        assert any("outside" in str(w.message).lower() for w in caught)

    def test_few_pre_obs_does_not_raise(self):
        import warnings
        from its2s.validation import validate_inputs
        df = pd.DataFrame({
            "ds": pd.date_range("2021-01-01", periods=15),
            "y": np.random.default_rng(0).standard_normal(15) + 10,
        })
        intv = df["ds"].iloc[5]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_inputs(df, intv, "ds", "y", None, "arima")
        assert any("observations" in str(w.message).lower() for w in caught)

    def test_high_missing_fraction_does_not_raise(self):
        import warnings
        from its2s.validation import validate_inputs
        df, intv, _ = make_missing_data_series(frac_missing=0.25, seed=42)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            validate_inputs(df, intv, "ds", "y", None, "arima")
        assert any("missing" in str(w.message).lower() for w in caught)


# ===================================================================
# Error Metrics
# ===================================================================
class TestErrorMetrics:
    def test_perfect_predictions(self):
        from its2s.metrics.error_metrics import compute_metrics
        a = np.arange(1.0, 101.0)
        m = compute_metrics(a, a)
        assert m.rmse == pytest.approx(0.0, abs=1e-10)
        assert m.mae == pytest.approx(0.0, abs=1e-10)
        assert m.r2 == pytest.approx(1.0, abs=1e-10)

    def test_rmse_known_value(self):
        from its2s.metrics.error_metrics import compute_metrics
        a = np.array([3.0, 3.0, 3.0])
        p = np.array([1.0, 2.0, 3.0])
        m = compute_metrics(a, p)
        expected_rmse = math.sqrt((4 + 1 + 0) / 3)
        assert m.rmse == pytest.approx(expected_rmse, rel=1e-6)

    def test_mae_known_value(self):
        from its2s.metrics.error_metrics import compute_metrics
        a = np.array([3.0, 3.0, 3.0])
        p = np.array([1.0, 2.0, 3.0])
        m = compute_metrics(a, p)
        assert m.mae == pytest.approx(1.0, rel=1e-6)

    def test_mape_known_value(self):
        from its2s.metrics.error_metrics import compute_metrics
        a = np.array([100.0, 200.0, 50.0])
        p = np.array([110.0, 190.0, 55.0])
        expected_mape = np.mean([10 / 100, 10 / 200, 5 / 50]) * 100
        m = compute_metrics(a, p)
        assert m.mape == pytest.approx(expected_mape, rel=1e-6)

    def test_smape_known_value(self):
        from its2s.metrics.error_metrics import compute_metrics
        a = np.array([100.0, 200.0])
        p = np.array([110.0, 180.0])
        expected = np.mean([20 / 210, 40 / 380]) * 100
        m = compute_metrics(a, p)
        assert m.smape == pytest.approx(expected, rel=1e-6)

    def test_mase_with_training_data(self):
        from its2s.metrics.error_metrics import compute_metrics
        training = np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0,
                             24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0])
        a = np.array([40.0, 42.0])
        p = np.array([39.0, 44.0])
        m = compute_metrics(a, p, training_actual=training, seasonality=7)
        assert m.mase is not None
        assert isinstance(m.mase, float)

    def test_mase_none_without_training(self):
        from its2s.metrics.error_metrics import compute_metrics
        a = np.array([1.0, 2.0, 3.0])
        p = np.array([1.1, 2.1, 3.1])
        m = compute_metrics(a, p)
        assert m.mase is None

    def test_r2_perfect(self):
        from its2s.metrics.error_metrics import compute_metrics
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = compute_metrics(a, a)
        assert m.r2 == pytest.approx(1.0, abs=1e-10)

    def test_r2_constant_prediction(self):
        from its2s.metrics.error_metrics import compute_metrics
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        p = np.full(5, 3.0)
        m = compute_metrics(a, p)
        assert m.r2 == pytest.approx(0.0, abs=1e-10)

    def test_mape_with_zeros_in_actual(self):
        from its2s.metrics.error_metrics import compute_metrics
        a = np.array([0.0, 10.0, 20.0])
        p = np.array([5.0, 10.0, 25.0])
        m = compute_metrics(a, p)
        assert np.isfinite(m.mape)

    def test_smape_all_zero(self):
        from its2s.metrics.error_metrics import compute_metrics
        a = np.array([0.0, 0.0])
        p = np.array([0.0, 0.0])
        m = compute_metrics(a, p)
        assert m.smape == pytest.approx(0.0, abs=1e-10)

    def test_metrics_result_dataclass(self):
        from its2s.metrics.error_metrics import compute_metrics, MetricsResult
        a = np.array([1.0, 2.0, 3.0])
        p = np.array([1.1, 2.1, 3.1])
        m = compute_metrics(a, p)
        assert isinstance(m, MetricsResult)
        for attr in ("rmse", "mae", "mape", "smape", "r2"):
            assert hasattr(m, attr)


# ===================================================================
# Bootstrap Internals
# ===================================================================
class TestBootstrap:
    """Unit tests for the MBB resampling mechanism and CI calculation.

    These tests use ARIMA as a generic fitted-model vehicle because the
    tested behavior belongs to the bootstrap infrastructure, not to the
    model.  Cross-model bootstrap tests live in test_models.py.
    """

    def test_resample_blocks_length(self):
        from its2s.bootstrap.mbb import _resample_blocks
        rng = np.random.default_rng(42)
        residuals = rng.standard_normal(100)
        resampled = _resample_blocks(residuals, block_length=5, rng=rng)
        assert len(resampled) == len(residuals)

    def test_resample_blocks_short_residuals(self):
        from its2s.bootstrap.mbb import _resample_blocks
        rng = np.random.default_rng(42)
        residuals = np.array([1.0, 2.0, 3.0])
        resampled = _resample_blocks(residuals, block_length=10, rng=rng)
        assert len(resampled) == len(residuals)

    def test_resample_blocks_reproducibility(self):
        from its2s.bootstrap.mbb import _resample_blocks
        residuals = np.random.default_rng(0).standard_normal(50)
        r1 = _resample_blocks(residuals, 5, np.random.default_rng(42))
        r2 = _resample_blocks(residuals, 5, np.random.default_rng(42))
        np.testing.assert_array_equal(r1, r2)

    def test_mbb_ci_level_affects_width(self):
        from its2s.bootstrap.mbb import MovingBlockBootstrap
        from its2s.models.arima import ARIMAModel
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_short_series(n_pre=180, n_post=30, seed=77)
        splits = prepare_splits(df, intv, split_method="days", test_days=30, holdout_days=30)
        model = ARIMAModel(params={"seasonal": False, "m": 1, "stepwise": True,
                                   "suppress_warnings": True})
        model.fit(splits.train_df)
        mbb90 = MovingBlockBootstrap(n_sim=10, block_length=7, ci_level=0.90, n_jobs=1)
        mbb99 = MovingBlockBootstrap(n_sim=10, block_length=7, ci_level=0.99, n_jobs=1)
        r90 = mbb90.generate_cis(model, splits.train_df, splits.full_predict_df, seed=42)
        r99 = mbb99.generate_cis(model, splits.train_df, splits.full_predict_df, seed=42)
        width90 = np.mean(r90.conf_hi - r90.conf_lo)
        width99 = np.mean(r99.conf_hi - r99.conf_lo)
        assert width99 >= width90

    def test_mbb_unfitted_model_raises(self):
        from its2s.bootstrap.mbb import MovingBlockBootstrap
        from its2s.models.arima import ARIMAModel
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_short_series(n_pre=180, n_post=30, seed=77)
        splits = prepare_splits(df, intv, split_method="days", test_days=30, holdout_days=30)
        model = ARIMAModel(params={})
        mbb = MovingBlockBootstrap(n_sim=5, block_length=7, n_jobs=1)
        with pytest.raises(ValueError, match="fitted"):
            mbb.generate_cis(model, splits.train_df, splits.full_predict_df, seed=42)

    def test_calculate_ci_quantile_method(self):
        from its2s.bootstrap.base import BaseBootstrap
        rng = np.random.default_rng(42)
        pred_matrix = rng.standard_normal((10, 1000))
        point_est = np.zeros(10)
        lo, hi = BaseBootstrap.calculate_ci(pred_matrix, point_est,
                                            method="quantile", level=0.95)
        assert lo.shape == (10,)
        assert hi.shape == (10,)
        assert np.all(lo < 0)
        assert np.all(hi > 0)

    def test_calculate_ci_symmetric_sd_method(self):
        from its2s.bootstrap.base import BaseBootstrap
        from scipy.stats import norm
        rng = np.random.default_rng(42)
        pred_matrix = rng.standard_normal((5, 500))
        point_est = np.zeros(5)
        lo, hi = BaseBootstrap.calculate_ci(pred_matrix, point_est,
                                            method="symmetric_sd", level=0.95)
        sd = np.nanstd(pred_matrix, axis=1)
        z = norm.ppf(0.975)
        np.testing.assert_allclose(lo, -z * sd, rtol=1e-6)
        np.testing.assert_allclose(hi, z * sd, rtol=1e-6)


# ===================================================================
# Block Length
# ===================================================================
class TestBlockLength:
    def test_fixed_block_length_default(self):
        from its2s.bootstrap.block_length import fixed_block_length
        assert fixed_block_length() == 14

    def test_fixed_block_length_custom(self):
        from its2s.bootstrap.block_length import fixed_block_length
        assert fixed_block_length(7) == 7

    def test_nppi_raises_not_implemented(self):
        from its2s.bootstrap.block_length import nppi_block_length
        with pytest.raises(NotImplementedError):
            nppi_block_length(np.zeros(100))


# ===================================================================
# Excess & ATE
# ===================================================================
class TestExcess:
    def test_obs_excess_columns(self):
        from its2s.metrics.excess import calculate_excess
        br = make_mock_bootstrap_result()
        intervention_date = pd.Timestamp("2021-01-31")
        result = calculate_excess(br, intervention_date)
        expected_cols = {"date", "observed", "expected", "expected_ci_lo",
                         "expected_ci_hi", "excess", "excess_ci_lo",
                         "excess_ci_hi", "excess_pct", "excess_pct_ci_lo",
                         "excess_pct_ci_hi"}
        assert expected_cols == set(result.obs_excess.columns)

    def test_obs_excess_values(self):
        from its2s.metrics.excess import calculate_excess
        br = make_mock_bootstrap_result(actual_shift=10.0)
        intervention_date = pd.Timestamp("2021-01-31")
        result = calculate_excess(br, intervention_date)
        mean_excess = result.obs_excess["excess"].mean()
        assert mean_excess > 5.0

    def test_period_excess_full_holdout(self):
        from its2s.metrics.excess import calculate_excess
        br = make_mock_bootstrap_result()
        intervention_date = pd.Timestamp("2021-01-31")
        result = calculate_excess(br, intervention_date)
        assert len(result.period_excess) >= 1
        assert "Full holdout" in result.period_excess["period"].values

    def test_period_excess_custom_periods_days(self):
        from its2s.metrics.excess import calculate_excess
        br = make_mock_bootstrap_result()
        intervention_date = pd.Timestamp("2021-01-31")
        periods = [{"name": "First 7 days",
                    "start_offset_days": 0, "end_offset_days": 7}]
        result = calculate_excess(br, intervention_date, periods_config=periods)
        period_names = result.period_excess["period"].values
        assert "First 7 days" in period_names
        first7 = result.period_excess[result.period_excess["period"] == "First 7 days"]
        assert first7["n_obs"].values[0] == 8  # inclusive

    def test_period_excess_custom_periods_obs(self):
        from its2s.metrics.excess import calculate_excess
        # Weekly grid: obs and calendar days disagree, so this exercises the
        # row-slicing family specifically (4 rows span 22 calendar days).
        br = make_mock_bootstrap_result(freq="W-SUN")
        dates = pd.to_datetime(br.dates)
        intervention_date = dates[30]
        periods = [{"name": "First 4 obs",
                    "start_offset_obs": 0, "end_offset_obs": 3}]
        result = calculate_excess(br, intervention_date, periods_config=periods)
        first4 = result.period_excess[result.period_excess["period"] == "First 4 obs"]
        assert first4["n_obs"].values[0] == 4
        assert first4["start_date"].values[0] == dates[30]
        assert first4["end_date"].values[0] == dates[33]

    def test_period_excess_legacy_offset_keys_raise(self):
        from its2s.metrics.excess import calculate_excess
        br = make_mock_bootstrap_result()
        intervention_date = pd.Timestamp("2021-01-31")
        periods = [{"name": "Legacy", "start_offset": 0, "end_offset": 7}]
        with pytest.raises(ValueError, match="no longer accepted"):
            calculate_excess(br, intervention_date, periods_config=periods)

    def test_period_excess_mixed_offset_units_raise(self):
        from its2s.metrics.excess import calculate_excess
        br = make_mock_bootstrap_result()
        intervention_date = pd.Timestamp("2021-01-31")
        periods = [{"name": "Mixed",
                    "start_offset_days": 0, "end_offset_obs": 7}]
        with pytest.raises(ValueError, match="mixes"):
            calculate_excess(br, intervention_date, periods_config=periods)

    def test_period_excess_missing_offsets_raise(self):
        from its2s.metrics.excess import calculate_excess
        br = make_mock_bootstrap_result()
        intervention_date = pd.Timestamp("2021-01-31")
        periods = [{"name": "No offsets"}]
        with pytest.raises(ValueError, match="no offsets"):
            calculate_excess(br, intervention_date, periods_config=periods)

    def test_excess_no_holdout(self):
        from its2s.metrics.excess import calculate_excess
        br = make_mock_bootstrap_result()
        future_date = pd.Timestamp("2025-01-01")
        result = calculate_excess(br, future_date)
        assert result.obs_excess.empty
        assert result.period_excess.empty

    def test_excess_pct_calculation(self):
        from its2s.metrics.excess import calculate_excess
        br = make_mock_bootstrap_result(base_predicted=100.0, actual_shift=10.0)
        intervention_date = pd.Timestamp("2021-01-31")
        result = calculate_excess(br, intervention_date)
        row = result.obs_excess.iloc[0]
        if row["expected"] != 0:
            expected_pct = row["excess"] / row["expected"] * 100
            assert row["excess_pct"] == pytest.approx(expected_pct, rel=1e-6)

    def test_calc_ate_summary_basic(self):
        from its2s.metrics.excess import calc_ate_summary, calculate_excess
        br = make_mock_bootstrap_result()
        intervention_date = pd.Timestamp("2021-01-31")
        excess = calculate_excess(br, intervention_date)
        ate = calc_ate_summary(excess.obs_excess)
        assert len(ate) == 2
        assert set(ate["metric"].values) == {"Total ATE", "Mean ATE per obs"}

    def test_calc_ate_summary_values(self):
        from its2s.metrics.excess import calc_ate_summary, calculate_excess
        br = make_mock_bootstrap_result(actual_shift=10.0)
        intervention_date = pd.Timestamp("2021-01-31")
        excess = calculate_excess(br, intervention_date)
        ate = calc_ate_summary(excess.obs_excess)
        total = ate[ate["metric"] == "Total ATE"]["estimate"].values[0]
        mean_per_obs = ate[ate["metric"] == "Mean ATE per obs"]["estimate"].values[0]
        n = len(excess.obs_excess)
        assert mean_per_obs == pytest.approx(total / n, rel=1e-6)

    def test_calc_ate_summary_empty(self):
        from its2s.metrics.excess import calc_ate_summary
        ate = calc_ate_summary(pd.DataFrame())
        assert ate.empty


# ===================================================================
# Residual Diagnostics
# ===================================================================
class TestDiagnostics:
    """Unit tests for compute_diagnostics() and DiagnosticsResult."""

    def _make_fit_result(self, n=300, seed=42):
        """Minimal FitResult with synthetic residuals."""
        from its2s.models.base import FitResult
        rng = np.random.default_rng(seed)
        fitted = 100.0 + 0.1 * np.arange(n, dtype=float)
        residuals = rng.normal(0, 2.0, n)
        return FitResult(fitted_values=fitted, residuals=residuals)

    def test_returns_diagnostics_result(self):
        from its2s.diagnostics import compute_diagnostics, DiagnosticsResult
        fr = self._make_fit_result()
        d = compute_diagnostics(fr, "test_model")
        assert isinstance(d, DiagnosticsResult)

    def test_residual_mean_and_std_are_finite(self):
        from its2s.diagnostics import compute_diagnostics
        fr = self._make_fit_result()
        d = compute_diagnostics(fr, "test_model")
        assert np.isfinite(d.residual_mean)
        assert np.isfinite(d.residual_std)
        assert d.residual_std > 0

    def test_residual_mean_near_zero_for_zero_mean_noise(self):
        from its2s.diagnostics import compute_diagnostics
        fr = self._make_fit_result(n=500, seed=7)
        d = compute_diagnostics(fr, "test_model")
        assert abs(d.residual_mean) < 1.0

    def test_acf_values_in_valid_range(self):
        from its2s.diagnostics import compute_diagnostics
        fr = self._make_fit_result(n=200, seed=1)
        d = compute_diagnostics(fr, "test_model")
        for acf_val in (d.acf_lag1, d.acf_lag7, d.acf_lag14):
            if np.isfinite(acf_val):
                assert -1.0 <= acf_val <= 1.0

    def test_ljung_box_pvalue_in_range(self):
        from its2s.diagnostics import compute_diagnostics
        fr = self._make_fit_result(n=200, seed=2)
        d = compute_diagnostics(fr, "test_model")
        if np.isfinite(d.ljung_box_pvalue):
            assert 0.0 <= d.ljung_box_pvalue <= 1.0

    def test_shapiro_none_for_large_n(self):
        from its2s.diagnostics import compute_diagnostics
        fr = self._make_fit_result(n=6000, seed=3)
        d = compute_diagnostics(fr, "test_model", max_shapiro_n=5000)
        assert d.shapiro_pvalue is None
        assert d.shapiro_stat is None

    def test_shapiro_computed_for_small_n(self):
        from its2s.diagnostics import compute_diagnostics
        fr = self._make_fit_result(n=100, seed=4)
        d = compute_diagnostics(fr, "test_model", max_shapiro_n=5000)
        assert d.shapiro_pvalue is not None
        assert isinstance(d.shapiro_pvalue, float)
        assert 0.0 <= d.shapiro_pvalue <= 1.0

    def test_nan_residuals_handled_gracefully(self):
        from its2s.diagnostics import compute_diagnostics
        from its2s.models.base import FitResult
        rng = np.random.default_rng(5)
        residuals = rng.normal(0, 1.0, 100)
        residuals[[0, 5, 10]] = np.nan
        fr = FitResult(fitted_values=np.ones(100), residuals=residuals)
        d = compute_diagnostics(fr, "np_model")
        assert np.isfinite(d.residual_mean)
        assert np.isfinite(d.residual_std)

    def test_model_name_stored_in_metadata(self):
        from its2s.diagnostics import compute_diagnostics
        fr = self._make_fit_result(n=100, seed=6)
        d = compute_diagnostics(fr, "my_model")
        assert d.model_metadata.get("model_name") == "my_model"

    def test_acf_lag14_nan_for_short_residuals(self):
        from its2s.diagnostics import compute_diagnostics
        from its2s.models.base import FitResult
        residuals = np.random.default_rng(7).normal(0, 1, 10)
        fr = FitResult(fitted_values=np.ones(10), residuals=residuals)
        d = compute_diagnostics(fr, "short_model")
        assert np.isnan(d.acf_lag14)


# ===================================================================
# Outputs
# ===================================================================
class TestOutputs:
    def _make_minimal_pipeline_result(self):
        """Build a minimal PipelineResult for output tests."""
        from its2s.pipeline import PipelineResult
        from its2s.models.base import FitResult
        from its2s.metrics.error_metrics import MetricsResult
        from its2s.metrics.excess import ExcessResult

        br = make_mock_bootstrap_result()
        fr = FitResult(
            fitted_values=np.ones(30),
            residuals=np.zeros(30),
        )
        mr = MetricsResult(rmse=1.0, mae=0.8, mape=5.0, smape=4.5, mase=0.9, r2=0.95)
        er = ExcessResult(
            obs_excess=pd.DataFrame({
                "date": pd.date_range("2021-01-31", periods=10),
                "observed": np.full(10, 110.0),
                "expected": np.full(10, 100.0),
                "expected_ci_lo": np.full(10, 95.0),
                "expected_ci_hi": np.full(10, 105.0),
                "excess": np.full(10, 10.0),
                "excess_ci_lo": np.full(10, 5.0),
                "excess_ci_hi": np.full(10, 15.0),
                "excess_pct": np.full(10, 10.0),
                "excess_pct_ci_lo": np.full(10, 5.0),
                "excess_pct_ci_hi": np.full(10, 15.0),
            }),
            period_excess=pd.DataFrame({
                "period": ["Full holdout"],
                "start_date": [pd.Timestamp("2021-01-31")],
                "end_date": [pd.Timestamp("2021-02-09")],
                "n_obs": [10],
                "total_observed": [1100.0],
                "total_expected": [1000.0],
                "total_excess": [100.0],
                "excess_ci_lo": [50.0],
                "excess_ci_hi": [150.0],
                "excess_pct": [10.0],
            }),
        )
        return PipelineResult(
            model_name="test_model",
            fit_result=fr,
            bootstrap_result=br,
            metrics_train=mr,
            metrics_test=mr,
            excess_table=er,
            config={"bootstrap": {"n_sim": 10}},
        ), mr, er

    def test_plot_counterfactual_saves_to_file(self, tmp_path):
        from its2s.outputs.plots import plot_counterfactual
        from its2s.data_prep import prepare_splits
        pr, _, _ = self._make_minimal_pipeline_result()
        df, intv, _ = make_short_series(n_pre=60, n_post=30)
        splits = prepare_splits(df, intv, split_method="days", test_days=30, holdout_days=30)
        save_path = tmp_path / "test_plot.png"
        plot_counterfactual(pr, splits, save_path=save_path)
        assert save_path.exists()
        assert save_path.stat().st_size > 0

    def test_plot_no_crash_without_save(self):
        from its2s.outputs.plots import plot_counterfactual
        from its2s.data_prep import prepare_splits
        import matplotlib.figure
        pr, _, _ = self._make_minimal_pipeline_result()
        df, intv, _ = make_short_series(n_pre=60, n_post=30)
        splits = prepare_splits(df, intv, split_method="days", test_days=30, holdout_days=30)
        fig = plot_counterfactual(pr, splits, save_path=None)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_counterfactual_uses_style_overrides(self):
        from its2s.outputs.plots import plot_counterfactual
        from its2s.data_prep import prepare_splits
        import matplotlib.pyplot as plt

        pr, _, _ = self._make_minimal_pipeline_result()
        df, intv, _ = make_short_series(n_pre=60, n_post=30)
        splits = prepare_splits(df, intv, split_method="days", test_days=30, holdout_days=30)
        config = {
            "output": {
                "plot_colors": ["#984136", "#c26a7a", "#ecc0a1", "#f0f0e4"],
                "plot_font_sizes": {
                    "title": 15,
                    "axis_label": 14,
                    "tick": 13,
                    "legend": 12,
                },
            },
        }

        fig = plot_counterfactual(pr, splits, save_path=None, config=config)
        ax = fig.axes[0]
        line_colors = {line.get_label(): line.get_color() for line in ax.lines}

        assert line_colors["Model fit (train)"] == "#984136"
        assert line_colors["Counterfactual prediction"] == "#c26a7a"
        assert line_colors["Intervention"] == "#ecc0a1"
        assert ax.title.get_fontsize() == 15
        assert ax.xaxis.label.get_fontsize() == 14
        assert ax.get_legend().get_texts()[0].get_fontsize() == 12
        plt.close(fig)

    def test_save_excess_table_csv(self, tmp_path):
        from its2s.outputs.tables import save_excess_table
        _, _, er = self._make_minimal_pipeline_result()
        path = tmp_path / "excess.csv"
        save_excess_table(er, path)
        assert path.exists()
        loaded = pd.read_csv(path)
        assert "excess" in loaded.columns

    def test_save_excess_table_period_csv(self, tmp_path):
        from its2s.outputs.tables import save_excess_table
        _, _, er = self._make_minimal_pipeline_result()
        path = tmp_path / "excess.csv"
        save_excess_table(er, path)
        period_path = tmp_path / "excess_period.csv"
        assert period_path.exists()

    def test_save_metrics_table(self, tmp_path):
        from its2s.outputs.tables import save_metrics_table
        _, mr, _ = self._make_minimal_pipeline_result()
        path = tmp_path / "metrics.csv"
        save_metrics_table({"train": mr, "test": mr}, path)
        assert path.exists()
        loaded = pd.read_csv(path)
        assert "window" in loaded.columns
        assert "rmse" in loaded.columns
        assert len(loaded) == 2

    def test_save_ate_summary(self, tmp_path):
        from its2s.outputs.tables import save_ate_summary
        from its2s.metrics.excess import calc_ate_summary
        _, _, er = self._make_minimal_pipeline_result()
        ate = calc_ate_summary(er.obs_excess)
        path = tmp_path / "ate.csv"
        save_ate_summary(ate, path)
        assert path.exists()
        loaded = pd.read_csv(path)
        assert len(loaded) == 2

    def test_pipeline_result_summary_is_string(self):
        pr, _, _ = self._make_minimal_pipeline_result()
        summary = pr.summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_pipeline_result_summary_contains_model_name(self):
        pr, _, _ = self._make_minimal_pipeline_result()
        summary = pr.summary()
        assert "test_model" in summary

    def test_pipeline_result_summary_contains_metrics(self):
        pr, _, _ = self._make_minimal_pipeline_result()
        summary = pr.summary()
        assert "RMSE" in summary or "rmse" in summary.lower()


# ===================================================================
# Batch
# ===================================================================
class TestBatch:
    def test_derive_seed_deterministic(self):
        from its2s.batch.seed_manager import derive_seed
        s1 = derive_seed(42, "series_a")
        s2 = derive_seed(42, "series_a")
        assert s1 == s2

    def test_derive_seed_different_markers(self):
        from its2s.batch.seed_manager import derive_seed
        s1 = derive_seed(42, "series_a")
        s2 = derive_seed(42, "series_b")
        assert s1 != s2

    def test_derive_seed_different_global(self):
        from its2s.batch.seed_manager import derive_seed
        s1 = derive_seed(42, "series_a")
        s2 = derive_seed(99, "series_a")
        assert s1 != s2

    def test_run_batch_two_series(self, tmp_path):
        from its2s.batch.runner import run_batch
        df1, intv1, _ = make_short_series(n_pre=180, n_post=30, seed=1)
        df2, intv2, _ = make_short_series(n_pre=180, n_post=30, seed=2)
        series_list = [
            {"series_id": "s1", "df": df1, "intervention_date": intv1,
             "model_name": "arima",
             "config_overrides": {"bootstrap": {"n_sim": 5, "n_jobs": 1},
                                  "periods": {"test_days": 30, "holdout_days": 30},
                                  "models": {"arima": {"seasonal": False, "m": 1}}}},
            {"series_id": "s2", "df": df2, "intervention_date": intv2,
             "model_name": "arima",
             "config_overrides": {"bootstrap": {"n_sim": 5, "n_jobs": 1},
                                  "periods": {"test_days": 30, "holdout_days": 30},
                                  "models": {"arima": {"seasonal": False, "m": 1}}}},
        ]
        results = run_batch(series_list, output_dir=str(tmp_path), n_jobs=1, seed=42)
        assert len(results) == 2

    def test_make_run_dir_versioning(self, tmp_path):
        from its2s.batch.runner import _make_run_dir
        d1 = _make_run_dir(str(tmp_path), n_sim=10)
        d2 = _make_run_dir(str(tmp_path), n_sim=10)
        assert d1 != d2
        assert d1.exists()
        assert d2.exists()


# ===================================================================
# Cross-Validation
# ===================================================================
class TestCrossValidation:
    """Unit tests for time_series_cv() and the CV result dataclasses."""

    _CV_CFG = {
        "bootstrap": {"n_sim": 5, "n_jobs": 1},
        "models": {"arima": {"seasonal": False, "m": 1, "stepwise": True,
                              "suppress_warnings": True}},
    }

    def test_returns_cv_result(self):
        from its2s.cross_validation import time_series_cv, CVResult
        df, intv, _ = make_daily_series(n_pre=730, n_post=180, seed=600)
        result = time_series_cv(df, intv, model_name="arima",
                                n_folds=3, test_obs=60, min_train_obs=180,
                                config_overrides=self._CV_CFG)
        assert isinstance(result, CVResult)

    def test_model_name_stored_in_result(self):
        from its2s.cross_validation import time_series_cv
        df, intv, _ = make_daily_series(n_pre=730, n_post=180, seed=601)
        result = time_series_cv(df, intv, model_name="arima",
                                n_folds=3, test_obs=60, min_train_obs=180,
                                config_overrides=self._CV_CFG)
        assert result.model_name == "arima"

    def test_n_folds_respected(self):
        from its2s.cross_validation import time_series_cv
        df, intv, _ = make_daily_series(n_pre=730, n_post=180, seed=602)
        result = time_series_cv(df, intv, model_name="arima",
                                n_folds=3, test_obs=60, min_train_obs=180,
                                config_overrides=self._CV_CFG)
        assert len(result.folds) <= 3
        assert len(result.folds) >= 1

    def test_mean_rmse_positive_and_finite(self):
        from its2s.cross_validation import time_series_cv
        df, intv, _ = make_daily_series(n_pre=730, n_post=180, seed=603)
        result = time_series_cv(df, intv, model_name="arima",
                                n_folds=3, test_obs=60, min_train_obs=180,
                                config_overrides=self._CV_CFG)
        assert np.isfinite(result.mean_rmse)
        assert result.mean_rmse > 0

    def test_fold_train_sizes_non_decreasing(self):
        from its2s.cross_validation import time_series_cv
        df, intv, _ = make_daily_series(n_pre=730, n_post=180, seed=604)
        result = time_series_cv(df, intv, model_name="arima",
                                n_folds=3, test_obs=60, min_train_obs=180,
                                config_overrides=self._CV_CFG)
        train_sizes = [f.n_train for f in result.folds]
        assert train_sizes == sorted(train_sizes)

    def test_summary_is_nonempty_string(self):
        from its2s.cross_validation import time_series_cv
        df, intv, _ = make_daily_series(n_pre=730, n_post=180, seed=605)
        result = time_series_cv(df, intv, model_name="arima",
                                n_folds=2, test_obs=60, min_train_obs=180,
                                config_overrides=self._CV_CFG)
        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 0
        assert "arima" in s.lower()

    def test_insufficient_data_raises(self):
        from its2s.cross_validation import time_series_cv
        df, intv, _ = make_short_series(n_pre=30, n_post=10, seed=606)
        with pytest.raises(ValueError, match="Not enough"):
            time_series_cv(df, intv, model_name="arima",
                           n_folds=3, test_obs=60, min_train_obs=365,
                           config_overrides=self._CV_CFG)

    def test_split_method_days_raises(self):
        # CV windows are observation counts; "days" exists only in
        # prepare_splits (GH #39).
        from its2s.cross_validation import time_series_cv
        df, intv, _ = make_daily_series(n_pre=730, n_post=180, seed=619)
        with pytest.raises(ValueError, match="observation counts"):
            time_series_cv(df, intv, model_name="arima",
                           split_method="days",
                           config_overrides=self._CV_CFG)

    def test_pct_args_under_observations_raise(self):
        # Cross-method window args raise instead of being silently
        # ignored (GH #55).
        from its2s.cross_validation import time_series_cv
        df, intv, _ = make_daily_series(n_pre=730, n_post=180, seed=619)
        with pytest.raises(ValueError, match="test_pct"):
            time_series_cv(df, intv, model_name="arima",
                           n_folds=2, test_obs=60, min_train_obs=180,
                           test_pct=0.10,
                           config_overrides=self._CV_CFG)

    def test_obs_args_under_percent_raise(self):
        # The reverse direction: obs args under split_method="percent"
        # used to be overwritten by pct-derived values (GH #55).
        from its2s.cross_validation import time_series_cv
        df, intv, _ = make_daily_series(n_pre=730, n_post=180, seed=619)
        with pytest.raises(ValueError, match="min_train_obs"):
            time_series_cv(df, intv, model_name="arima",
                           n_folds=2, split_method="percent",
                           min_train_obs=180,
                           config_overrides=self._CV_CFG)

    def test_fold_result_fields(self):
        from its2s.cross_validation import time_series_cv, CVFoldResult
        df, intv, _ = make_daily_series(n_pre=730, n_post=180, seed=607)
        result = time_series_cv(df, intv, model_name="arima",
                                n_folds=2, test_obs=60, min_train_obs=180,
                                config_overrides=self._CV_CFG)
        for fold in result.folds:
            assert isinstance(fold, CVFoldResult)
            assert fold.n_train > 0
            assert fold.n_test > 0
            assert fold.train_end < fold.test_start

    # --- skip_obs: non-overlapping fold windows ---

    def test_skip_obs_zero_folds_are_adjacent(self):
        # With skip_obs=0, fold i+1 test starts exactly where fold i test ends.
        from its2s.cross_validation import time_series_cv
        df, intv, _ = make_daily_series(n_pre=730, n_post=180, seed=610)
        result = time_series_cv(df, intv, model_name="arima",
                                n_folds=3, test_obs=60, min_train_obs=180,
                                skip_obs=0, config_overrides=self._CV_CFG)
        for i in range(len(result.folds) - 1):
            gap = (result.folds[i + 1].test_start
                   - result.folds[i].test_end).days
            # Adjacent folds: gap should be exactly 1 day (end is inclusive,
            # start of next is the following day) or 0 if timestamps coincide.
            assert gap <= 1, f"Folds {i} and {i+1} have unexpected gap {gap}"

    def test_skip_obs_nonzero_enforces_gap(self):
        from its2s.cross_validation import time_series_cv
        skip = 30
        df, intv, _ = make_daily_series(n_pre=730, n_post=180, seed=611)
        result = time_series_cv(df, intv, model_name="arima",
                                n_folds=3, test_obs=60, min_train_obs=180,
                                skip_obs=skip, config_overrides=self._CV_CFG)
        for i in range(len(result.folds) - 1):
            gap = (result.folds[i + 1].test_start
                   - result.folds[i].test_end).days
            assert gap >= skip - 1, (
                f"Gap between folds {i} and {i+1} ({gap} days) "
                f"should be >= skip_obs ({skip})"
            )

    def test_skip_obs_folds_never_overlap(self):
        # No two fold test windows should share any dates.
        from its2s.cross_validation import time_series_cv
        df, intv, _ = make_daily_series(n_pre=730, n_post=180, seed=612)
        result = time_series_cv(df, intv, model_name="arima",
                                n_folds=4, test_obs=60, min_train_obs=180,
                                skip_obs=0, config_overrides=self._CV_CFG)
        for i in range(len(result.folds) - 1):
            assert result.folds[i].test_end < result.folds[i + 1].test_start, (
                f"Fold {i} test end ({result.folds[i].test_end}) "
                f">= fold {i+1} test start ({result.folds[i+1].test_start})"
            )

    # --- cv_end_date: scoping CV to training window ---

    def test_cv_end_date_excludes_later_data(self):
        # No fold test window should reach or exceed cv_end_date.
        from its2s.cross_validation import time_series_cv
        df, intv, _ = make_daily_series(n_pre=730, n_post=180, seed=613)
        cv_end = intv - pd.Timedelta(days=90)
        result = time_series_cv(df, intv, model_name="arima",
                                n_folds=3, test_obs=60, min_train_obs=180,
                                cv_end_date=cv_end,
                                config_overrides=self._CV_CFG)
        for fold in result.folds:
            assert fold.test_end < cv_end, (
                f"Fold test_end {fold.test_end} reached cv_end_date {cv_end}"
            )

    def test_cv_end_date_after_intervention_raises(self):
        from its2s.cross_validation import time_series_cv
        df, intv, _ = make_daily_series(n_pre=730, n_post=180, seed=614)
        with pytest.raises(ValueError, match="cv_end_date"):
            time_series_cv(df, intv, model_name="arima",
                           n_folds=2, test_obs=60, min_train_obs=180,
                           cv_end_date=intv + pd.Timedelta(days=10),
                           config_overrides=self._CV_CFG)

    def test_cv_end_date_none_derives_test_boundary(self):
        # Default (cv_end_date=None) derives the start of the held-out test
        # window from the run's split config (GH #40): identical to passing
        # that boundary explicitly, and NOT identical to using all
        # pre-intervention data.
        from its2s.cross_validation import time_series_cv
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_daily_series(n_pre=300, n_post=90, seed=615)
        boundary = prepare_splits(df, intv).test_df["ds"].min()
        r_none = time_series_cv(df, intv, model_name="arima",
                                n_folds=2, test_obs=60, min_train_obs=180,
                                cv_end_date=None,
                                config_overrides=self._CV_CFG)
        r_boundary = time_series_cv(df, intv, model_name="arima",
                                    n_folds=2, test_obs=60, min_train_obs=180,
                                    cv_end_date=boundary,
                                    config_overrides=self._CV_CFG)
        r_all_pre = time_series_cv(df, intv, model_name="arima",
                                   n_folds=2, test_obs=60, min_train_obs=180,
                                   cv_end_date=intv,
                                   config_overrides=self._CV_CFG)
        assert r_none.cv_end_date == boundary
        assert len(r_none.folds) == len(r_boundary.folds)
        assert abs(r_none.mean_rmse - r_boundary.mean_rmse) < 1e-9
        assert len(r_all_pre.folds) > len(r_none.folds)

    def test_cv_default_excludes_run_test_window(self):
        # With everything at defaults, no CV fold may touch the window
        # prepare_splits reserves for run_single_its evaluation (GH #40).
        from its2s.cross_validation import time_series_cv
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_daily_series(n_pre=300, n_post=90, seed=616)
        boundary = prepare_splits(df, intv).test_df["ds"].min()
        result = time_series_cv(df, intv, model_name="arima",
                                n_folds=2, test_obs=60, min_train_obs=180,
                                config_overrides=self._CV_CFG)
        for fold in result.folds:
            assert fold.test_end < boundary, (
                f"Fold test_end {fold.test_end} reached the run's held-out "
                f"test window starting {boundary}"
            )

    def test_cv_default_weekly_row_exact(self):
        # On a weekly grid the derived cap equals the run's test boundary
        # (row-exact) and DIFFERS from the naive calendar back-off
        # intervention - Timedelta(days=n_test) that GH #40 retired.
        from its2s.cross_validation import time_series_cv
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_weekly_series(n_pre_weeks=156, n_post_weeks=26,
                                         seed=617)
        splits = prepare_splits(df, intv)
        boundary = splits.test_df["ds"].min()
        n_test = len(splits.test_df)
        assert boundary != intv - pd.Timedelta(days=n_test)
        result = time_series_cv(df, intv, model_name="arima",
                                n_folds=2, test_obs=20, min_train_obs=60,
                                config_overrides=self._CV_CFG)
        assert result.cv_end_date == boundary
        for fold in result.folds:
            assert fold.test_end < boundary

    def test_cv_default_days_method_config(self):
        # A days-method periods config drives the derivation row-exactly on
        # a weekly grid: the cap is the first row of the calendar window.
        from its2s.cross_validation import time_series_cv
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_weekly_series(n_pre_weeks=156, n_post_weeks=26,
                                         seed=618)
        overrides = dict(self._CV_CFG)
        overrides["periods"] = {"split_method": "days",
                                "test_days": 180, "holdout_days": 60}
        boundary = prepare_splits(df, intv, split_method="days",
                                  test_days=180, holdout_days=60,
                                  min_test_obs=0).test_df["ds"].min()
        result = time_series_cv(df, intv, model_name="arima",
                                n_folds=2, test_obs=20, min_train_obs=60,
                                config_overrides=overrides)
        assert result.cv_end_date == boundary
        for fold in result.folds:
            assert fold.test_end < boundary

    def test_cv_end_date_explicit_intervention_reproduces_old_behavior(self):
        # The escape hatch: cv_end_date=intervention_date uses all
        # pre-intervention data, so a fold CAN land inside the run's
        # held-out test window (the pre-GH-#40 layout).
        from its2s.cross_validation import time_series_cv
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_daily_series(n_pre=300, n_post=90, seed=619)
        boundary = prepare_splits(df, intv).test_df["ds"].min()
        result = time_series_cv(df, intv, model_name="arima",
                                n_folds=2, test_obs=60, min_train_obs=180,
                                cv_end_date=intv,
                                config_overrides=self._CV_CFG)
        assert max(f.test_end for f in result.folds) >= boundary

    def test_cv_result_records_effective_cv_end_date(self):
        # CVResult carries the effective cap for both paths.
        from its2s.cross_validation import time_series_cv
        from its2s.data_prep import prepare_splits
        df, intv, _ = make_daily_series(n_pre=300, n_post=90, seed=621)
        r_derived = time_series_cv(df, intv, model_name="arima",
                                   n_folds=2, test_obs=60, min_train_obs=180,
                                   config_overrides=self._CV_CFG)
        assert (r_derived.cv_end_date
                == prepare_splits(df, intv).test_df["ds"].min())
        explicit = intv - pd.Timedelta(days=90)
        r_explicit = time_series_cv(df, intv, model_name="arima",
                                    n_folds=2, test_obs=60,
                                    min_train_obs=120,
                                    cv_end_date=explicit,
                                    config_overrides=self._CV_CFG)
        assert r_explicit.cv_end_date == explicit

    def test_insufficient_data_error_names_cv_end_date(self):
        # Fixed observation windows that fit the full pre period but not
        # the capped frame raise, and the message names the cap and the
        # escape hatch.
        from its2s.cross_validation import time_series_cv
        df, intv, _ = make_daily_series(n_pre=300, n_post=90, seed=622)
        with pytest.raises(ValueError, match="cv_end_date"):
            time_series_cv(df, intv, model_name="arima",
                           n_folds=2, test_obs=60, min_train_obs=200,
                           config_overrides=self._CV_CFG)

    # --- percent-based CV ---

    def test_cv_percent_basic(self):
        from its2s.cross_validation import time_series_cv
        df, intv, _ = make_daily_series(n_pre=200, n_post=50, seed=620)
        result = time_series_cv(df, intv, model_name="arima",
                                n_folds=3, split_method="percent",
                                test_pct=0.10, min_train_pct=0.50,
                                config_overrides=self._CV_CFG)
        assert len(result.folds) >= 1
        for fold in result.folds:
            assert fold.n_train > 0
            assert fold.n_test > 0

    def test_cv_percent_overflow_raises(self):
        from its2s.cross_validation import time_series_cv
        df, intv, _ = make_daily_series(n_pre=200, n_post=50, seed=621)
        with pytest.raises(ValueError, match="budget"):
            time_series_cv(df, intv, model_name="arima",
                           n_folds=6, split_method="percent",
                           test_pct=0.20, min_train_pct=0.50,
                           config_overrides=self._CV_CFG)


# ===================================================================
# Model Comparison
# ===================================================================
class TestCompare:
    """Unit tests for compare_models()."""

    _COMPARE_CFG = {
        "bootstrap": {"n_sim": 5, "n_jobs": 1},
        "periods": {"test_days": 30, "holdout_days": 30},
        "models": {"arima": {"seasonal": False, "m": 1, "stepwise": True,
                              "suppress_warnings": True}},
    }

    def test_returns_tuple_of_df_and_dict(self):
        from its2s.compare import compare_models
        df, intv, _ = make_short_series(n_pre=180, n_post=30, seed=700)
        out = _run_quiet(compare_models, df, intv,
                         model_names=["arima"],
                         config_overrides=self._COMPARE_CFG, seed=42)
        assert isinstance(out, tuple)
        assert len(out) == 2
        comparison_df, results_dict = out
        assert isinstance(comparison_df, pd.DataFrame)
        assert isinstance(results_dict, dict)

    def test_comparison_df_has_model_column(self):
        from its2s.compare import compare_models
        df, intv, _ = make_short_series(n_pre=180, n_post=30, seed=701)
        comparison_df, _ = _run_quiet(compare_models, df, intv,
                                       model_names=["arima"],
                                       config_overrides=self._COMPARE_CFG, seed=42)
        assert "model" in comparison_df.columns

    def test_results_dict_keys_match_model_names(self):
        from its2s.compare import compare_models
        df, intv, _ = make_short_series(n_pre=180, n_post=30, seed=702)
        _, results_dict = _run_quiet(compare_models, df, intv,
                                      model_names=["arima"],
                                      config_overrides=self._COMPARE_CFG, seed=42)
        assert "arima" in results_dict

    def test_two_model_comparison_has_two_rows(self):
        from its2s.compare import compare_models
        df, intv, _ = make_short_series(n_pre=180, n_post=30, seed=703)
        cfg = {
            **self._COMPARE_CFG,
            "models": {
                "arima": {"seasonal": False, "m": 1, "stepwise": True,
                          "suppress_warnings": True},
                "prophet_xgb": {},
            },
        }
        comparison_df, _ = _run_quiet(compare_models, df, intv,
                                       model_names=["arima", "prophet_xgb"],
                                       config_overrides=cfg, seed=42)
        assert len(comparison_df) == 2

    def test_comparison_df_has_metric_columns(self):
        from its2s.compare import compare_models
        df, intv, _ = make_short_series(n_pre=180, n_post=30, seed=704)
        comparison_df, _ = _run_quiet(compare_models, df, intv,
                                       model_names=["arima"],
                                       config_overrides=self._COMPARE_CFG, seed=42)
        for col in ("test_rmse", "test_mae", "test_r2"):
            assert col in comparison_df.columns


# ===================================================================
# Main entry point
# ===================================================================
if __name__ == "__main__":
    import sys
    exit_code = pytest.main([__file__, "-v", "--tb=short", "-x"] + sys.argv[1:])
    sys.exit(exit_code)
