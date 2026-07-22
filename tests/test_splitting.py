# Description: Unit tests for percent-based and day-based splitting in data_prep.
# Usage: python -m pytest tests/test_splitting.py -v --tb=short
# Dependencies: pytest, numpy, pandas, its2s

import numpy as np
import pandas as pd
import pytest

from its2s.data_prep import prepare_splits


def _make_series(n_pre, n_post, start="2020-01-01", seed=0):
    n = n_pre + n_post
    dates = pd.date_range(start, periods=n, freq="D")
    rng = np.random.default_rng(seed)
    y = 100 + rng.normal(0, 1, n)
    df = pd.DataFrame({"ds": dates, "y": y})
    intervention_date = dates[n_pre]
    return df, intervention_date


def test_split_by_percent_basic():
    df, intv = _make_series(n_pre=100, n_post=100)  # 200-day total, intv at day 100
    splits = prepare_splits(df, intv, split_method="percent",
                            test_pct=0.20, holdout_pct=1.0)
    # 20% of 100 pre-intv rows = 20 test days; train = 80 rows
    assert len(splits.test_df) == 20
    assert len(splits.train_df) == 80
    assert len(splits.holdout_df) == 100  # intervention day inclusive of post


def test_split_method_dispatch():
    df, intv = _make_series(n_pre=200, n_post=100)
    pct = prepare_splits(df, intv, split_method="percent",
                         test_pct=0.25, holdout_pct=1.0)
    days = prepare_splits(df, intv, split_method="days",
                          test_days=50, holdout_days=100)
    # 25% of 200 = 50 days, equivalent to days=50
    assert len(pct.test_df) == len(days.test_df) == 50


def test_split_validation_pct_out_of_range():
    df, intv = _make_series(n_pre=100, n_post=50)
    with pytest.raises(ValueError):
        # Out-of-range test_pct triggers via validate_inputs in pipeline; here
        # we call run_single_its to get the validation path.
        from its2s import run_single_its
        run_single_its(df, intv,
                       config_overrides={
                           "periods": {"split_method": "percent",
                                       "test_pct": 1.2,
                                       "holdout_pct": 1.0},
                           "bootstrap": {"n_sim": 5},
                       })


def test_split_method_days_empty_train_raises():
    df, intv = _make_series(n_pre=365, n_post=50)
    from its2s import run_single_its
    with pytest.raises(ValueError, match="test_days"):
        run_single_its(df, intv,
                       config_overrides={
                           "periods": {"split_method": "days",
                                       "test_days": 400,
                                       "holdout_days": 30},
                           "bootstrap": {"n_sim": 5},
                       })


def test_split_default_short_series_no_empty_splits():
    """Issue 2.3: with percent defaults, a 100/50 series must not break."""
    df, intv = _make_series(n_pre=100, n_post=50)
    splits = prepare_splits(df, intv)  # defaults: percent, test_pct=0.20
    assert len(splits.train_df) > 0
    assert len(splits.test_df) > 0
    assert len(splits.holdout_df) > 0


def _make_regular_series(n_pre, n_post, freq, start="2020-01-06", seed=0):
    n = n_pre + n_post
    dates = pd.date_range(start, periods=n, freq=freq)
    rng = np.random.default_rng(seed)
    y = 100 + rng.normal(0, 1, n)
    df = pd.DataFrame({"ds": dates, "y": y})
    return df, dates[n_pre]


def test_split_percent_weekly_counts_observations():
    """Issue #28: the percent path sizes windows in observations, not days."""
    df, intv = _make_regular_series(n_pre=181, n_post=52, freq="W-MON")
    splits = prepare_splits(df, intv, split_method="percent",
                            test_pct=0.20, holdout_pct=1.0)
    assert len(splits.test_df) == 36   # round(0.20 * 181), was 5 under #28
    assert len(splits.train_df) == 145
    assert len(splits.holdout_df) == 52  # all post rows, was ~8 under #28
    assert len(splits.full_predict_df) == 36 + 52


def test_split_method_observations():
    df, intv = _make_regular_series(n_pre=181, n_post=52, freq="W-MON")
    splits = prepare_splits(df, intv, split_method="observations",
                            test_obs=36, holdout_obs=52)
    assert len(splits.test_df) == 36
    assert len(splits.train_df) == 145
    assert len(splits.holdout_df) == 52


def test_split_method_observations_requires_explicit_counts():
    df, intv = _make_series(n_pre=100, n_post=50)
    with pytest.raises(ValueError, match="test_obs"):
        prepare_splits(df, intv, split_method="observations")


def test_split_cross_method_args_raise():
    """Issue #54: arguments for another split method are never silently ignored."""
    df, intv = _make_series(n_pre=100, n_post=50)
    with pytest.raises(ValueError, match="test_days"):
        prepare_splits(df, intv, test_days=30, holdout_days=30)  # percent default
    with pytest.raises(ValueError, match="test_pct"):
        prepare_splits(df, intv, split_method="days", test_pct=0.20)
    with pytest.raises(ValueError, match="test_obs"):
        prepare_splits(df, intv, split_method="percent", test_obs=20)


def test_validate_days_weekly_within_span_passes():
    """Issue #49 false positive: 252 days is valid on a 180-week pre span."""
    from its2s.validation import validate_inputs
    df, intv = _make_regular_series(n_pre=181, n_post=52, freq="W-MON")
    validate_inputs(df, intv, "ds", "y", None, "prophet_xgb",
                    split_method="days", test_days=252, holdout_days=52)


def test_validate_days_beyond_span_raises():
    """Issue #49 false negative: 30 days exceeds ~8.3 days of hourly data."""
    from its2s.validation import validate_inputs
    df, intv = _make_regular_series(n_pre=200, n_post=24, freq="h")
    with pytest.raises(ValueError, match="test_days"):
        validate_inputs(df, intv, "ds", "y", None, "prophet_xgb",
                        split_method="days", test_days=30, holdout_days=1)


def test_validate_observations_empty_train_raises():
    from its2s.validation import validate_inputs
    df, intv = _make_series(n_pre=100, n_post=50)
    with pytest.raises(ValueError, match="test_obs"):
        validate_inputs(df, intv, "ds", "y", None, "prophet_xgb",
                        split_method="observations", test_obs=100,
                        holdout_obs=50)


def test_min_test_obs_warns_on_small_window():
    """Issue #29: a degenerate test window warns whatever its cause."""
    df, intv = _make_series(n_pre=100, n_post=50)
    with pytest.warns(UserWarning, match="min_test_obs"):
        prepare_splits(df, intv, split_method="percent",
                       test_pct=0.10)  # 10 obs < default threshold 30


def test_min_test_obs_silent_at_threshold():
    import warnings
    df, intv = _make_series(n_pre=150, n_post=50)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prepare_splits(df, intv, split_method="percent",
                       test_pct=0.20)  # exactly 30 obs, no warning


def test_min_test_obs_zero_disables():
    import warnings
    df, intv = _make_series(n_pre=100, n_post=50)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        prepare_splits(df, intv, split_method="percent",
                       test_pct=0.10, min_test_obs=0)
