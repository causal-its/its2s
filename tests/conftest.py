# Description: Shared fixtures and synthetic data generators for its2s test suite.
# Usage: pytest discovers this automatically.
# Dependencies: numpy, pandas, pytest

import warnings

import numpy as np
import pandas as pd
import pytest

from its2s.settings import load_config


# ---------------------------------------------------------------------------
# Shared helpers (imported by test_its2s.py and test_models.py)
# ---------------------------------------------------------------------------

def _has_neuralprophet():
    """Return True if neuralprophet and its dependencies are importable."""
    try:
        import neuralprophet  # noqa: F401
        return True
    except ImportError:
        return False


def _run_quiet(func, *args, **kwargs):
    """Call func with all warnings suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(*args, **kwargs)


# Fast configuration constants used across test files.
_NP_FAST_PARAMS = {
    "n_lags": 7,
    "yearly_seasonality": False,
    "weekly_seasonality": False,
    "epochs": 5,
    "batch_size": 32,
    "learning_rate": 0.01,
}

_FAST = {"bootstrap": {"n_sim": 10, "n_jobs": 1}}
_NP_FAST = {**_FAST, "models": {"neuralprophet": _NP_FAST_PARAMS}}


def collect_model_params():
    """Return (model_name, ModelClass, init_params) for all available models.

    NeuralProphet is included only when its dependencies are installed.
    Call at module level in test files to build pytest.mark.parametrize lists.
    """
    from its2s.models.arima import ARIMAModel
    from its2s.models.prophet_xgb import ProphetXGBHybridModel
    from its2s.models.prophet_then_xgb import ProphetThenXGBModel

    params = [
        ("arima", ARIMAModel,
         {"seasonal": False, "m": 1, "stepwise": True, "suppress_warnings": True}),
        ("prophet_xgb", ProphetXGBHybridModel, {}),
        ("prophet_then_xgb", ProphetThenXGBModel, {}),
    ]
    if _has_neuralprophet():
        from its2s.models.neuralprophet import NeuralProphetModel
        params.append(("neuralprophet", NeuralProphetModel, _NP_FAST_PARAMS))
    return params


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def make_daily_series(n_pre=1095, n_post=365, intervention_effect=0.0,
                      trend=0.01, noise_sd=5.0, seasonal_amplitude=10.0,
                      seasonal_period=365, base_level=100.0, seed=42):
    """Daily time series with optional intervention shift."""
    rng = np.random.default_rng(seed)
    n = n_pre + n_post
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)
    y = (base_level
         + trend * t
         + seasonal_amplitude * np.sin(2 * np.pi * t / seasonal_period)
         + rng.normal(0, noise_sd, n))
    y[n_pre:] += intervention_effect
    intervention_date = dates[n_pre]
    df = pd.DataFrame({"ds": dates, "y": y})
    return df, intervention_date, intervention_effect


def make_weekly_series(n_pre_weeks=156, n_post_weeks=52,
                       intervention_effect=0.0, seed=42):
    """Weekly resolution time series."""
    rng = np.random.default_rng(seed)
    n = n_pre_weeks + n_post_weeks
    dates = pd.date_range("2018-01-01", periods=n, freq="W-MON")
    t = np.arange(n, dtype=float)
    y = 50 + 0.05 * t + 8 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 3, n)
    y[n_pre_weeks:] += intervention_effect
    intervention_date = dates[n_pre_weeks]
    df = pd.DataFrame({"ds": dates, "y": y})
    return df, intervention_date, intervention_effect


def make_monthly_series(n_pre_months=48, n_post_months=12,
                        intervention_effect=0.0, seed=42):
    """Monthly resolution time series."""
    rng = np.random.default_rng(seed)
    n = n_pre_months + n_post_months
    dates = pd.date_range("2018-01-01", periods=n, freq="MS")
    t = np.arange(n, dtype=float)
    y = 200 + 0.5 * t + 20 * np.sin(2 * np.pi * t / 12) + rng.normal(0, 8, n)
    y[n_pre_months:] += intervention_effect
    intervention_date = dates[n_pre_months]
    df = pd.DataFrame({"ds": dates, "y": y})
    return df, intervention_date, intervention_effect


def make_quarterly_series(n_pre_q=20, n_post_q=4,
                          intervention_effect=0.0, seed=42):
    """Quarterly resolution time series."""
    rng = np.random.default_rng(seed)
    n = n_pre_q + n_post_q
    dates = pd.date_range("2013-01-01", periods=n, freq="QS")
    t = np.arange(n, dtype=float)
    y = 300 + 1.0 * t + 15 * np.sin(2 * np.pi * t / 4) + rng.normal(0, 10, n)
    y[n_pre_q:] += intervention_effect
    intervention_date = dates[n_pre_q]
    df = pd.DataFrame({"ds": dates, "y": y})
    return df, intervention_date, intervention_effect


def make_count_series(n_pre=1095, n_post=365, intervention_effect=0.0, seed=42):
    """Poisson count data (non-negative integers)."""
    rng = np.random.default_rng(seed)
    n = n_pre + n_post
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)
    lam = 50 + 0.005 * t + 10 * np.sin(2 * np.pi * t / 365)
    lam[n_pre:] += intervention_effect
    lam = np.clip(lam, 1, None)
    y = rng.poisson(lam).astype(float)
    intervention_date = dates[n_pre]
    df = pd.DataFrame({"ds": dates, "y": y})
    return df, intervention_date, intervention_effect


def make_rate_series(n_pre=1095, n_post=365, intervention_effect=0.0, seed=42):
    """Rate/proportion data in [0, 1] via logistic transform."""
    rng = np.random.default_rng(seed)
    n = n_pre + n_post
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)
    logit = -0.5 + 0.0002 * t + 0.3 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 0.2, n)
    logit[n_pre:] += intervention_effect
    y = 1.0 / (1.0 + np.exp(-logit))
    intervention_date = dates[n_pre]
    df = pd.DataFrame({"ds": dates, "y": y})
    return df, intervention_date, intervention_effect


def make_series_with_covariates(n_pre=1095, n_post=365,
                                intervention_effect=5.0, seed=42):
    """Daily series with two covariates: temperature and holiday."""
    rng = np.random.default_rng(seed)
    n = n_pre + n_post
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)
    temperature = 15 + 10 * np.sin(2 * np.pi * t / 365) + rng.normal(0, 2, n)
    holiday = np.zeros(n)
    holiday_idx = rng.choice(n, size=int(n * 10 / 365), replace=False)
    holiday[holiday_idx] = 1.0
    y = (100 + 0.01 * t
         + 10 * np.sin(2 * np.pi * t / 365)
         + 2.0 * temperature
         + 15.0 * holiday
         + rng.normal(0, 5, n))
    y[n_pre:] += intervention_effect
    intervention_date = dates[n_pre]
    df = pd.DataFrame({
        "ds": dates, "y": y,
        "temperature": temperature, "holiday": holiday,
    })
    return df, intervention_date, intervention_effect, ["temperature", "holiday"]


def make_short_series(n_pre=90, n_post=30, intervention_effect=5.0, seed=42):
    """Short daily series without strong seasonality."""
    rng = np.random.default_rng(seed)
    n = n_pre + n_post
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    t = np.arange(n, dtype=float)
    y = 50 + 0.05 * t + rng.normal(0, 3, n)
    y[n_pre:] += intervention_effect
    intervention_date = dates[n_pre]
    df = pd.DataFrame({"ds": dates, "y": y})
    return df, intervention_date, intervention_effect


def make_missing_data_series(n_pre=1095, n_post=365, frac_missing=0.05,
                             intervention_effect=10.0, seed=42):
    """Daily series with NaN values injected at random."""
    df, intervention_date, effect = make_daily_series(
        n_pre=n_pre, n_post=n_post, intervention_effect=intervention_effect,
        seed=seed,
    )
    rng = np.random.default_rng(seed + 1)
    n = len(df)
    nan_idx = rng.choice(n, size=int(n * frac_missing), replace=False)
    df.loc[nan_idx, "y"] = np.nan
    return df, intervention_date, effect


def make_constant_series(n_pre=365, n_post=365, value=50.0, seed=42):
    """All-constant outcome."""
    n = n_pre + n_post
    dates = pd.date_range("2019-01-01", periods=n, freq="D")
    df = pd.DataFrame({"ds": dates, "y": np.full(n, value)})
    intervention_date = dates[n_pre]
    return df, intervention_date, 0.0


def make_outlier_series(n_pre=1095, n_post=365, n_outliers=10, seed=42):
    """Daily series with extreme outlier spikes in pre-intervention period."""
    df, intervention_date, effect = make_daily_series(
        n_pre=n_pre, n_post=n_post, intervention_effect=10.0, seed=seed,
    )
    rng = np.random.default_rng(seed + 2)
    outlier_idx = rng.choice(n_pre, size=n_outliers, replace=False)
    df.loc[outlier_idx, "y"] *= 10
    return df, intervention_date, effect


def make_intervention_at_boundary(position="start", seed=42):
    """Series where intervention is at the very start or end."""
    n = 500
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2019-01-01", periods=n, freq="D")
    y = 100 + 0.01 * np.arange(n) + rng.normal(0, 3, n)
    df = pd.DataFrame({"ds": dates, "y": y})
    if position == "start":
        return df, dates[0]
    else:
        return df, dates[-1]


# ---------------------------------------------------------------------------
# Helper: mock BootstrapCIResult for unit-testing excess/output modules
# ---------------------------------------------------------------------------

def make_mock_bootstrap_result(n_dates=60, n_sim=10, intervention_idx=30,
                               base_predicted=100.0, actual_shift=10.0, seed=42,
                               freq="D"):
    """Create a synthetic BootstrapCIResult without running MBB."""
    from its2s.bootstrap.base import BootstrapCIResult

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n_dates, freq=freq)
    predicted = np.full(n_dates, base_predicted) + rng.normal(0, 2, n_dates)
    actual = predicted.copy()
    actual[intervention_idx:] += actual_shift
    pred_matrix = predicted[:, None] + rng.normal(0, 3, (n_dates, n_sim))
    conf_lo = np.percentile(pred_matrix, 2.5, axis=1)
    conf_hi = np.percentile(pred_matrix, 97.5, axis=1)
    return BootstrapCIResult(
        dates=dates.values,
        actual=actual,
        predicted=predicted,
        conf_lo=conf_lo,
        conf_hi=conf_hi,
        pred_matrix=pred_matrix,
        n_successful=n_sim,
        ci_method="quantile",
        ci_level=0.95,
    )


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def default_config():
    return load_config()


@pytest.fixture(scope="session")
def fast_config():
    return load_config(overrides={"bootstrap": {"n_sim": 10, "n_jobs": 1}})


@pytest.fixture(scope="session")
def daily_series():
    return make_daily_series(intervention_effect=10.0)


@pytest.fixture(scope="session")
def daily_null_series():
    return make_daily_series(intervention_effect=0.0, seed=99)


@pytest.fixture(scope="session")
def weekly_series():
    return make_weekly_series(intervention_effect=5.0)


@pytest.fixture(scope="session")
def monthly_series():
    return make_monthly_series(intervention_effect=10.0)


@pytest.fixture(scope="session")
def quarterly_series():
    return make_quarterly_series(intervention_effect=15.0)


@pytest.fixture(scope="session")
def short_series():
    return make_short_series()


@pytest.fixture(scope="session")
def covariate_series():
    return make_series_with_covariates(intervention_effect=5.0)
