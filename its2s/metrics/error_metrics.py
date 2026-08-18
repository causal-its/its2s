# Description: Standard forecast error metrics (RMSE, MAE, MAPE, MASE).
#   Each reported metric has one job (GH #62): RMSE is accuracy on the mean and
#   the selection metric; MAE is native-units accuracy, robust; MAPE is
#   percentage communication, guarded near zero; MASE is the seasonal-naive
#   benchmark ratio, reported with its period m and denominator.
# Usage: from its2s.metrics.error_metrics import compute_metrics
# Dependencies: numpy

import warnings
from dataclasses import dataclass

import numpy as np

from ..frequency import dominant_seasonal_period


@dataclass
class MetricsResult:
    """Container for forecast error metrics.

    mase is the seasonal-naive benchmark ratio: model MAE over the in-sample
    MAE of the m-period seasonal naive. mase_m and mase_denominator (native
    units) are always emitted alongside it, since the ratio is meaningless
    without the benchmark it was scaled by.
    """

    rmse: float
    mae: float
    mape: float
    mase: float | None
    mase_m: int | None
    mase_denominator: float | None


def _safe_divide(num, denom):
    if denom == 0 or np.isnan(denom):
        return np.nan
    return num / denom


def resolve_metrics_seasonality(seasonality, n_train, series_freq=None):
    """Resolve the metrics.seasonality config into a usable period m.

    Parameters
    ----------
    seasonality : "auto" or int
        The config value. "auto" derives m from the resolved series frequency
        (daily 7, weekly 52, monthly 12); an integer is honored as given.
    n_train : int
        Number of training observations available to the seasonal-naive
        benchmark. The benchmark needs at least 2*m observations to produce
        a single error term from a full cycle.
    series_freq : SeriesFrequency, optional
        Required on the "auto" path; ignored for explicit integers.

    Returns
    -------
    int
        The period m.

    Raises
    ------
    ValueError
        If an explicit integer fails the n_train >= 2*m guard: the user named
        a benchmark, so it is never silently substituted.
    """
    if seasonality == "auto":
        m = dominant_seasonal_period(series_freq)
        if m is None:
            alias = series_freq.alias if series_freq is not None else "unknown"
            warnings.warn(
                f"metrics.seasonality='auto': no dominant seasonal period is "
                f"mapped for series frequency '{alias}'. Falling back to m=1 "
                "(plain naive benchmark). Set metrics.seasonality to an "
                "integer to name the benchmark period explicitly.",
                UserWarning,
                stacklevel=2,
            )
            return 1
        if n_train < 2 * m:
            warnings.warn(
                f"metrics.seasonality='auto': the seasonal-naive benchmark "
                f"needs n_train >= 2*m ({2 * m}) but only {n_train} training "
                f"observations are available. Falling back to m=1 (plain "
                "naive benchmark).",
                UserWarning,
                stacklevel=2,
            )
            return 1
        return m

    m = int(seasonality)
    if m < 1:
        raise ValueError(f"metrics.seasonality must be >= 1, got {m}.")
    if n_train < 2 * m:
        raise ValueError(
            f"metrics.seasonality={m} requires n_train >= 2*m ({2 * m}), but "
            f"only {n_train} training observations are available. A benchmark "
            "you named is never silently substituted; reduce the period or "
            "set metrics.seasonality to 'auto'."
        )
    return m


def seasonal_naive_mae(training_actual, m):
    """In-sample MAE of the m-period seasonal-naive forecast (native units)."""
    training_actual = np.asarray(training_actual, dtype=float)
    if len(training_actual) <= m:
        return np.nan
    naive_errors = np.abs(training_actual[m:] - training_actual[:-m])
    return float(np.nanmean(naive_errors))


def calc_mase(actual, predicted, training_actual, seasonality):
    """Calculate Mean Absolute Scaled Error against the m-period seasonal naive.

    Parameters
    ----------
    actual : array-like
    predicted : array-like
    training_actual : array-like
        Training set actuals for computing naive seasonal forecast errors.
    seasonality : int
        Seasonal period m for the naive benchmark. No default: m is resolved
        upstream (resolve_metrics_seasonality), never assumed daily-shaped.

    Returns
    -------
    float
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae_model = np.nanmean(np.abs(actual - predicted))
    mae_naive = seasonal_naive_mae(training_actual, seasonality)

    return _safe_divide(mae_model, mae_naive)


def compute_metrics(actual, predicted, training_actual=None, seasonality=None):
    """Compute the forecast error metric suite.

    Parameters
    ----------
    actual : array-like
    predicted : array-like
    training_actual : array-like, optional
        Required for MASE. If None, MASE and its denominator are None.
    seasonality : int, optional
        Seasonal period m for the MASE benchmark. Required when
        training_actual is given; resolve it with
        resolve_metrics_seasonality rather than passing a raw config value.

    Returns
    -------
    MetricsResult
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = actual - predicted
    abs_errors = np.abs(errors)

    rmse = float(np.sqrt(np.nanmean(errors ** 2)))
    mae = float(np.nanmean(abs_errors))

    # MAPE is undefined at zero actuals. Skipping zeros would silently drop
    # the hardest observations, so the metric goes NaN loudly instead.
    n_zero = int(np.sum(actual == 0))
    if n_zero > 0:
        warnings.warn(
            f"MAPE is undefined: {n_zero} of {len(actual)} actual values are "
            "zero. Emitting NaN; use MAE or MASE on series with zero counts.",
            UserWarning,
            stacklevel=2,
        )
        mape = np.nan
    else:
        mape = float(np.nanmean(np.abs(errors / actual)) * 100)

    mase = None
    mase_denominator = None
    if training_actual is not None:
        if seasonality is None:
            raise ValueError(
                "compute_metrics requires seasonality (the period m) when "
                "training_actual is given; resolve it with "
                "resolve_metrics_seasonality."
            )
        mase = float(calc_mase(actual, predicted, training_actual, seasonality))
        mase_denominator = seasonal_naive_mae(training_actual, seasonality)

    return MetricsResult(
        rmse=rmse,
        mae=mae,
        mape=mape,
        mase=mase,
        mase_m=int(seasonality) if seasonality is not None else None,
        mase_denominator=mase_denominator,
    )
