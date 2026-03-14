# Description: Standard forecast error metrics (RMSE, MAE, MAPE, SMAPE, MASE, R2).
# Usage: from its2s.metrics.error_metrics import compute_metrics
# Dependencies: numpy

from dataclasses import dataclass

import numpy as np


@dataclass
class MetricsResult:
    """Container for forecast error metrics."""

    rmse: float
    mae: float
    mape: float
    smape: float
    mase: float | None
    r2: float


def _safe_divide(num, denom):
    if denom == 0 or np.isnan(denom):
        return np.nan
    return num / denom


def calc_mase(actual, predicted, training_actual, seasonality=7):
    """Calculate Mean Absolute Scaled Error.

    Parameters
    ----------
    actual : array-like
    predicted : array-like
    training_actual : array-like
        Training set actuals for computing naive seasonal forecast errors.
    seasonality : int
        Seasonal period for the naive forecast.

    Returns
    -------
    float
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    training_actual = np.asarray(training_actual, dtype=float)

    mae_model = np.nanmean(np.abs(actual - predicted))
    naive_errors = np.abs(training_actual[seasonality:] - training_actual[:-seasonality])
    mae_naive = np.nanmean(naive_errors)

    return _safe_divide(mae_model, mae_naive)


def compute_metrics(actual, predicted, training_actual=None, seasonality=7):
    """Compute a suite of forecast error metrics.

    Parameters
    ----------
    actual : array-like
    predicted : array-like
    training_actual : array-like, optional
        Required for MASE. If None, MASE is set to None.
    seasonality : int
        Seasonal period for MASE naive forecast.

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

    # MAPE -- skip zeros in actual
    mask = actual != 0
    mape = float(np.nanmean(np.abs(errors[mask] / actual[mask])) * 100) if mask.any() else np.nan

    # SMAPE
    denom = np.abs(actual) + np.abs(predicted)
    smape_vals = np.where(denom > 0, 2 * abs_errors / denom, 0)
    smape = float(np.nanmean(smape_vals) * 100)

    # MASE
    mase = None
    if training_actual is not None:
        mase = float(calc_mase(actual, predicted, training_actual, seasonality))

    # R2
    ss_res = np.nansum(errors ** 2)
    ss_tot = np.nansum((actual - np.nanmean(actual)) ** 2)
    r2 = float(1 - _safe_divide(ss_res, ss_tot))

    return MetricsResult(rmse=rmse, mae=mae, mape=mape, smape=smape, mase=mase, r2=r2)
