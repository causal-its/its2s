# Description: Post-fit model diagnostics (residual checks).
#   Lag semantics are frequency-conditional (GH #61, #35): the full ACF vector
#   is persisted as the complete descriptive record, a minimal key-lag set
#   {1, m} carries the pre-specified inferential claims, and the Ljung-Box
#   depth follows the seasonal prescription min(2m, n // 5). m comes from the
#   same resolver mapping as the metrics (its2s.frequency), one concept in one
#   place.
# Usage: from its2s.diagnostics import compute_diagnostics
# Dependencies: numpy, scipy, statsmodels

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np

from .frequency import dominant_seasonal_period

logger = logging.getLogger(__name__)

# An ACF estimate at lag k uses n - k pairs; below this floor it stops being
# an estimate. Together with the n // 2 half-sample bound this caps the
# persisted vector: max_lag = min(n // 2, n - _MIN_ACF_PAIRS).
_MIN_ACF_PAIRS = 30

# Ljung-Box pooled depth when the frequency has no mapped seasonal period:
# the conventional non-seasonal prescription, still power-capped by n // 5.
_LB_NONSEASONAL_DEPTH = 10


@dataclass
class DiagnosticsResult:
    """Container for post-fit residual diagnostics.

    acf is the persisted lag-keyed ACF vector at lags 1..max_lag, the complete
    descriptive record. key_lags lists the pre-specified inferential lags
    ({1, m}); their values are an index into acf, not a second copy. A key lag
    the series is too short to estimate is present in acf as NaN. params
    carries n, max_lag, m, the frequency alias, and any fallback notes, so
    per-lag pair counts and every substitution are reconstructible. The acf
    vector is descriptive context, not a menu of hypothesis tests; the key
    lags are the pre-specified checks.
    """

    residual_mean: float
    residual_std: float
    acf: dict = field(default_factory=dict)
    key_lags: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    ljung_box_stat: float = np.nan
    ljung_box_pvalue: float = np.nan
    ljung_box_lags: int | None = None
    shapiro_stat: float | None = None
    shapiro_pvalue: float | None = None
    model_metadata: dict = field(default_factory=dict)


def _acf_at_lag(x, lag):
    if len(x) <= lag:
        return np.nan
    xm = x - np.mean(x)
    c0 = np.dot(xm, xm) / len(x)
    if c0 == 0:
        return 0.0
    ck = np.dot(xm[lag:], xm[:-lag]) / len(x)
    return float(ck / c0)


def compute_diagnostics(fit_result, model_name, series_freq, max_shapiro_n=5000):
    """Compute residual diagnostics from a fitted model result.

    Parameters
    ----------
    fit_result : FitResult
        Output of model.fit().
    model_name : str
        Name of the model (for metadata reporting).
    series_freq : SeriesFrequency or None
        The resolved series frequency (its2s.frequency.resolve_frequency).
        Determines the seasonal key lag and the Ljung-Box depth. None is
        accepted for frequencies without a mapped seasonal cycle and falls
        back loudly to non-seasonal semantics.
    max_shapiro_n : int
        Maximum sample size for Shapiro-Wilk test (too slow for large n).

    Returns
    -------
    DiagnosticsResult
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    residuals = fit_result.residuals
    residuals_clean = residuals[~np.isnan(residuals)]
    n = len(residuals_clean)

    # Basic statistics
    residual_mean = float(np.mean(residuals_clean))
    residual_std = float(np.std(residuals_clean, ddof=1)) if n > 1 else 0.0

    freq_alias = series_freq.alias if series_freq is not None else None
    m = dominant_seasonal_period(series_freq)
    notes = []
    if m is None:
        notes.append(
            f"No dominant seasonal period is mapped for series frequency "
            f"'{freq_alias}'; key lags reduce to {{1}} and the Ljung-Box "
            f"depth uses the non-seasonal prescription "
            f"min({_LB_NONSEASONAL_DEPTH}, n // 5)."
        )
        warnings.warn(
            f"Diagnostics [{model_name}]: {notes[-1]}",
            UserWarning,
            stacklevel=2,
        )

    # Full ACF vector: the complete descriptive record.
    max_lag = max(0, min(n // 2, n - _MIN_ACF_PAIRS))
    acf = {lag: _acf_at_lag(residuals_clean, lag)
           for lag in range(1, max_lag + 1)}

    # Key lags: the minimal pre-specified inferential set.
    key_lags = [1] if m is None else sorted({1, m})
    for lag in key_lags:
        if lag > max_lag:
            acf[lag] = np.nan
            reason = (
                f"key lag {lag} exceeds max_lag={max_lag} "
                f"(n={n} residuals): the series is too short to estimate "
                "autocorrelation at this lag."
            )
            notes.append(reason)
            warnings.warn(
                f"Diagnostics [{model_name}]: {reason}",
                UserWarning,
                stacklevel=2,
            )

    # Ljung-Box test for residual autocorrelation, at the seasonal pooled
    # depth min(2m, n // 5), power-capped.
    lb_target = _LB_NONSEASONAL_DEPTH if m is None else 2 * m
    lb_depth = None
    ljung_box_stat = np.nan
    ljung_box_pvalue = np.nan
    if n > 15:
        lb_depth = max(1, min(lb_target, n // 5))
        if m is not None and lb_depth < m:
            notes.append(
                f"Ljung-Box depth {lb_depth} (= n // 5 cap) is below the "
                f"seasonal period m={m}: seasonal-lag autocorrelation is "
                "outside the pooled window; the key-lag ACF carries the "
                "seasonal check."
            )
        try:
            lb_result = acorr_ljungbox(residuals_clean, lags=[lb_depth],
                                       return_df=True)
            ljung_box_stat = float(lb_result["lb_stat"].iloc[0])
            ljung_box_pvalue = float(lb_result["lb_pvalue"].iloc[0])
        except Exception:
            pass

    # Shapiro-Wilk test for normality (skip for large samples)
    shapiro_stat = None
    shapiro_pvalue = None
    if 3 <= n <= max_shapiro_n:
        from scipy.stats import shapiro
        try:
            stat, pval = shapiro(residuals_clean)
            shapiro_stat = float(stat)
            shapiro_pvalue = float(pval)
        except Exception:
            pass

    # Model-specific metadata
    metadata = dict(fit_result.metadata) if fit_result.metadata else {}
    metadata["model_name"] = model_name
    metadata["n_residuals"] = n

    result = DiagnosticsResult(
        residual_mean=residual_mean,
        residual_std=residual_std,
        acf=acf,
        key_lags=key_lags,
        params={
            "n": n,
            "max_lag": max_lag,
            "m": m,
            "freq_alias": freq_alias,
            "min_acf_pairs": _MIN_ACF_PAIRS,
            "notes": notes,
        },
        ljung_box_stat=ljung_box_stat,
        ljung_box_pvalue=ljung_box_pvalue,
        ljung_box_lags=lb_depth,
        shapiro_stat=shapiro_stat,
        shapiro_pvalue=shapiro_pvalue,
        model_metadata=metadata,
    )

    # Log key lags only, resolver-labeled; the full vector is persisted, not
    # narrated.
    key_str = ", ".join(f"acf[{lag}]={acf[lag]:.3f}" for lag in key_lags)
    logger.info(
        "Diagnostics [%s]: residual_mean=%.4f, residual_std=%.4f, %s "
        "(key lags from freq=%s, m=%s), LB(%s) p=%.4f",
        model_name, residual_mean, residual_std, key_str,
        freq_alias, m, lb_depth, ljung_box_pvalue,
    )
    if ljung_box_pvalue < 0.05:
        logger.warning(
            "Ljung-Box test (p=%.4f) suggests significant residual "
            "autocorrelation for model '%s'. Bootstrap CIs may undercover.",
            ljung_box_pvalue, model_name,
        )

    return result
