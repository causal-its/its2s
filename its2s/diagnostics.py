# Description: Post-fit model diagnostics (residual checks).
# Usage: from its2s.diagnostics import compute_diagnostics
# Dependencies: numpy, scipy, statsmodels

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticsResult:
    """Container for post-fit residual diagnostics."""

    residual_mean: float
    residual_std: float
    acf_lag1: float
    acf_lag7: float
    acf_lag14: float
    ljung_box_stat: float
    ljung_box_pvalue: float
    shapiro_stat: float | None
    shapiro_pvalue: float | None
    model_metadata: dict = field(default_factory=dict)


def compute_diagnostics(fit_result, model_name, max_shapiro_n=5000):
    """Compute residual diagnostics from a fitted model result.

    Parameters
    ----------
    fit_result : FitResult
        Output of model.fit().
    model_name : str
        Name of the model (for metadata reporting).
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

    # Autocorrelation at key lags
    def _acf_at_lag(x, lag):
        if len(x) <= lag:
            return np.nan
        xm = x - np.mean(x)
        c0 = np.dot(xm, xm) / len(x)
        if c0 == 0:
            return 0.0
        ck = np.dot(xm[lag:], xm[:-lag]) / len(x)
        return float(ck / c0)

    acf_lag1 = _acf_at_lag(residuals_clean, 1)
    acf_lag7 = _acf_at_lag(residuals_clean, 7)
    acf_lag14 = _acf_at_lag(residuals_clean, 14)

    # Ljung-Box test for residual autocorrelation
    ljung_box_stat = np.nan
    ljung_box_pvalue = np.nan
    if n > 15:
        try:
            lb_result = acorr_ljungbox(residuals_clean, lags=[10],
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
        acf_lag1=acf_lag1,
        acf_lag7=acf_lag7,
        acf_lag14=acf_lag14,
        ljung_box_stat=ljung_box_stat,
        ljung_box_pvalue=ljung_box_pvalue,
        shapiro_stat=shapiro_stat,
        shapiro_pvalue=shapiro_pvalue,
        model_metadata=metadata,
    )

    # Log key diagnostics
    logger.info(
        "Diagnostics [%s]: residual_mean=%.4f, residual_std=%.4f, "
        "acf_lag1=%.3f, acf_lag7=%.3f, LB(10) p=%.4f",
        model_name, residual_mean, residual_std,
        acf_lag1, acf_lag7, ljung_box_pvalue,
    )
    if ljung_box_pvalue < 0.05:
        logger.warning(
            "Ljung-Box test (p=%.4f) suggests significant residual "
            "autocorrelation for model '%s'. Bootstrap CIs may undercover.",
            ljung_box_pvalue, model_name,
        )

    return result
