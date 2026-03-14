# Description: Excess calculation from MBB bootstrap results with CIs.
# Usage: from its2s.metrics.excess import calculate_excess, calc_ate_summary
# Dependencies: numpy, pandas

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ExcessResult:
    """Container for excess estimates."""

    daily_excess: pd.DataFrame
    period_excess: pd.DataFrame


def calculate_excess(bootstrap_result, intervention_date, periods_config=None,
                     ci_level=0.95):
    """Calculate daily and period-level excess from bootstrap results.

    Parameters
    ----------
    bootstrap_result : BootstrapCIResult
        Output of MBB bootstrap.
    intervention_date : pd.Timestamp
        Intervention date (only holdout dates are used for excess).
    periods_config : list[dict], optional
        Custom sub-periods, each with 'name', 'start_offset', 'end_offset'.
    ci_level : float
        Confidence level for CIs.

    Returns
    -------
    ExcessResult
    """
    alpha = 1 - ci_level
    lo_q = alpha / 2
    hi_q = 1 - alpha / 2

    dates = pd.to_datetime(bootstrap_result.dates)
    intervention_date = pd.Timestamp(intervention_date)

    # Restrict to holdout (post-intervention)
    holdout_mask = dates >= intervention_date
    if not holdout_mask.any():
        return ExcessResult(daily_excess=pd.DataFrame(), period_excess=pd.DataFrame())

    h_dates = dates[holdout_mask]
    h_actual = bootstrap_result.actual[holdout_mask] if bootstrap_result.actual is not None else None
    h_predicted = bootstrap_result.predicted[holdout_mask]
    h_pred_matrix = bootstrap_result.pred_matrix[holdout_mask, :]
    h_conf_lo = bootstrap_result.conf_lo[holdout_mask]
    h_conf_hi = bootstrap_result.conf_hi[holdout_mask]

    # Daily excess
    daily_rows = []
    for i in range(len(h_dates)):
        observed = float(h_actual[i]) if h_actual is not None else np.nan
        expected = float(h_predicted[i])
        excess = observed - expected

        # Excess CI from bootstrap: observed - sim_predicted for each sim
        if h_actual is not None:
            excess_sims = observed - h_pred_matrix[i, :]
            excess_lo = float(np.nanpercentile(excess_sims, 100 * lo_q))
            excess_hi = float(np.nanpercentile(excess_sims, 100 * hi_q))
        else:
            excess_lo = excess_hi = np.nan

        excess_pct = (excess / expected * 100) if expected != 0 else np.nan
        if h_actual is not None and expected != 0:
            pct_sims = excess_sims / expected * 100
            excess_pct_lo = float(np.nanpercentile(pct_sims, 100 * lo_q))
            excess_pct_hi = float(np.nanpercentile(pct_sims, 100 * hi_q))
        else:
            excess_pct_lo = excess_pct_hi = np.nan

        daily_rows.append({
            "date": h_dates.iloc[i] if hasattr(h_dates, "iloc") else h_dates[i],
            "observed": observed,
            "expected": expected,
            "expected_ci_lo": float(h_conf_lo[i]),
            "expected_ci_hi": float(h_conf_hi[i]),
            "excess": excess,
            "excess_ci_lo": excess_lo,
            "excess_ci_hi": excess_hi,
            "excess_pct": excess_pct,
            "excess_pct_ci_lo": excess_pct_lo,
            "excess_pct_ci_hi": excess_pct_hi,
        })

    daily_excess = pd.DataFrame(daily_rows)

    # Period-level excess
    period_rows = []

    # Default: full holdout period
    all_periods = [{"name": "Full holdout", "start_offset": 0, "end_offset": None}]
    if periods_config:
        all_periods.extend(periods_config)

    holdout_start = h_dates.min() if hasattr(h_dates, "min") else h_dates[0]

    for pconf in all_periods:
        p_start = holdout_start + pd.Timedelta(days=pconf.get("start_offset", 0))
        if pconf.get("end_offset") is not None:
            p_end = holdout_start + pd.Timedelta(days=pconf["end_offset"])
        else:
            p_end = h_dates.max() if hasattr(h_dates, "max") else h_dates[-1]

        p_mask = (h_dates >= p_start) & (h_dates <= p_end)
        if not p_mask.any():
            continue

        p_actual = h_actual[p_mask] if h_actual is not None else None
        p_predicted = h_predicted[p_mask]
        p_pred_matrix = h_pred_matrix[p_mask, :]

        total_observed = float(np.nansum(p_actual)) if p_actual is not None else np.nan
        total_expected = float(np.nansum(p_predicted))
        total_excess = total_observed - total_expected

        # Period CIs: sum across dates per simulation, then percentile
        if p_actual is not None:
            sim_totals = total_observed - np.nansum(p_pred_matrix, axis=0)
            excess_lo = float(np.nanpercentile(sim_totals, 100 * lo_q))
            excess_hi = float(np.nanpercentile(sim_totals, 100 * hi_q))
        else:
            excess_lo = excess_hi = np.nan

        excess_pct = (total_excess / total_expected * 100) if total_expected != 0 else np.nan

        period_rows.append({
            "period": pconf["name"],
            "start_date": p_start,
            "end_date": p_end,
            "n_days": int(p_mask.sum()),
            "total_observed": total_observed,
            "total_expected": total_expected,
            "total_excess": total_excess,
            "excess_ci_lo": excess_lo,
            "excess_ci_hi": excess_hi,
            "excess_pct": excess_pct,
        })

    period_excess = pd.DataFrame(period_rows)

    return ExcessResult(daily_excess=daily_excess, period_excess=period_excess)


def calc_ate_summary(daily_excess):
    """Calculate Average Treatment Effect summary from daily excess.

    Parameters
    ----------
    daily_excess : pd.DataFrame
        Daily excess DataFrame from calculate_excess.

    Returns
    -------
    pd.DataFrame
        Summary with total ATE and mean daily ATE.
    """
    if daily_excess.empty:
        return pd.DataFrame()

    n = len(daily_excess)
    total_excess = daily_excess["excess"].sum()
    mean_daily = total_excess / n

    # SE from excess CIs (approximate)
    total_ci_width = daily_excess["excess_ci_hi"].sum() - daily_excess["excess_ci_lo"].sum()
    se_total = total_ci_width / (2 * 1.96)
    se_daily = se_total / n

    return pd.DataFrame([
        {
            "metric": "Total ATE",
            "estimate": total_excess,
            "se": se_total,
            "ci_lo": daily_excess["excess_ci_lo"].sum(),
            "ci_hi": daily_excess["excess_ci_hi"].sum(),
            "n_days": n,
        },
        {
            "metric": "Mean Daily ATE",
            "estimate": mean_daily,
            "se": se_daily,
            "ci_lo": daily_excess["excess_ci_lo"].sum() / n,
            "ci_hi": daily_excess["excess_ci_hi"].sum() / n,
            "n_days": n,
        },
    ])
