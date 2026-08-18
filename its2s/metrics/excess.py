# Description: Excess calculation from MBB bootstrap results with CIs.
# Usage: from its2s.metrics.excess import calculate_excess, calc_ate_summary
# Dependencies: numpy, pandas

from dataclasses import dataclass

import numpy as np
import pandas as pd

_PERIOD_KEYS_DAYS = {"start_offset_days", "end_offset_days"}
_PERIOD_KEYS_OBS = {"start_offset_obs", "end_offset_obs"}
_PERIOD_KEYS_LEGACY = {"start_offset", "end_offset"}


@dataclass
class ExcessResult:
    """Container for excess estimates."""

    obs_excess: pd.DataFrame
    period_excess: pd.DataFrame


def _validate_period_keys(pconf):
    """Check one sub-period config's offset keys; return (name, has_days).

    Raises on the legacy ambiguous-unit keys, on mixing the _days and _obs
    families in one period, and on a period with no offsets at all. Shared by
    _period_bounds (at computation time) and validate_excess_periods (at
    input-validation time, so a bad config fails before any expensive work).
    """
    name = pconf.get("name", "<unnamed>")
    keys = set(pconf)
    if keys & _PERIOD_KEYS_LEGACY:
        raise ValueError(
            f"excess period '{name}': start_offset/end_offset are no longer "
            "accepted because their unit was ambiguous. Use "
            "start_offset_days/end_offset_days (calendar days) or "
            "start_offset_obs/end_offset_obs (observation counts)."
        )
    has_days = bool(keys & _PERIOD_KEYS_DAYS)
    has_obs = bool(keys & _PERIOD_KEYS_OBS)
    if has_days and has_obs:
        raise ValueError(
            f"excess period '{name}': mixes _days and _obs offsets; "
            "delimit each period in exactly one unit family."
        )
    if not has_days and not has_obs:
        raise ValueError(
            f"excess period '{name}': no offsets given. Use "
            "start_offset_days/end_offset_days or "
            "start_offset_obs/end_offset_obs."
        )
    return name, has_days


def validate_excess_periods(periods_config):
    """Validate excess-period configs without computing anything.

    Runs the same key checks _period_bounds applies, so the pipeline can
    reject a stale or malformed excess_periods section at input validation
    instead of after the model fit and the full bootstrap.
    """
    for pconf in periods_config or []:
        _validate_period_keys(pconf)


def _period_bounds(pconf, h_dates, holdout_start, holdout_end):
    """Resolve one sub-period config to (mask, start, end) over the holdout rows.

    Offsets are unit-explicit: exactly one of the two key families per period.
    start_offset_days/end_offset_days delimit in calendar time from the holdout
    start (inclusive end date); start_offset_obs/end_offset_obs delimit by row
    position (inclusive end index). The two agree only on daily series, which
    is why the unit must be named.
    """
    name, has_days = _validate_period_keys(pconf)

    if has_days:
        p_start = holdout_start + pd.Timedelta(days=pconf.get("start_offset_days", 0))
        if pconf.get("end_offset_days") is not None:
            p_end = holdout_start + pd.Timedelta(days=pconf["end_offset_days"])
        else:
            p_end = holdout_end
        return (h_dates >= p_start) & (h_dates <= p_end), p_start, p_end

    idx = np.arange(len(h_dates))
    start_obs = pconf.get("start_offset_obs") or 0
    end_obs = pconf.get("end_offset_obs")
    mask = idx >= start_obs
    if end_obs is not None:
        mask &= idx <= end_obs
    if not mask.any():
        return mask, None, None
    masked = h_dates[mask]
    p_start = masked.min() if hasattr(masked, "min") else masked[0]
    p_end = masked.max() if hasattr(masked, "max") else masked[-1]
    return mask, p_start, p_end


def calculate_excess(bootstrap_result, intervention_date, periods_config=None,
                     ci_level=0.95):
    """Calculate per-observation and period-level excess from bootstrap results.

    Parameters
    ----------
    bootstrap_result : BootstrapCIResult
        Output of MBB bootstrap.
    intervention_date : pd.Timestamp
        Intervention date (only holdout dates are used for excess).
    periods_config : list[dict], optional
        Custom sub-periods, each with 'name' plus offsets in exactly one
        unit family: 'start_offset_days'/'end_offset_days' (calendar days
        from the holdout start, inclusive end) or 'start_offset_obs'/
        'end_offset_obs' (row counts into the holdout, inclusive end).
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
        return ExcessResult(obs_excess=pd.DataFrame(), period_excess=pd.DataFrame())

    h_dates = dates[holdout_mask]
    h_actual = bootstrap_result.actual[holdout_mask] if bootstrap_result.actual is not None else None
    h_predicted = bootstrap_result.predicted[holdout_mask]
    h_pred_matrix = bootstrap_result.pred_matrix[holdout_mask, :]
    h_conf_lo = bootstrap_result.conf_lo[holdout_mask]
    h_conf_hi = bootstrap_result.conf_hi[holdout_mask]

    # Per-observation excess
    obs_rows = []
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

        obs_rows.append({
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

    obs_excess = pd.DataFrame(obs_rows)

    # Period-level excess
    period_rows = []

    # Default: full holdout period
    all_periods = [{"name": "Full holdout",
                    "start_offset_obs": 0, "end_offset_obs": None}]
    if periods_config:
        all_periods.extend(periods_config)

    holdout_start = h_dates.min() if hasattr(h_dates, "min") else h_dates[0]
    holdout_end = h_dates.max() if hasattr(h_dates, "max") else h_dates[-1]

    for pconf in all_periods:
        p_mask, p_start, p_end = _period_bounds(
            pconf, h_dates, holdout_start, holdout_end)
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
            "n_obs": int(p_mask.sum()),
            "total_observed": total_observed,
            "total_expected": total_expected,
            "total_excess": total_excess,
            "excess_ci_lo": excess_lo,
            "excess_ci_hi": excess_hi,
            "excess_pct": excess_pct,
        })

    period_excess = pd.DataFrame(period_rows)

    return ExcessResult(obs_excess=obs_excess, period_excess=period_excess)


def calc_ate_summary(excess_result):
    """Calculate Average Treatment Effect summary from excess results.

    CIs are derived from the period-level "Full holdout" row, which
    computes total excess per bootstrap simulation and then takes
    percentiles. This correctly accounts for temporal correlation in
    bootstrap predictions (unlike summing independent per-observation CIs).

    Parameters
    ----------
    excess_result : ExcessResult or pd.DataFrame
        Either an ExcessResult (preferred) or an obs_excess DataFrame
        (legacy fallback -- CIs will be approximate).

    Returns
    -------
    pd.DataFrame
        Summary with total ATE and mean ATE per observation.
    """
    # Accept either ExcessResult or bare DataFrame for backwards compat
    if isinstance(excess_result, ExcessResult):
        obs_excess = excess_result.obs_excess
        period_excess = excess_result.period_excess
    else:
        obs_excess = excess_result
        period_excess = pd.DataFrame()

    if obs_excess.empty:
        return pd.DataFrame()

    n = len(obs_excess)
    total_excess = obs_excess["excess"].sum()
    mean_per_obs = total_excess / n

    # Try to get CIs from the period-level "Full holdout" row, which
    # sums per-simulation predictions then takes percentiles (correct).
    fullhold = period_excess[period_excess["period"] == "Full holdout"] if not period_excess.empty else pd.DataFrame()

    if not fullhold.empty:
        row = fullhold.iloc[0]
        total_ci_lo = float(row["excess_ci_lo"])
        total_ci_hi = float(row["excess_ci_hi"])
    else:
        # Fallback: sum per-observation CIs (approximate, assumes independence)
        total_ci_lo = obs_excess["excess_ci_lo"].sum()
        total_ci_hi = obs_excess["excess_ci_hi"].sum()

    return pd.DataFrame([
        {
            "metric": "Total ATE",
            "estimate": total_excess,
            "ci_lo": total_ci_lo,
            "ci_hi": total_ci_hi,
            "n_obs": n,
        },
        {
            "metric": "Mean ATE per obs",
            "estimate": mean_per_obs,
            "ci_lo": total_ci_lo / n,
            "ci_hi": total_ci_hi / n,
            "n_obs": n,
        },
    ])
