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


def _period_bounds(pconf, post_dates, post_event_start, post_event_end):
    """Resolve one sub-period config to (mask, start, end) over the post-event rows.

    Offsets are unit-explicit: exactly one of the two key families per period.
    start_offset_days/end_offset_days delimit in calendar time from the post-event
    start (inclusive end date); start_offset_obs/end_offset_obs delimit by row
    position (inclusive end index). The two agree only on daily series, which
    is why the unit must be named.
    """
    name, has_days = _validate_period_keys(pconf)

    if has_days:
        p_start = post_event_start + pd.Timedelta(days=pconf.get("start_offset_days", 0))
        if pconf.get("end_offset_days") is not None:
            p_end = post_event_start + pd.Timedelta(days=pconf["end_offset_days"])
        else:
            p_end = post_event_end
        return (post_dates >= p_start) & (post_dates <= p_end), p_start, p_end

    idx = np.arange(len(post_dates))
    start_obs = pconf.get("start_offset_obs") or 0
    end_obs = pconf.get("end_offset_obs")
    mask = idx >= start_obs
    if end_obs is not None:
        mask &= idx <= end_obs
    if not mask.any():
        return mask, None, None
    masked = post_dates[mask]
    p_start = masked.min() if hasattr(masked, "min") else masked[0]
    p_end = masked.max() if hasattr(masked, "max") else masked[-1]
    return mask, p_start, p_end


def excess_from_paths(observed, point, paths, level, period_masks=()):
    """Per-step and per-period excess from forecast paths.

    The excess arithmetic shared by every caller that holds bootstrap paths,
    so each forms its intervals in the same way. Totals are formed within
    each path and then summarised, never by adding per-step bounds.

    Parameters
    ----------
    observed : np.ndarray or None
        Observed values, shape (n,). None gives NaN excess and bounds.
    point : np.ndarray
        Point forecast, shape (n,).
    paths : np.ndarray
        Shape (n, n_paths), one column per bootstrap replication.
    level : float
        Interval level, e.g. 0.95.
    period_masks : sequence of np.ndarray
        Boolean masks of shape (n,), one per period to total.

    Returns
    -------
    tuple[list[dict], list[dict]]
        One dict per step (keys observed, expected, excess, excess_lo,
        excess_hi, excess_pct, excess_pct_lo, excess_pct_hi), then one dict
        per mask (keys n_obs, total_observed, total_expected, total_excess,
        excess_lo, excess_hi, excess_pct).
    """
    alpha = 1 - level
    lo_q = alpha / 2
    hi_q = 1 - alpha / 2

    steps = []
    for i in range(len(point)):
        obs_i = float(observed[i]) if observed is not None else np.nan
        expected = float(point[i])
        excess = obs_i - expected

        # Excess bounds: observed minus each path's forecast.
        if observed is not None:
            excess_sims = obs_i - paths[i, :]
            excess_lo = float(np.nanpercentile(excess_sims, 100 * lo_q))
            excess_hi = float(np.nanpercentile(excess_sims, 100 * hi_q))
        else:
            excess_lo = excess_hi = np.nan

        excess_pct = (excess / expected * 100) if expected != 0 else np.nan
        if observed is not None and expected != 0:
            pct_sims = excess_sims / expected * 100
            excess_pct_lo = float(np.nanpercentile(pct_sims, 100 * lo_q))
            excess_pct_hi = float(np.nanpercentile(pct_sims, 100 * hi_q))
        else:
            excess_pct_lo = excess_pct_hi = np.nan

        steps.append({
            "observed": obs_i,
            "expected": expected,
            "excess": excess,
            "excess_lo": excess_lo,
            "excess_hi": excess_hi,
            "excess_pct": excess_pct,
            "excess_pct_lo": excess_pct_lo,
            "excess_pct_hi": excess_pct_hi,
        })

    periods = []
    for mask in period_masks:
        p_observed = observed[mask] if observed is not None else None
        total_observed = float(np.nansum(p_observed)) if p_observed is not None else np.nan
        total_expected = float(np.nansum(point[mask]))
        total_excess = total_observed - total_expected

        # Period bounds: total each path over the period, then take quantiles.
        if observed is not None:
            sim_totals = total_observed - np.nansum(paths[mask, :], axis=0)
            excess_lo = float(np.nanpercentile(sim_totals, 100 * lo_q))
            excess_hi = float(np.nanpercentile(sim_totals, 100 * hi_q))
        else:
            excess_lo = excess_hi = np.nan

        periods.append({
            "n_obs": int(mask.sum()),
            "total_observed": total_observed,
            "total_expected": total_expected,
            "total_excess": total_excess,
            "excess_lo": excess_lo,
            "excess_hi": excess_hi,
            "excess_pct": ((total_excess / total_expected * 100)
                           if total_expected != 0 else np.nan),
        })

    return steps, periods


def calculate_excess(bootstrap_result, intervention_date, periods_config=None,
                     ci_level=0.95):
    """Calculate per-observation and period-level excess from bootstrap results.

    Parameters
    ----------
    bootstrap_result : BootstrapCIResult
        Output of MBB bootstrap.
    intervention_date : pd.Timestamp
        Intervention date (only post-event dates are used for excess).
    periods_config : list[dict], optional
        Custom sub-periods, each with 'name' plus offsets in exactly one
        unit family: 'start_offset_days'/'end_offset_days' (calendar days
        from the post-event start, inclusive end) or 'start_offset_obs'/
        'end_offset_obs' (row counts into the post-event window, inclusive end).
    ci_level : float
        Confidence level for CIs.

    Returns
    -------
    ExcessResult
    """
    dates = pd.to_datetime(bootstrap_result.dates)
    intervention_date = pd.Timestamp(intervention_date)

    # Restrict to the post-event window
    post_event_mask = dates >= intervention_date
    if not post_event_mask.any():
        return ExcessResult(obs_excess=pd.DataFrame(), period_excess=pd.DataFrame())

    post_dates = dates[post_event_mask]
    post_actual = bootstrap_result.actual[post_event_mask] if bootstrap_result.actual is not None else None
    post_predicted = bootstrap_result.predicted[post_event_mask]
    post_pred_matrix = bootstrap_result.pred_matrix[post_event_mask, :]
    post_conf_lo = bootstrap_result.conf_lo[post_event_mask]
    post_conf_hi = bootstrap_result.conf_hi[post_event_mask]

    # Periods: the full post-event window first, then any configured ones;
    # a period covering no post-event row is left out.
    all_periods = [{"name": "Full post-event",
                    "start_offset_obs": 0, "end_offset_obs": None}]
    if periods_config:
        all_periods.extend(periods_config)

    post_event_start = post_dates.min() if hasattr(post_dates, "min") else post_dates[0]
    post_event_end = post_dates.max() if hasattr(post_dates, "max") else post_dates[-1]

    resolved = []
    for pconf in all_periods:
        p_mask, p_start, p_end = _period_bounds(
            pconf, post_dates, post_event_start, post_event_end)
        if p_mask.any():
            resolved.append((pconf["name"], p_mask, p_start, p_end))

    steps, periods = excess_from_paths(
        post_actual, post_predicted, post_pred_matrix, ci_level,
        [mask for _, mask, _, _ in resolved])

    obs_rows = []
    for i, step in enumerate(steps):
        obs_rows.append({
            "date": post_dates.iloc[i] if hasattr(post_dates, "iloc") else post_dates[i],
            "observed": step["observed"],
            "expected": step["expected"],
            "expected_ci_lo": float(post_conf_lo[i]),
            "expected_ci_hi": float(post_conf_hi[i]),
            "excess": step["excess"],
            "excess_ci_lo": step["excess_lo"],
            "excess_ci_hi": step["excess_hi"],
            "excess_pct": step["excess_pct"],
            "excess_pct_ci_lo": step["excess_pct_lo"],
            "excess_pct_ci_hi": step["excess_pct_hi"],
        })
    obs_excess = pd.DataFrame(obs_rows)

    period_rows = []
    for (name, _, p_start, p_end), period in zip(resolved, periods):
        period_rows.append({
            "period": name,
            "start_date": p_start,
            "end_date": p_end,
            "n_obs": period["n_obs"],
            "total_observed": period["total_observed"],
            "total_expected": period["total_expected"],
            "total_excess": period["total_excess"],
            "excess_ci_lo": period["excess_lo"],
            "excess_ci_hi": period["excess_hi"],
            "excess_pct": period["excess_pct"],
        })
    period_excess = pd.DataFrame(period_rows)

    return ExcessResult(obs_excess=obs_excess, period_excess=period_excess)


def calc_ate_summary(excess_result):
    """Calculate Average Treatment Effect summary from excess results.

    CIs are derived from the period-level "Full post-event" row, which
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

    # Try to get CIs from the period-level "Full post-event" row, which
    # sums per-simulation predictions then takes percentiles (correct).
    full_post_row = period_excess[period_excess["period"] == "Full post-event"] if not period_excess.empty else pd.DataFrame()

    if not full_post_row.empty:
        row = full_post_row.iloc[0]
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
