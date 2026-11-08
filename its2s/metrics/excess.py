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


# Why a percent excess is empty under percent="within_path": the point
# forecast it divides by is not positive, or some path's denominator is not.
PCT_FORECAST_NOT_POSITIVE = "forecast_not_positive"
PCT_PATH_NOT_POSITIVE = "path_not_positive"


def _quantiles(values, lo_q, hi_q):
    return (float(np.percentile(values, 100 * lo_q)),
            float(np.percentile(values, 100 * hi_q)))


def _within_path_pct(observed, point, path_values, lo_q, hi_q):
    """Percent excess formed within each path, then summarised.

    The point is (observed - point) / point * 100; the bounds are quantiles
    of (observed - path) / path * 100 over the paths. A ratio is defined only
    over a positive denominator, and no path is dropped to make one so: the
    point and bounds are None when the point is not positive, the bounds
    alone when any path is not. Returns (pct, lo, hi, reason), reason None
    when nothing is empty.
    """
    if not point > 0:
        return None, None, None, PCT_FORECAST_NOT_POSITIVE
    pct = (observed - point) / point * 100
    if not np.all(path_values > 0):
        return pct, None, None, PCT_PATH_NOT_POSITIVE
    lo, hi = _quantiles((observed - path_values) / path_values * 100, lo_q, hi_q)
    return pct, lo, hi, None


def excess_from_paths(observed, point, paths, level, period_masks=(),
                      percent="point"):
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
        Shape (n, n_paths), one column per SUCCESSFUL bootstrap replication.
        Every value must be finite: a failed replication contributes no
        column, so no path is ever summed or ranked as if NaN were 0.
    level : float
        Interval level, e.g. 0.95.
    period_masks : sequence of np.ndarray
        Boolean masks of shape (n,), one per period to total.
    percent : str
        "point" (the old path's): percent bounds divide each path's excess
        by the point forecast, and periods carry no percent bounds.
        "within_path" (step 4's): each path's excess is divided by that
        path's own counterfactual, per step and per period total, and the
        result summarised; see _within_path_pct for when it is empty.

    Returns
    -------
    tuple[list[dict], list[dict]]
        One dict per step (keys observed, expected, counterfactual_lo,
        counterfactual_hi, excess, excess_lo, excess_hi, excess_pct,
        excess_pct_lo, excess_pct_hi), then one dict per mask (keys n_obs,
        total_observed, total_expected, total_excess, excess_lo, excess_hi,
        excess_pct, point_outside_interval, median_path_minus_point). Under
        "within_path" each step and period also carries excess_pct_null,
        the reason its percent is empty or None, and periods carry
        excess_pct_lo and excess_pct_hi.
    """
    if percent not in ("point", "within_path"):
        raise ValueError(f"percent must be 'point' or 'within_path'; got {percent!r}.")
    within = percent == "within_path"
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 2 or paths.shape[1] == 0:
        raise ValueError("excess_from_paths needs at least one path.")
    if not np.all(np.isfinite(paths)):
        raise ValueError("excess_from_paths was given a non-finite path; "
                         "failed replications must be removed, not passed as NaN.")
    if within and observed is None:
        raise ValueError("percent='within_path' needs the observed values.")

    alpha = 1 - level
    lo_q = alpha / 2
    hi_q = 1 - alpha / 2

    steps = []
    for i in range(len(point)):
        obs_i = float(observed[i]) if observed is not None else np.nan
        expected = float(point[i])
        excess = obs_i - expected
        cf_lo, cf_hi = _quantiles(paths[i, :], lo_q, hi_q)
        step = {"observed": obs_i, "expected": expected,
                "counterfactual_lo": cf_lo, "counterfactual_hi": cf_hi,
                "excess": excess}

        # Excess bounds: observed minus each path's forecast.
        if observed is not None:
            excess_sims = obs_i - paths[i, :]
            excess_lo, excess_hi = _quantiles(excess_sims, lo_q, hi_q)
        else:
            excess_lo = excess_hi = np.nan
        step["excess_lo"], step["excess_hi"] = excess_lo, excess_hi

        if within:
            (step["excess_pct"], step["excess_pct_lo"], step["excess_pct_hi"],
             step["excess_pct_null"]) = _within_path_pct(obs_i, expected, paths[i, :],
                                                          lo_q, hi_q)
        else:
            step["excess_pct"] = (excess / expected * 100) if expected != 0 else np.nan
            if observed is not None and expected != 0:
                step["excess_pct_lo"], step["excess_pct_hi"] = _quantiles(
                    excess_sims / expected * 100, lo_q, hi_q)
            else:
                step["excess_pct_lo"] = step["excess_pct_hi"] = np.nan
        steps.append(step)

    periods = []
    for mask in period_masks:
        p_observed = observed[mask] if observed is not None else None
        total_observed = float(np.nansum(p_observed)) if p_observed is not None else np.nan
        total_expected = float(np.nansum(point[mask]))
        total_excess = total_observed - total_expected
        path_totals = np.sum(paths[mask, :], axis=0)

        # Period bounds: total each path over the period, then take quantiles.
        if observed is not None:
            excess_lo, excess_hi = _quantiles(total_observed - path_totals, lo_q, hi_q)
        else:
            excess_lo = excess_hi = np.nan

        period = {
            "n_obs": int(mask.sum()),
            "total_observed": total_observed,
            "total_expected": total_expected,
            "total_excess": total_excess,
            "excess_lo": excess_lo,
            "excess_hi": excess_hi,
            # The point estimate outside its own interval: the paths are
            # shifted off the point forecast.
            "point_outside_interval": bool(total_excess < excess_lo
                                           or total_excess > excess_hi),
            "median_path_minus_point": float(np.median(path_totals) - total_expected),
        }
        if within:
            (period["excess_pct"], period["excess_pct_lo"], period["excess_pct_hi"],
             period["excess_pct_null"]) = _within_path_pct(
                total_observed, total_expected, path_totals, lo_q, hi_q)
        else:
            period["excess_pct"] = ((total_excess / total_expected * 100)
                                    if total_expected != 0 else np.nan)
        periods.append(period)

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
