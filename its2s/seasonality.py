# Description: The declared seasonal cycles of an analysis and their periods m
#   (contract P11.04 sections 3.5 and 5.4). Every model family consumes one
#   Seasonality, so no fit decides its own seasonal structure from its own
#   length (GH #74). This module holds only the value and its arithmetic;
#   validating and printing a declaration for the analyst belongs to step 0.
#   The second half of this module serves only the old path (run_single_its,
#   tune_model), which has no declaration and derives one from its config; it
#   is deleted with that path in P11.04.07.
# Usage: from its2s.seasonality import make_seasonality
# Dependencies: pandas

import logging
import warnings
from dataclasses import dataclass

import pandas as pd
from pandas.tseries import offsets

from .frequency import SeriesFrequency, dominant_seasonal_period

logger = logging.getLogger(__name__)

CYCLES = ("weekly", "annual")

# Cycle lengths in calendar terms, for day- and week-based sampling.
_CYCLE_DAYS = {"weekly": 7.0, "annual": 365.25}


@dataclass(frozen=True)
class Seasonality:
    """The seasonal cycles every fit of an analysis carries.

    Attributes
    ----------
    cycles : tuple of str
        Subset of ("weekly", "annual"), in that order. () asserts the series
        is aseasonal.
    freq : SeriesFrequency
        The series' resolved sampling interval.
    periods : dict
        cycle -> period m in observations, e.g. {"annual": 52.18} on weekly
        data. Not rounded; a family needing an integer rounds it.
    declared : bool
        False when the cycles were left at the default rather than declared.
    """

    cycles: tuple
    freq: SeriesFrequency
    periods: dict
    declared: bool = True

    def to_dict(self):
        return {"cycles": list(self.cycles), "freq": self.freq.alias,
                "periods": dict(self.periods), "declared": self.declared}

    @classmethod
    def from_dict(cls, d):
        return make_seasonality(d["cycles"], SeriesFrequency.from_alias(d["freq"]),
                                declared=d["declared"])


def _sampling_interval(freq):
    """Return ("days", n) for day/week-based sampling or ("months", n)."""
    off = freq.offset
    if isinstance(off, offsets.Tick):
        return "days", pd.Timedelta(off) / pd.Timedelta(days=1)
    if isinstance(off, offsets.Week):
        return "days", 7.0 * off.n
    if isinstance(off, (offsets.MonthBegin, offsets.MonthEnd)):
        return "months", float(off.n)
    if isinstance(off, (offsets.QuarterBegin, offsets.QuarterEnd)):
        return "months", 3.0 * off.n
    if isinstance(off, (offsets.YearBegin, offsets.YearEnd)):
        return "months", 12.0 * off.n
    raise ValueError(
        f"No seasonal period arithmetic for series frequency {freq.alias!r}; "
        "supported are day-, week-, month-, quarter- and year-based series.")


def cycle_period(cycle, freq):
    """Period m of one cycle at the series' sampling interval, in observations.

    Annual is 365.25 days on day- and week-based sampling and exactly 12 / 4
    on monthly / quarterly data. A cycle no longer than two sampling
    intervals cannot be estimated and raises.
    """
    if cycle not in CYCLES:
        raise ValueError(f"Unknown cycle {cycle!r}; expected one of {list(CYCLES)}.")
    unit, n = _sampling_interval(freq)
    if unit == "days":
        m = _CYCLE_DAYS[cycle] / n
    else:
        m = 12.0 / n if cycle == "annual" else 0.0
    if m <= 2:
        raise ValueError(
            f"A {cycle} cycle cannot be carried on {freq.alias!r} data: it spans "
            f"{m:.3g} observations, and a cycle needs more than two.")
    return m


def make_seasonality(cycles, freq, declared=True):
    """Build the Seasonality for the given cycles on a series of frequency freq."""
    unknown = set(cycles) - set(CYCLES)
    if unknown:
        raise ValueError(
            f"Unknown cycles {sorted(unknown)}; expected a subset of {list(CYCLES)}.")
    cycles = tuple(c for c in CYCLES if c in set(cycles))
    periods = {c: cycle_period(c, freq) for c in cycles}
    return Seasonality(cycles=cycles, freq=freq, periods=periods, declared=declared)


# --- Old path only: deleted with run_single_its and tune_model in P11.04.07 ---

# Both prophet 1.3.0 and neuralprophet 0.8.0 enable yearly seasonality under
# "auto" only when the training history spans at least this many days.
_YEARLY_AUTO_MIN_DAYS = 730


def report_auto_yearly_resolution(df, yearly_arg):
    """Report visibly which way the library's auto rule resolves yearly seasonality.

    Prophet and NeuralProphet apply the identical rule: under "auto", the
    yearly component is enabled only when the non-null training history spans
    at least 730 days. The libraries decide this quietly, with at most a log
    line, so a series one day either side of the threshold gets a materially
    different model with nothing in the output to say so (GH #60, D-057,
    D-080). This reports the resolution in BOTH directions, so the choice is
    always visible rather than only visible when it goes badly.

    An explicit True/False is honored silently: the user made the choice, so
    there is nothing to report. Only "auto" is announced.

    df must carry ds and y columns.
    """
    if yearly_arg != "auto":
        return
    dates = pd.to_datetime(df.loc[df["y"].notna(), "ds"])
    if dates.empty:
        return
    span_days = (dates.max() - dates.min()).days
    if span_days < _YEARLY_AUTO_MIN_DAYS:
        warnings.warn(
            f"yearly_seasonality='auto': the training history spans "
            f"{span_days} days, under the {_YEARLY_AUTO_MIN_DAYS}-day "
            f"(two annual cycles) minimum, so the yearly component is "
            f"DISABLED. This threshold is a hard boundary: a series near it "
            f"can change substantially on one more day of history. Pass "
            f"yearly_seasonality=True to force the component on.",
            UserWarning,
            stacklevel=2,
        )
    else:
        warnings.warn(
            f"yearly_seasonality='auto': the training history spans "
            f"{span_days} days, at or over the {_YEARLY_AUTO_MIN_DAYS}-day "
            f"(two annual cycles) minimum, so the yearly component is "
            f"ENABLED. Pass yearly_seasonality=False to suppress it.",
            UserWarning,
            stacklevel=2,
        )


def resolve_arima_m(m_cfg, n_train, series_freq=None):
    """Resolve the seasonal period m for the auto_arima search (GH #59).

    "auto" derives the dominant cycle from the resolved series frequency
    (daily 7, weekly 52, monthly 12) and falls back LOUDLY to m=1
    (non-seasonal) when the frequency is unmapped or the training window
    fails the length guard n_train >= 2m (one cycle is consumed by seasonal
    differencing; at least another is needed to estimate the seasonal terms).

    An explicit integer is always honored, never substituted: m < 1 raises,
    and an explicit value failing the length guard fits as asked with an
    advisory warning. This deliberately differs from
    resolve_metrics_seasonality, which raises on an explicit guard failure:
    there m defines the benchmark a reported metric is measured against;
    here it is a model specification whose quality is visible in the fit's
    own diagnostics.
    """
    if m_cfg == "auto":
        m = dominant_seasonal_period(series_freq)
        if m is None:
            alias = series_freq.alias if series_freq is not None else "unknown"
            warnings.warn(
                f"models.arima.m='auto': no dominant seasonal period is "
                f"mapped for series frequency '{alias}'. Falling back to "
                "m=1 (non-seasonal ARIMA). Set models.arima.m to an integer "
                "to name the period explicitly: "
                'config_overrides={"models": {"arima": {"m": <period>}}}.',
                UserWarning,
                stacklevel=2,
            )
            return 1
        if n_train < 2 * m:
            warnings.warn(
                f"models.arima.m='auto': a seasonal fit at m={m} needs "
                f"n_train >= 2*m ({2 * m}) but only {n_train} training "
                "observations are available. Falling back to m=1 "
                "(non-seasonal ARIMA). Set models.arima.m to an integer to "
                "override.",
                UserWarning,
                stacklevel=2,
            )
            return 1
        logger.info(
            "ARIMA seasonal period resolved from series frequency %s: m=%d",
            series_freq.alias, m,
        )
        return m

    m = int(m_cfg)
    if m < 1:
        raise ValueError(f"models.arima.m must be >= 1, got {m_cfg!r}.")
    if n_train < 2 * m:
        warnings.warn(
            f"models.arima.m={m}: a seasonal fit at m={m} needs n_train >= "
            f"2*m ({2 * m}) but only {n_train} training observations are "
            "available. The explicit value is honored; expect an unstable "
            "or degenerate seasonal fit. Set a smaller m, or "
            "models.arima.seasonal=false, to silence this warning.",
            UserWarning,
            stacklevel=2,
        )
    return m


def _library_auto_rule(cycle, dates):
    """Prophet's and NeuralProphet's "auto" rule, identical in prophet 1.3.0
    and neuralprophet 0.8.0: yearly needs two years of history; weekly needs
    two weeks of history and a spacing under seven days."""
    first, last = dates.min(), dates.max()
    dt = dates.diff()
    min_dt = dt.iloc[dt.values.nonzero()[0]].min()
    if cycle == "annual":
        return last - first >= pd.Timedelta(days=_YEARLY_AUTO_MIN_DAYS)
    return last - first >= pd.Timedelta(weeks=2) and min_dt < pd.Timedelta(weeks=1)


def _legacy_cycle_on(key, arg, cycle, dates):
    if arg is True or arg is False:
        return arg
    if arg == "auto":
        return _library_auto_rule(cycle, dates)
    raise ValueError(
        f"{key}={arg!r}: only True, False or 'auto' are supported; a seasonal "
        "component is declared, never sized by a Fourier order.")


def _cycles_for_m(m, freq):
    """The declared cycle an ARIMA period m stands for (spec_P11.04.02 OQ-9)."""
    if m == 1:
        return ()
    for cycle in CYCLES:
        try:
            if round(cycle_period(cycle, freq)) == m:
                return (cycle,)
        except ValueError:
            continue
    raise ValueError(
        f"models.arima.m={m} is not the period of a declarable cycle on "
        f"{freq.alias!r} data (weekly or annual). ARIMA carries the declared "
        "cycle's period; declare the cycle rather than the period.")


def legacy_seasonality(model_name, params, fit_df, freq, date_col="ds",
                       target_col="y"):
    """Derive the Seasonality the old path's config asks for, for one fit.

    Reproduces what each family decided for itself before P11.04.02:
    prophet_xgb and neuralprophet apply their library's "auto" rule to the
    fit's own history (reported both ways for yearly, D-057, D-080); arima
    resolves m with resolve_arima_m and carries the cycle that m stands for.
    """
    if model_name == "arima":
        if not params.get("seasonal", True):
            return make_seasonality((), freq, declared=False)
        m = resolve_arima_m(params.get("m", "auto"), n_train=len(fit_df),
                            series_freq=freq)
        return make_seasonality(_cycles_for_m(m, freq), freq, declared=False)

    if model_name == "prophet_xgb":
        p = params.get("prophet", {})
        if p.get("daily_seasonality", False) is not False:
            raise ValueError("models.prophet_xgb.prophet.daily_seasonality: a "
                             "daily cycle is not declarable; only False is supported.")
        # Prophet's history is the rows with an observed outcome.
        history = fit_df.loc[fit_df[target_col].notna()]
    else:
        p = params
        history = fit_df
    dates = pd.to_datetime(history[date_col]).reset_index(drop=True)
    yearly = p.get("yearly_seasonality", "auto")
    weekly = p.get("weekly_seasonality", "auto")
    report_auto_yearly_resolution(
        pd.DataFrame({"ds": dates, "y": history[target_col].to_numpy()}), yearly)
    cycles = [c for c, key, arg in (("weekly", "weekly_seasonality", weekly),
                                    ("annual", "yearly_seasonality", yearly))
              if _legacy_cycle_on(key, arg, c, dates)]
    return make_seasonality(cycles, freq, declared=False)
