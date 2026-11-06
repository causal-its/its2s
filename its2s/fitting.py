# Description: What every workflow step that fits a model shares: the clock
#   that stamps the first fit and counts them, the one helper through which
#   every family fit runs, forecasting with a finite-value check, the rows of
#   a sealed window, the declared seasonality rebuilt from a handoff, the
#   seasonal-naive period, the span of a set of rows, and the check that the
#   data is the data a design was sealed against. Screening, the check and
#   the final fit all use these, so "fit a family on these rows" has one
#   implementation.
# Usage: from its2s.fitting import FitClock, fit_family, predict_family, rows_in
# Dependencies: numpy, pandas

import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .design_record import _iso, fingerprint
from .folds import longest_cycle
from .frequency import SeriesFrequency
from .seasonality import make_seasonality


# ---------------------------------------------------------------------------
# Fits: every family fit in the workflow runs through fit_family
# ---------------------------------------------------------------------------

class FitClock:
    """Counts the fits run and stamps the moment the first one starts.

    Every family fit in the workflow goes through fit_family, which ticks
    the clock immediately before calling the family, so first_fit_at is the
    time no fit can precede.
    """

    def __init__(self):
        self.first_fit_at = None
        self.n_fits = 0

    def tick(self):
        if self.first_fit_at is None:
            self.first_fit_at = utc_now()
        self.n_fits += 1


def utc_now():
    """The current time in UTC to the microsecond, as the records state it."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class Columns:
    def __init__(self, data_handoff):
        self.date = data_handoff["date_col"]
        self.y = data_handoff["outcome_col"]
        self.cov = list(data_handoff["covariate_cols"]) or None


def count_warnings(caught, counter):
    for w in caught:
        counter[w.category.__name__] += 1


def fit_family(model, df, cols, clock, counter):
    """Fit model on df; library warnings are counted by category, not shown."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        clock.tick()
        try:
            return model.fit(df, target_col=cols.y, date_col=cols.date,
                             covariate_cols=cols.cov)
        finally:
            count_warnings(caught, counter)


def predict_family(model, df, cols, counter):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            pred = model.predict(df, target_col=cols.y, date_col=cols.date,
                                 covariate_cols=cols.cov)
        finally:
            count_warnings(caught, counter)
    predicted = np.asarray(pred.predicted, dtype=float)
    if not np.all(np.isfinite(predicted)):
        raise ValueError(f"the forecast holds {int((~np.isfinite(predicted)).sum())} "
                         "non-finite values")
    return predicted


def error_line(e):
    return f"{type(e).__name__}: {e}"


def selected_structure(family, fit_result):
    """What the fit selected, where the family selects anything itself.

    ARIMA's tuned values are search bounds, so each fit picks its own order;
    the order, seasonal order and intercept are recorded per fit.
    """
    if family != "arima":
        return None
    model = fit_result.model_object
    return {"order": list(fit_result.metadata["order"]),
            "seasonal_order": list(fit_result.metadata["seasonal_order"]),
            "with_intercept": bool(getattr(model, "with_intercept", False))}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def rows_in(df, date_col, bounds, what):
    """The rows of df dated in [start, end); their count must be sealed n."""
    dates = pd.to_datetime(df[date_col])
    keep = (dates >= pd.Timestamp(bounds["start"])) & (dates < pd.Timestamp(bounds["end"]))
    part = df.loc[keep].copy()
    if len(part) != bounds["n"]:
        raise ValueError(
            f"{what} [{bounds['start']}, {bounds['end']}) holds {len(part)} rows in "
            f"the data and {bounds['n']} in the sealed design.")
    return part


def span_of(df, date_col):
    """The first and last date of df's rows, and how many there are."""
    dates = pd.to_datetime(df[date_col])
    return {"first": _iso(dates.iloc[0]), "last": _iso(dates.iloc[-1]), "n": len(df)}


def seasonality_of(handoff):
    """The declared Seasonality, rebuilt from a handoff and checked against
    the periods it sealed."""
    s = handoff["seasonality"]
    seasonality = make_seasonality(s["cycles"],
                                   SeriesFrequency.from_alias(handoff["data"]["freq"]))
    if dict(seasonality.periods) != dict(s["periods"]):
        raise ValueError(
            f"This package derives the seasonal periods {dict(seasonality.periods)} "
            f"where the design sealed {dict(s['periods'])}.")
    return seasonality


def mase_period(seasonality):
    """The seasonal-naive period MASE compares against.

    One cycle of the longest declared cycle, rounded to whole rows (annual on
    weekly data is 52); the plain naive (1) when no cycle is declared.
    """
    cycle = longest_cycle(seasonality)
    return 1 if cycle is None else int(round(seasonality.periods[cycle]))


def verify_data(handoff, data, error):
    """The data must be the data the design was sealed against.

    Raises error (the calling step's exception class) when a column is
    missing or the fingerprint differs; returns the fingerprint.
    """
    d = handoff["data"]
    cols = [d["outcome_col"], *d["covariate_cols"]]
    missing = [c for c in [d["date_col"], *cols] if c not in data.columns]
    if missing:
        raise error(f"The data has no column(s) {missing}; the design was "
                    f"sealed on {[d['date_col'], *cols]}.")
    fp = fingerprint(data, d["date_col"], cols, handoff["sub_periods"]["full_window"]["end"])
    if fp["digest"] != d["fingerprint"]:
        rows, sealed = fp["rows"], d["rows"]
        raise error(
            f"The data does not match the design's fingerprint. Sealed: {d['fingerprint']} "
            f"over {sealed['n']} rows, {sealed['first']} to {sealed['last']}. Given: "
            f"{fp['digest']} over {rows['n']} rows, {rows['first']} to {rows['last']}. "
            "Nothing has been fit.")
    return fp
