# Description: The optional look at a series before its design is declared.
#   explore() reads ONLY the rows strictly before a boundary date the analyst
#   names, and reports what
#   bears on the seasonality declaration: the series, a periodogram of the
#   detrended, variance-stabilised series with the candidate cycles marked,
#   the autocorrelation at the candidate lags, and a seasonal decomposition at
#   the longest candidate period. It prints the cautions that short,
#   overdispersed health series require. The result is handed to
#   Design.set_exploration, so the sealed record says what was looked at and
#   where the look stopped. Nothing is fitted that could inform a model
#   choice beyond the cycle, and nothing is written unless asked.
# Usage: from its2s import explore
#        x = explore(df, "week", "deaths", before="2025-06-16")
#        design.set_exploration(x)
# Dependencies: numpy, pandas, scipy, statsmodels, matplotlib

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .design_record import fingerprint
from .frequency import resolve_frequency
from .seasonality import CYCLES, cycle_period
from .windows import _grid_position

TRANSFORMS = ("sqrt", "log", None)

PLOT_FILE = "explore_plot.png"
SUMMARY_FILE = "explore.json"

# Seed of the shuffled copy the seasonal strength is compared with. Fixed, so
# explore is deterministic, and printed, so the reference can be reproduced.
SHUFFLE_SEED = 0

# The cautions of the explore call for short, overdispersed health series.
CAUTIONS = (
    "The series was detrended (a straight line removed) before the periodogram: "
    "a secular trend puts power at the long-period end and swamps everything else.",
    "Overdispersed counts are variance-stabilised first, or a few large values "
    "dominate the spectrum.",
    "Frequency resolution is poor on short series: the bins near a long cycle are "
    "coarse (their periods are listed above), so one bump in an adjacent bin is "
    "not a second cycle.",
    "A sharply peaked annual cycle produces a semi-annual harmonic. That is the "
    "shape of the annual cycle, not a separate cycle to declare.",
    "STL takes the period as an input. It confirms how stable and how large a "
    "cycle is; it cannot discover which cycle the series has.",
    "STL's seasonal strength runs high on series with no seasonal pattern at "
    "all, close to 1 under about three cycles of data. Read it only against "
    "the same statistic on the shuffled copy, which has no seasonal pattern "
    "by construction.",
    "Nothing here looked at or after the boundary. Choose a boundary no later "
    "than the start of every test window, or the test window has informed the "
    "declaration it is meant to check.",
)


@dataclass(frozen=True)
class Exploration:
    """What one explore() call read and computed.

    Attributes
    ----------
    boundary : pd.Timestamp
        Rows strictly before this date were read; none at or after it.
    rows_read : int
    first_date, last_date : pd.Timestamp
        The first and last rows read.
    freq : str
        The resolved sampling interval of the rows read (a pandas alias).
    transform : str or None
        The variance-stabilising transform applied before the periodogram,
        the autocorrelation and the decomposition.
    cycles : tuple of str
        The candidate cycles examined.
    numbers : dict
        Per cycle: its period m, the periodogram bins around it, the
        autocorrelation at its rounded lag; plus lag-1 autocorrelation, the
        strongest periodogram bin, and the decomposition's seasonal strength
        beside the same statistic on a shuffled copy of the values.
    fingerprint : dict
        The fingerprint of the rows read (date and outcome), in the form a
        design record uses, so a design can check it declares the same rows.
    figure : matplotlib.figure.Figure
    """

    boundary: pd.Timestamp
    rows_read: int
    first_date: pd.Timestamp
    last_date: pd.Timestamp
    freq: str
    transform: object
    cycles: tuple
    numbers: dict
    fingerprint: dict
    figure: object = field(default=None, repr=False, compare=False)

    def to_dict(self):
        return {
            "boundary": self.boundary.isoformat(),
            "rows_read": self.rows_read,
            "first_date": self.first_date.isoformat(),
            "last_date": self.last_date.isoformat(),
            "freq": self.freq,
            "transform": self.transform,
            "cycles": list(self.cycles),
            "numbers": self.numbers,
            "fingerprint": self.fingerprint,
            "cautions": list(CAUTIONS),
        }


# ---------------------------------------------------------------------------
# Reading the rows before the boundary
# ---------------------------------------------------------------------------

def _read_before(df, date_col, outcome_col, before):
    """The rows strictly before the boundary, sorted, as (dates, values)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame; got {type(df).__name__}.")
    for name, col in (("date_col", date_col), ("outcome_col", outcome_col)):
        if col not in df.columns:
            raise ValueError(f"{name} {col!r} is not a column of df.")
    try:
        boundary = pd.Timestamp(before)
    except (TypeError, ValueError) as e:
        raise TypeError(f"before {before!r} is not a date: {e}") from None
    dates = pd.to_datetime(df[date_col])
    # A boundary between two rows is refused: a row is labelled with one date
    # but covers a whole interval, so the row just before an off-grid date
    # can hold observations from after it.
    _, issue = _grid_position(pd.DatetimeIndex(dates.sort_values()), boundary,
                              "The boundary before=")
    if issue is not None:
        raise ValueError(issue.message + " Give a date of the series; rows "
                         "strictly before it are read.")
    keep = (dates < boundary).to_numpy()
    rows = pd.DataFrame({date_col: dates[keep].to_numpy(),
                         outcome_col: df.loc[keep, outcome_col].to_numpy()})
    rows = rows.sort_values(date_col, kind="stable").reset_index(drop=True)
    if len(rows) < 3:
        raise ValueError(f"{len(rows)} rows of df fall before the boundary "
                         f"{boundary.date()}; explore needs at least 3.")
    if not pd.api.types.is_numeric_dtype(rows[outcome_col]):
        raise ValueError(f"outcome_col {outcome_col!r} is not numeric.")
    n_missing = int(rows[outcome_col].isna().sum())
    if n_missing:
        raise ValueError(
            f"{n_missing} of the {len(rows)} rows before the boundary have no "
            f"{outcome_col!r} value. The periodogram and the decomposition need "
            "a complete series; fill or trim the frame, and say how in the paper.")
    return boundary, rows


def _transform(y, transform):
    if transform not in TRANSFORMS:
        raise ValueError(f"transform must be one of {list(TRANSFORMS)}; got {transform!r}.")
    y = np.asarray(y, dtype=float)
    if transform == "sqrt":
        if (y < 0).any():
            raise ValueError("transform='sqrt' needs non-negative values.")
        return np.sqrt(y)
    if transform == "log":
        if (y <= 0).any():
            raise ValueError("transform='log' needs positive values; a series with "
                             "zeros can use 'sqrt'.")
        return np.log(y)
    return y


def _check_cycles(cycles):
    if cycles is None:
        return ()
    if isinstance(cycles, str):
        raise TypeError(f"cycles must be a list of cycle names; got the string {cycles!r}.")
    cycles = list(cycles)
    unknown = [c for c in cycles if c not in CYCLES]
    if unknown:
        raise ValueError(f"Unknown cycle(s) {unknown}; expected a subset of {list(CYCLES)}.")
    return tuple(c for c in CYCLES if c in cycles)


# ---------------------------------------------------------------------------
# The numbers
# ---------------------------------------------------------------------------

def _detrend(z):
    t = np.arange(len(z), dtype=float)
    slope, intercept = np.polyfit(t, z, 1)
    return z - (intercept + slope * t)


def _periodogram(x):
    """Frequencies (cycles per observation, zero dropped) and power."""
    from scipy.signal import periodogram
    f, p = periodogram(x, fs=1.0, detrend=False, scaling="spectrum")
    return f[1:], p[1:]


def _bins_near(freqs, period):
    """Periods of the bin nearest 1/period and of its two neighbours."""
    i = int(np.argmin(np.abs(freqs - 1.0 / period)))
    idx = [j for j in (i - 1, i, i + 1) if 0 <= j < len(freqs)]
    return i, [round(1.0 / freqs[j], 2) for j in idx]


def _acf(x, max_lag):
    from statsmodels.tsa.stattools import acf
    return acf(x, nlags=max_lag, fft=True)


def _stl(z, period):
    """STL of the transformed series at an integer period, and the seasonal
    strength max(0, 1 - var(remainder) / var(seasonal + remainder))."""
    from statsmodels.tsa.seasonal import STL
    res = STL(z, period=period, robust=True).fit()
    resid, seas = np.asarray(res.resid), np.asarray(res.seasonal)
    strength = max(0.0, 1.0 - np.var(resid) / np.var(seas + resid))
    return res, float(strength)


def _shuffled_strength(z, period):
    """The seasonal strength of a shuffled copy of z: same values, same
    period, no seasonal pattern, so the statistic's level on this series
    length when there is nothing to find."""
    shuffled = np.random.default_rng(SHUFFLE_SEED).permutation(z)
    return _stl(shuffled, period)[1]


def _numbers(z, x, freq, cycles):
    n = len(z)
    periods = {c: cycle_period(c, freq) for c in cycles}
    freqs, power = _periodogram(x)
    lags = sorted({1, *[int(round(m)) for m in periods.values()]})
    usable = [lag for lag in lags if lag < n]
    rho = _acf(x, max(usable)) if usable else np.array([1.0])
    band = 1.96 / np.sqrt(n)

    per_cycle = {}
    for c, m in periods.items():
        i, near = _bins_near(freqs, m)
        lag = int(round(m))
        per_cycle[c] = {
            "m": round(m, 4),
            "periodogram_bins": near,
            "power_rank": int((power > power[i]).sum()) + 1,
            "acf_lag": lag,
            "acf": None if lag >= n else round(float(rho[lag]), 4),
        }
    top = int(np.argmax(power))
    numbers = {
        "n": n,
        "frequency_resolution": f"1/{n} cycles per observation",
        "acf_lag1": round(float(rho[1]), 4) if n > 1 else None,
        "acf_band": round(float(band), 4),
        "strongest_bin_period": round(1.0 / freqs[top], 2),
        "cycles": per_cycle,
        "stl": None,
    }

    stl = None
    if periods:
        longest = max(periods, key=periods.get)
        period = int(round(periods[longest]))
        if n < 2 * period:
            numbers["stl"] = {"cycle": longest, "period": period, "skipped": (
                f"{n} rows read, fewer than two periods ({2 * period}); "
                "no decomposition")}
        else:
            stl, strength = _stl(z, period)
            numbers["stl"] = {"cycle": longest, "period": period,
                              "seasonal_strength": round(strength, 4),
                              "shuffled_strength": round(_shuffled_strength(z, period), 4),
                              "shuffle_seed": SHUFFLE_SEED}
    return numbers, (freqs, power, rho, band, stl)


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------

def explore(df, date_col, outcome_col, before, cycles=("annual",), transform="sqrt",
            save_to=None):
    """Look at the series before declaring its design.

    Parameters
    ----------
    df : pd.DataFrame
    date_col, outcome_col : str
    before : date-like
        The boundary. Only rows strictly before it are read.
    cycles : list of str or None
        Candidate cycles to mark and test, a subset of ("weekly", "annual").
        None examines no cycle: the periodogram is drawn unmarked and no
        decomposition is made.
    transform : "sqrt", "log" or None
        Variance stabilisation applied before everything except the series
        panel, which shows the values as given.
    save_to : path or None
        A directory to write the figure and a JSON summary into. Nothing is
        written when None.

    Returns
    -------
    Exploration
        Pass it to Design.set_exploration so the sealed record carries the
        boundary and the number of rows read.
    """
    from .design import _show
    from .outputs.explore_plot import explore_figure

    cycles = _check_cycles(cycles)
    boundary, rows = _read_before(df, date_col, outcome_col, before)
    freq = resolve_frequency(rows[date_col])
    z = _transform(rows[outcome_col], transform)
    x = _detrend(z)
    numbers, parts = _numbers(z, x, freq, cycles)

    fig = explore_figure(rows[date_col], rows[outcome_col].to_numpy(dtype=float),
                         outcome_col, transform, numbers, *parts)
    result = Exploration(boundary=boundary, rows_read=len(rows),
                         first_date=pd.Timestamp(rows[date_col].iloc[0]),
                         last_date=pd.Timestamp(rows[date_col].iloc[-1]),
                         freq=freq.alias, transform=transform, cycles=cycles,
                         numbers=numbers,
                         fingerprint=fingerprint(rows, date_col, [outcome_col], boundary),
                         figure=fig)
    print(render_exploration(result))
    _show(fig)
    if save_to is not None:
        _save(result, save_to)
    return result


def _save(result, save_to):
    os.makedirs(save_to, exist_ok=True)
    with open(os.path.join(save_to, SUMMARY_FILE), "w") as fh:
        json.dump(result.to_dict(), fh, indent=2)
        fh.write("\n")
    result.figure.savefig(os.path.join(save_to, PLOT_FILE), dpi=150,
                          bbox_inches="tight")


def render_exploration(result):
    """The text explore() prints: what was read, the numbers, the cautions."""
    nb = result.numbers
    lines = [
        "EXPLORE",
        f"  boundary     {result.boundary.date()} (rows strictly before it)",
        f"  rows read    {result.rows_read}, {result.first_date.date()} to "
        f"{result.last_date.date()}, frequency {result.freq}",
        f"  transform    {result.transform or 'none'}, then a straight-line detrend",
        f"  resolution   {nb['frequency_resolution']}; strongest bin at period "
        f"{nb['strongest_bin_period']} observations",
        f"  acf lag 1    {nb['acf_lag1']}  (band +/- {nb['acf_band']})",
    ]
    if not result.cycles:
        lines.append("  cycles       none examined")
    for c, v in nb["cycles"].items():
        acf = ("not computable, lag beyond the rows read" if v["acf"] is None
               else f"{v['acf']}")
        lines.append(
            f"  {c:<12} m = {v['m']}; periodogram bins at periods "
            f"{', '.join(str(p) for p in v['periodogram_bins'])} "
            f"(the nearest ranks {v['power_rank']} by power); acf at lag "
            f"{v['acf_lag']} {acf}")
    s = nb["stl"]
    if s is not None:
        if "skipped" in s:
            lines.append(f"  stl          {s['cycle']} at period {s['period']}: {s['skipped']}")
        else:
            lines.append(f"  stl          {s['cycle']} at period {s['period']}: "
                         f"seasonal strength {s['seasonal_strength']}; on a shuffled "
                         f"copy of the same values (seed {s['shuffle_seed']}) "
                         f"{s['shuffled_strength']}")
    lines.append("")
    lines.append("CAUTIONS")
    for c in CAUTIONS:
        lines.append("  - " + c)
    return "\n".join(lines)
