# Description: The check origins of step 2, declared and placed at step 0
#   (contract P11.04 section 5.9, SN-6; spec_P11.04.03 4.3 and OQ-11). An
#   origin is a grid date o: the check fit uses every row before o and
#   forecasts o to o + H - 1. Placement under a count spaces origins evenly
#   from the training window's end to H before the test window's end and pins
#   one at the EVENT-PHASE ORIGIN; seasonal coverage is reported as the
#   maximum seasonal gap.
# Usage: from its2s.origins import place_origins, max_seasonal_gap
# Dependencies: numpy, pandas

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .folds import longest_cycle
from .seasonality import _sampling_interval
from .windows import Issue, _round_half_up

N_ORIGINS_DEFAULT = 4
N_ORIGINS_RANGE = (3, 6)
GAP_WARN = 0.5


@dataclass(frozen=True)
class Origins:
    """Where step 2 forecasts from.

    phase_origin : the origin at the event's point in the seasonal cycle,
        or None (no cycle declared, or no test date at that phase)
    max_gap : largest fraction of the longest cycle no origin tests, or None
        when no cycle is declared
    """

    dates: tuple
    placed_by: str            # "count" | "dates"
    phase_origin: pd.Timestamp | None
    max_gap: float | None


def _phase(date, cycle, unit):
    """A date's fractional position in a cycle: day of year / 365.25 for
    annual (month / 12 on month-based sampling), day of week / 7 for weekly."""
    if cycle == "weekly":
        return date.day_of_week / 7
    if unit == "months":
        return (date.month - 1) / 12
    return (date.day_of_year - 1) / 365.25


def _at_event_phase(event, seasonality):
    """A predicate: is a grid date at the event's phase (OQ-11)?

    Annual: within half a sampling interval of an anniversary of the event,
    one year back or more (month-based sampling: the same calendar month).
    Weekly only: the same day of the week.
    """
    cycle = longest_cycle(seasonality)
    unit, n = _sampling_interval(seasonality.freq)
    if cycle == "weekly":
        return lambda d: d.day_of_week == event.day_of_week and d < event
    if unit == "months":
        return lambda d: d.month == event.month and d < event
    tol = pd.Timedelta(days=n / 2)

    def at_phase(d):
        for k in (event.year - d.year - 1, event.year - d.year, event.year - d.year + 1):
            if k >= 1 and abs(d - (event - pd.DateOffset(years=k))) <= tol:
                return True
        return False
    return at_phase


def max_seasonal_gap(dates, seasonality):
    """The largest fraction of the longest declared cycle that falls between
    two successive origins' phases, the wrap included; None with no cycle."""
    cycle = longest_cycle(seasonality)
    if cycle is None:
        return None
    unit, _ = _sampling_interval(seasonality.freq)
    phases = sorted(_phase(pd.Timestamp(d), cycle, unit) for d in dates)
    gaps = [b - a for a, b in zip(phases, phases[1:])]
    gaps.append(1 - (phases[-1] - phases[0]))
    return max(gaps)


def _spaced(positions, H):
    return all(b - a >= H for a, b in zip(positions, positions[1:]))


def _even(a, b, r):
    """r positions spaced evenly strictly between a and b."""
    return [_round_half_up(x) for x in np.linspace(a, b, r + 2)[1:-1]]


def _place(n, last, H, p):
    """Positions 0..last for n origins, the phase position p pinned.

    Even spacing first; p replaces the nearest interior position; if that
    leaves two origins under H apart, the other interior origins are spread
    evenly over the stretches either side of p, choosing the split between
    the stretches with the widest smallest spacing. None if nothing fits.
    """
    pos = [_round_half_up(x) for x in np.linspace(0, last, n)]
    if p is None or p in pos:
        return pos
    nearest = min(pos[1:-1], key=lambda q: (abs(q - p), q))
    pinned = sorted([q for q in pos if q != nearest] + [p])
    if _spaced(pinned, H):
        return pinned
    best, best_gap = None, -1
    r = n - 3
    for left in range(r + 1):
        cand = sorted({0, p, last, *_even(0, p, left), *_even(p, last, r - left)})
        if len(cand) == n and _spaced(cand, H):
            gap = min(b - a for a, b in zip(cand, cand[1:]))
            if gap > best_gap:
                best, best_gap = cand, gap
    return best


def place_origins(test, H, seasonality, event_date, n_origins=None, origin_dates=None):
    """Place step 2's origins in the test window.

    Parameters
    ----------
    test : Window
        The test window.
    H : int
        The post-event horizon; origins must be at least H apart.
    seasonality : Seasonality
    event_date : date-like
    n_origins : int, optional
        Count, three to six; default four. Exclusive with origin_dates.
    origin_dates : sequence of date-like, optional
        Explicit origins; the event-phase origin is reported, never inserted.

    Returns
    -------
    (Origins or None, list of Issue)
    """
    if n_origins is not None and origin_dates is not None:
        raise TypeError("Give n_origins or origin_dates, not both.")
    event = pd.Timestamp(event_date)
    grid = [test.date(k) for k in range(test.n)]
    last = test.n - H
    lo, hi = N_ORIGINS_RANGE
    at_phase = (_at_event_phase(event, seasonality)
                if longest_cycle(seasonality) else None)

    if origin_dates is None:
        n = N_ORIGINS_DEFAULT if n_origins is None else n_origins
        if not lo <= n <= hi:
            raise ValueError(f"n_origins must be {lo} to {hi}; got {n}.")
        if n * H > test.n:
            return None, [Issue("error", f"{n} check origins at H={H} need a test window "
                                         f"of at least {n * H} rows; it holds {test.n}.")]
        p = None
        if at_phase is not None:
            p = next((k for k in range(last, -1, -1) if at_phase(grid[k])), None)
        positions = _place(n, last, H, p)
        if positions is None:
            return None, [Issue("error", f"No placement of {n} origins at least H={H} "
                                         "apart includes the event-phase origin "
                                         f"{grid[p].date()}; declare fewer origins, or "
                                         "origin_dates.")]
        dates = tuple(grid[k] for k in positions)
        phase_origin = grid[p] if p is not None else None
        placed_by = "count"
    else:
        dates = tuple(sorted(pd.Timestamp(d) for d in origin_dates))
        issues = []
        if not lo <= len(dates) <= hi:
            issues.append(Issue("error", f"origin_dates holds {len(dates)} dates; "
                                         f"declare {lo} to {hi}."))
        index = {d: k for k, d in enumerate(grid)}
        positions = []
        for d in dates:
            k = index.get(d)
            if k is None or k > last:
                issues.append(Issue("error", f"origin {d.date()} is not a test-window date "
                                             f"from {grid[0].date()} to "
                                             f"{grid[max(last, 0)].date()}, the last date "
                                             f"leaving H={H} rows to forecast."))
            else:
                positions.append(k)
        if len(positions) == len(dates) and not _spaced(positions, H):
            issues.append(Issue("error", f"origin_dates must be at least H={H} rows "
                                         "apart, or their forecasts overlap."))
        if len(dates) * H > test.n:
            issues.append(Issue("error", f"{len(dates)} check origins at H={H} need a test "
                                         f"window of at least {len(dates) * H} rows; it "
                                         f"holds {test.n}."))
        if issues:
            return None, issues
        phase_origin = (next((d for d in reversed(dates) if at_phase(d)), None)
                        if at_phase is not None else None)
        placed_by = "dates"

    gap = max_seasonal_gap(dates, seasonality)
    issues = []
    if gap is not None and gap > GAP_WARN:
        issues.append(Issue("warning", f"The check origins leave {gap:.2f} of the "
                                       f"{longest_cycle(seasonality)} cycle untested "
                                       f"(above {GAP_WARN}); the check says little about "
                                       "the seasons it misses."))
    return Origins(dates=dates, placed_by=placed_by, phase_origin=phase_origin,
                   max_gap=gap), issues
