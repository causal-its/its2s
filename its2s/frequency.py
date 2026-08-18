# Description: Series frequency resolution for ITS analysis (GH #48, #52).
#   Resolves the frequency of a time series ONCE, from the data, and proves the
#   series is a complete, regularly spaced grid. Observation counts equal
#   calendar spans only under that assumption, so the resolver fails loudly on
#   gaps, duplicates, and irregular spacing rather than letting downstream
#   window arithmetic go silently wrong. Also maps a resolved frequency to its
#   dominant seasonal cycle, the single mapping shared by metrics (MASE m,
#   GH #62) and diagnostics (key ACF lags and Ljung-Box depth, GH #61, #35).
# Usage: from its2s.frequency import resolve_frequency, SeriesFrequency
# Dependencies: pandas

from dataclasses import dataclass, field

import pandas as pd
from pandas.tseries.frequencies import to_offset


@dataclass(frozen=True)
class SeriesFrequency:
    """Resolved frequency of a regular, complete time series.

    Attributes
    ----------
    alias : str
        Pandas offset alias inferred from the data (e.g. "D", "W-SAT", "MS").
    offset : pd.DateOffset
        The corresponding pandas offset object, for consumers that need
        calendar arithmetic. Consumers that need window sizes should count
        observations (consecutive rows), not calendar time.
    """

    alias: str
    offset: pd.DateOffset = field(compare=False)

    @classmethod
    def from_alias(cls, alias):
        return cls(alias=alias, offset=to_offset(alias))


def resolve_frequency(dates):
    """Resolve the frequency of a date sequence and prove it is regular.

    The alias is inferred with ``pd.infer_freq`` and then verified by
    reconstructing the full expected grid with ``pd.date_range`` and comparing
    it to the actual dates. Inference alone is not proof; the reconstruction
    is what guarantees the series is complete, so that a window of n
    observations reliably spans n periods.

    Parameters
    ----------
    dates : sequence of datetime-like
        The series' date column, sorted ascending. Anything
        ``pd.DatetimeIndex`` accepts.

    Returns
    -------
    SeriesFrequency

    Raises
    ------
    ValueError
        If the dates contain missing values or duplicates, are unsorted, are
        too few to infer from (fewer than 3), or do not form a complete
        regular grid. Gaps created upstream (e.g. by dropping rows with
        missing outcomes) are reported here: for surveillance data a missing
        period usually means missing outcomes, not zero outcomes, and
        observation-counted windows on a gapped grid are wrong.
    """
    idx = pd.DatetimeIndex(dates)

    if idx.hasnans:
        raise ValueError(
            "Cannot resolve series frequency: the date column contains "
            "missing values (NaT)."
        )
    if len(idx) < 3:
        raise ValueError(
            f"Cannot resolve series frequency from {len(idx)} dates; at "
            "least 3 are required."
        )
    if idx.has_duplicates:
        first_dup = idx[idx.duplicated()][0]
        raise ValueError(
            f"Cannot resolve series frequency: duplicate date {first_dup} "
            "in the series."
        )
    if not idx.is_monotonic_increasing:
        raise ValueError(
            "Cannot resolve series frequency: dates are not sorted "
            "ascending."
        )

    alias = pd.infer_freq(idx)
    if alias is None:
        _raise_irregular(idx)

    expected = pd.date_range(start=idx[0], periods=len(idx), freq=alias)
    if not expected.equals(idx):
        # infer_freq found a pattern the full grid does not obey; report the
        # first departure just as for an uninferrable series.
        _raise_irregular(idx)

    return SeriesFrequency.from_alias(alias)


# Dominant seasonal cycle per frequency family: the shortest strong cycle on
# daily data, the annual cycle on weekly and monthly data. This is THE mapping
# behind seasonality: auto for metrics and the key ACF lag set for diagnostics;
# a second mapping anywhere else would drift.
_DOMINANT_CYCLES = {
    "D": 7,    # daily: day-of-week
    "W": 52,   # weekly: annual
    "M": 12,   # monthly: annual
    "MS": 12,
    "ME": 12,
}


def dominant_seasonal_period(series_freq):
    """Return the dominant seasonal period m for a resolved frequency.

    Parameters
    ----------
    series_freq : SeriesFrequency or None
        Output of resolve_frequency. None is accepted and returns None, so
        callers can treat "no frequency" and "no mapping" identically.

    Returns
    -------
    int or None
        The dominant cycle in observations (daily 7, weekly 52, monthly 12),
        or None when the frequency has no mapped cycle (e.g. quarterly,
        hourly). Callers must handle None loudly, never by silently
        substituting a period.
    """
    if series_freq is None:
        return None
    family = series_freq.alias.split("-")[0]
    return _DOMINANT_CYCLES.get(family)


def _raise_irregular(idx):
    """Raise ValueError naming the first irregularly spaced timestamp."""
    diffs = idx[1:] - idx[:-1]
    modal = diffs.value_counts().index[0]
    position = next(
        (i for i, d in enumerate(diffs) if d != modal), None,
    )
    detail = ""
    if position is not None:
        detail = (
            f" Most spacings are {modal}, but {idx[position + 1]} follows "
            f"{idx[position]} after {diffs[position]}."
        )
    raise ValueError(
        "Cannot resolve series frequency: the series is not a complete, "
        f"regularly spaced grid.{detail} If rows were removed (e.g. by "
        "dropping missing outcomes), the series has gaps; fill or "
        "aggregate the series to a regular grid before analysis."
    )
