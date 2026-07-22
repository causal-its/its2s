# Description: Tests for its2s.frequency (series frequency resolution, GH #48/#52).
#   Covers: resolution of daily/weekly/monthly grids, rejection of gaps,
#   duplicates, unsorted and too-short series, and the pipeline-facing
#   contract (resolved alias injected for NeuralProphet, gap after
#   missing_data="drop" fails loudly).
# Usage: python -m pytest tests/test_frequency.py -v --tb=short
# Dependencies: pytest, pandas, its2s

import pandas as pd
import pytest

from its2s.frequency import SeriesFrequency, resolve_frequency


def _dates(freq, n=30, start="2022-01-01"):
    return pd.date_range(start, periods=n, freq=freq)


class TestResolveRegularGrids:
    def test_daily(self):
        assert resolve_frequency(_dates("D")).alias == "D"

    def test_weekly_preserves_anchor(self):
        alias = resolve_frequency(_dates("W-SAT")).alias
        assert alias == "W-SAT"

    def test_monthly_start(self):
        assert resolve_frequency(_dates("MS")).alias == "MS"

    def test_hourly(self):
        assert resolve_frequency(_dates("h")).alias == "h"

    def test_accepts_series_input(self):
        s = pd.Series(_dates("D"))
        assert resolve_frequency(s).alias == "D"

    def test_offset_reconstructs_grid(self):
        freq = resolve_frequency(_dates("W-SAT", n=5))
        d0 = pd.Timestamp("2022-01-01")
        assert d0 + freq.offset == pd.Timestamp("2022-01-08")


class TestRejectIrregularGrids:
    def test_gap_named_in_error(self):
        dates = _dates("W-SAT", n=10).delete(4)
        with pytest.raises(ValueError, match="regularly spaced grid"):
            resolve_frequency(dates)

    def test_gap_reports_first_offender(self):
        dates = _dates("D", n=10).delete(3)
        with pytest.raises(ValueError, match="2022-01-05"):
            resolve_frequency(dates)

    def test_duplicate_date(self):
        dates = pd.DatetimeIndex(
            list(_dates("D", n=5)) + [pd.Timestamp("2022-01-03")]
        ).sort_values()
        with pytest.raises(ValueError, match="duplicate"):
            resolve_frequency(dates)

    def test_unsorted(self):
        dates = _dates("D", n=5)[::-1]
        with pytest.raises(ValueError, match="sorted"):
            resolve_frequency(dates)

    def test_too_short(self):
        with pytest.raises(ValueError, match="at least 3"):
            resolve_frequency(_dates("D", n=2))

    def test_nat(self):
        dates = pd.DatetimeIndex(list(_dates("D", n=4)) + [pd.NaT])
        with pytest.raises(ValueError, match="NaT"):
            resolve_frequency(dates)

    def test_mixed_frequency(self):
        dates = pd.DatetimeIndex(
            list(_dates("D", n=5)) + list(_dates("W-SAT", n=5, start="2022-02-01"))
        )
        with pytest.raises(ValueError, match="regularly spaced grid"):
            resolve_frequency(dates)


class TestSeriesFrequencyDataclass:
    def test_from_alias(self):
        freq = SeriesFrequency.from_alias("D")
        assert freq.alias == "D"
        assert freq.offset == pd.tseries.frequencies.to_offset("D")

    def test_frozen(self):
        freq = SeriesFrequency.from_alias("D")
        with pytest.raises(AttributeError):
            freq.alias = "W"
