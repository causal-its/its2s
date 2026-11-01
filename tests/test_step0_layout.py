# Description: Tests for the pure layout functions of step 0 (P11.04.03 C1):
#   canonical windows (windows.py), the end-anchored fold layout and seasonal
#   coverage (folds.py), check origins and the seasonal gap (origins.py), the
#   moved seed utility (seeds.py) and the import-free ARIMA period rule.
#   Reproduces the contract's worked examples SN-6 and SN-13 on synthetic
#   dates (OI-16).
# Usage: python -m pytest tests/test_step0_layout.py -v --tb=short
# Dependencies: pytest, pandas, its2s

import subprocess
import sys
import warnings

import pandas as pd
import pytest

from its2s.folds import cycles_seen, layout_folds
from its2s.frequency import SeriesFrequency, resolve_frequency
from its2s.models.families import arima_period
from its2s.origins import _place, max_seasonal_gap, place_origins
from its2s.seasonality import make_seasonality
from its2s.windows import event_position, resolve_split, split_form

EVENT = pd.Timestamp("2026-06-22")


def _grid(start=None, end=None, periods=None, freq="W-MON"):
    """A complete regular grid with its frequency set, as step 0 builds it."""
    dates = pd.date_range(start=start, end=end, periods=periods, freq=freq)
    return pd.DatetimeIndex(dates, freq=resolve_frequency(dates).alias)


@pytest.fixture
def weekly():
    """The contract's example: 181 pre-event weeks from 2023-01-02 to an event
    on 2026-06-22, then 20 post-event weeks."""
    dates = _grid(start="2023-01-02", periods=201)
    assert (dates < EVENT).sum() == 181 and dates[181] == EVENT
    return dates


@pytest.fixture
def annual():
    return make_seasonality(["annual"], SeriesFrequency.from_alias("W-MON"))


def _resolve(dates, H=3, event=EVENT, **typed):
    return resolve_split(split_form(typed), typed, dates, event, H)


def _ok(dates, **typed):
    split, issues = _resolve(dates, **typed)
    assert issues == [], issues
    return split


def _errors(dates, **typed):
    split, issues = _resolve(dates, **typed)
    assert split is None
    return " | ".join(i.message for i in issues if i.level == "error")


# ---------------------------------------------------------------------------
# windows.py
# ---------------------------------------------------------------------------

class TestForms:

    def test_three_forms_agree(self, weekly):
        # 0.80 of 181 is 144.8, which rounds half up to 145 (SN-13).
        splits = [
            _ok(weekly, train_obs=145, test_obs=36),
            _ok(weekly, train_obs=145),
            _ok(weekly, test_obs=36),
            _ok(weekly, train_end="2025-10-06", test_end="2026-06-15"),
            _ok(weekly, train_end="2025-10-06"),
            _ok(weekly, train_frac=0.80),
            _ok(weekly, test_frac=0.20),
            _ok(weekly, train_frac=0.80, test_frac=0.20),
        ]
        assert len({s.canonical for s in splits}) == 1
        s = splits[0]
        assert (s.train.n, s.test.n, s.post_event.n) == (145, 36, 3)
        assert s.train.start == weekly[0]
        assert s.train.end == s.test.start == pd.Timestamp("2025-10-13")
        assert s.test.end == s.post_event.start == EVENT
        assert s.unused is None

    def test_counts_and_dates_agree_on_sp1(self, weekly):
        a = _ok(weekly, train_obs=129, test_obs=52)
        b = _ok(weekly, train_end="2025-06-16", test_end="2026-06-15")
        assert a.canonical == b.canonical
        assert (a.train.n, a.test.n) == (129, 52)

    def test_typed_kept_verbatim(self, weekly):
        s = _ok(weekly, train_end="2025-06-16", test_end="2026-06-15")
        assert s.typed == {"train_end": "2025-06-16", "test_end": "2026-06-15"}
        assert s.form == "dates"

    def test_fraction_rounds_half_up(self):
        dates = _grid(start="2023-01-02", periods=11, freq="D")
        # 5 pre-event rows; 0.5 * 5 = 2.5 -> 3.
        s = _ok(dates, H=3, event=dates[5], train_frac=0.5)
        assert (s.train.n, s.test.n) == (3, 2)

    def test_mixing_forms_raises(self):
        with pytest.raises(TypeError, match="exactly one form"):
            split_form({"train_obs": 129, "test_end": "2026-06-15"})
        with pytest.raises(TypeError, match="got none"):
            split_form({"train_start": "2023-01-02"})
        with pytest.raises(TypeError, match="Unknown split argument"):
            split_form({"train_obs": 129, "holdout_obs": 3})


class TestOneConversion:

    def test_inclusive_end_becomes_next_grid_date(self, weekly):
        s = _ok(weekly, train_end="2025-06-16")
        assert s.train.end == pd.Timestamp("2025-06-23")
        assert weekly[(weekly >= s.train.start) & (weekly < s.train.end)][-1] == \
            pd.Timestamp("2025-06-16")

    def test_seven_day_window_holds_seven_rows(self):
        # The D-106 defect: an inclusive upper bound put eight daily rows in
        # a seven-day window. Here the test window is seven days.
        dates = _grid(start="2026-01-01", end="2026-03-31", freq="D")
        event = pd.Timestamp("2026-03-01")
        s = _ok(dates, H=7, event=event, train_end="2026-02-21")
        inside = dates[(dates >= s.test.start) & (dates < s.test.end)]
        assert s.test.n == len(inside) == 7
        post = dates[(dates >= s.post_event.start) & (dates < s.post_event.end)]
        assert s.post_event.n == len(post) == 7


class TestTrainStart:

    def test_each_form_counts_from_train_start(self, weekly):
        start = "2024-01-01"
        n_pre = int(((weekly >= pd.Timestamp(start)) & (weekly < EVENT)).sum())
        a = _ok(weekly, train_obs=n_pre - 36, train_start=start)
        b = _ok(weekly, train_end="2025-10-06", train_start=start)
        c = _ok(weekly, test_frac=36 / n_pre, train_start=start)
        assert a.canonical == b.canonical == c.canonical
        assert a.train.start == pd.Timestamp(start)
        assert (a.train.n, a.test.n) == (n_pre - 36, 36)

    def test_fraction_is_of_rows_used(self, weekly):
        s = _ok(weekly, train_frac=0.80, train_start="2024-01-01")
        n_pre = 181 - s.unused.n
        assert s.train.n == int(0.80 * n_pre + 0.5)

    def test_unused_rows_recorded(self, weekly):
        s = _ok(weekly, train_obs=100, train_start="2024-01-01")
        assert s.unused.start == weekly[0]
        assert s.unused.end == pd.Timestamp("2024-01-01")
        assert s.unused.n == 52

    def test_start_is_part_of_the_canonical_form(self, weekly):
        a = _ok(weekly, train_end="2025-06-16")
        b = _ok(weekly, train_end="2025-06-16", train_start="2024-01-01")
        assert a.test == b.test and a.canonical != b.canonical

    def test_off_grid_train_start(self, weekly):
        msg = _errors(weekly, train_obs=100, train_start="2024-01-03")
        assert "2024-01-01 and 2024-01-08" in msg

    def test_train_start_at_or_after_test_start(self, weekly):
        msg = _errors(weekly, train_end="2025-06-16", train_start="2025-06-23")
        assert "at or after the start of the test window" in msg
        msg = _errors(weekly, test_obs=36, train_start="2026-06-22")
        assert "no pre-event rows" in msg


class TestSplitValidation:

    def test_counts_must_account_for_every_row(self, weekly):
        assert "leave unaccounted" in _errors(weekly, train_obs=100, test_obs=52)
        assert "exceed" in _errors(weekly, train_obs=150, test_obs=52)

    def test_test_window_must_end_before_event(self, weekly):
        assert "one step before the event" in _errors(
            weekly, train_end="2025-06-16", test_end="2026-06-08")

    def test_test_end_alone_cannot_place_the_cut(self, weekly):
        assert "needs train_end" in _errors(weekly, test_end="2026-06-15")

    def test_off_grid_train_end_names_neighbours(self, weekly):
        msg = _errors(weekly, train_end="2025-09-14")
        assert "2025-09-08 and 2025-09-15" in msg

    def test_empty_windows(self, weekly):
        assert "test window is empty" in _errors(weekly, train_obs=181)
        assert "test window is empty" in _errors(weekly, train_end="2026-06-15")
        assert "first row" in _errors(weekly, train_frac=0.001)

    def test_bad_values(self, weekly):
        assert "whole number" in _errors(weekly, train_obs=12.5)
        assert "strictly between" in _errors(weekly, train_frac=1.2)
        assert "sum to 1" in _errors(weekly, train_frac=0.8, test_frac=0.3)

    def test_post_event_shorter_than_H(self, weekly):
        assert "fewer than H=30" in _errors(weekly, H=30, train_obs=129)

    def test_event_off_grid_names_neighbours(self, weekly):
        pos, issues = event_position(weekly, "2026-06-24")
        assert pos is None and "2026-06-22 and 2026-06-29" in issues[0].message
        assert event_position(weekly, EVENT) == (181, [])


# ---------------------------------------------------------------------------
# folds.py
# ---------------------------------------------------------------------------

class TestFolds:

    def test_sn13_smallest_fold(self, weekly, annual):
        sp1 = _ok(weekly, train_obs=129, test_obs=52)
        folds, issues = layout_folds(sp1.train, n_folds=4, fold_eval_obs=13)
        assert issues == []
        first = folds[0].train
        assert first.n == 77
        assert (first.start, first.end) == (pd.Timestamp("2023-01-02"),
                                            pd.Timestamp("2024-06-24"))
        assert round(cycles_seen(first, annual), 2) == 1.48

    def test_end_anchored_no_gap(self, weekly):
        # V-9: the last slice ends at the training window's end; every slice
        # starts where its training portion ends; slices are equal.
        train = _ok(weekly, train_obs=129).train
        folds, _ = layout_folds(train, n_folds=3, fold_eval_obs=6)
        assert folds[-1].evaluate.end == train.end
        for f in folds:
            assert f.train.start == train.start
            assert f.evaluate.start == f.train.end
            assert f.evaluate.n == 6
        assert [f.train.n for f in folds] == [111, 117, 123]
        for a, b in zip(folds, folds[1:]):
            assert a.evaluate.end == b.evaluate.start

    def test_slice_crossing_the_start(self, weekly):
        train = _ok(weekly, train_obs=20).train
        folds, issues = layout_folds(train, n_folds=4, fold_eval_obs=5)
        assert folds == [] and "crosses its start" in issues[0].message

    def test_cycles_seen_monthly_is_exact(self):
        dates = _grid(start="2020-01-01", periods=40, freq="MS")
        s = make_seasonality(["annual"], SeriesFrequency.from_alias("MS"))
        train = _ok(dates, H=3, event=dates[30], train_obs=24).train
        assert cycles_seen(train, s) == 2.0

    def test_cycles_seen_weekly_cycle_and_none(self):
        dates = _grid(start="2026-01-01", periods=60, freq="D")
        train = _ok(dates, H=3, event=dates[30], train_obs=14).train
        weekly_cycle = make_seasonality(["weekly"], SeriesFrequency.from_alias("D"))
        assert cycles_seen(train, weekly_cycle) == 2.0
        none = make_seasonality([], SeriesFrequency.from_alias("D"))
        assert cycles_seen(train, none) is None


# ---------------------------------------------------------------------------
# origins.py
# ---------------------------------------------------------------------------

def _weeks_from(test, dates):
    return [int((d - test.start).days // 7) for d in dates]


class TestOrigins:

    def test_sn6_full_year(self, weekly, annual):
        test = _ok(weekly, train_obs=129, test_obs=52).test
        o, issues = place_origins(test, 3, annual, EVENT, n_origins=4)
        assert issues == []
        assert _weeks_from(test, o.dates) == [0, 16, 33, 49]
        assert round(o.max_gap, 2) == 0.33
        # A year before the event falls within half a week of the first origin.
        assert o.phase_origin == test.start
        assert o.placed_by == "count"

    def test_sn6_spring_window_warns(self, weekly, annual):
        test = _ok(weekly, test_obs=13).test
        o, issues = place_origins(test, 3, annual, EVENT)
        assert _weeks_from(test, o.dates) == [0, 3, 7, 10]
        assert round(o.max_gap, 2) == 0.81
        assert o.phase_origin is None
        assert [i.level for i in issues] == ["warning"]

    def test_capacity(self, weekly, annual):
        test = _ok(weekly, test_obs=52).test
        o, issues = place_origins(test, 11, annual, EVENT, n_origins=5)
        assert o is None and "at least 55 rows" in issues[0].message

    def test_event_phase_origin_replaces_nearest_interior(self, weekly, annual):
        test = _ok(weekly, test_obs=78).test
        o, _ = place_origins(test, 3, annual, EVENT, n_origins=4)
        # Even spacing gives 0, 25, 50, 75; the anniversary sits at week 26.
        assert _weeks_from(test, o.dates) == [0, 26, 50, 75]
        anniversary = EVENT - pd.DateOffset(years=1)
        assert abs(o.phase_origin - anniversary) <= pd.Timedelta(days=3.5)

    def test_respacing_around_the_phase_origin(self):
        # Even 0,5,10,15,20; pinning 7 in place of 5 leaves 7 and 10 under
        # H=4 apart, so the other two are spread over 7..20.
        assert _place(n=5, last=20, H=4, p=7) == [0, 7, 11, 16, 20]
        assert _place(n=5, last=20, H=3, p=7) == [0, 7, 10, 15, 20]

    def test_no_placement_is_an_error(self):
        dates = _grid(start="2026-01-01", periods=60, freq="D")
        event = dates[40]
        weekly_cycle = make_seasonality(["weekly"], SeriesFrequency.from_alias("D"))
        test = _ok(dates, H=3, event=event, test_obs=18).test
        # Even 0,3,6,9,12,15; the event's weekday is at 11, within H of 9.
        o, issues = place_origins(test, 3, weekly_cycle, event, n_origins=6)
        assert o is None and "No placement" in issues[0].message

    def test_no_cycle_pins_nothing(self, weekly):
        none = make_seasonality([], SeriesFrequency.from_alias("W-MON"))
        test = _ok(weekly, test_obs=52).test
        o, issues = place_origins(test, 3, none, EVENT)
        assert o.phase_origin is None and o.max_gap is None and issues == []
        assert _weeks_from(test, o.dates) == [0, 16, 33, 49]

    def test_origin_dates(self, weekly, annual):
        test = _ok(weekly, test_obs=52).test
        given = ["2025-06-23", "2025-12-01", "2026-03-02"]
        o, issues = place_origins(test, 3, annual, EVENT, origin_dates=given)
        assert issues == [] or all(i.level == "warning" for i in issues)
        assert [d.date().isoformat() for d in o.dates] == given
        assert o.phase_origin == pd.Timestamp("2025-06-23")
        assert o.placed_by == "dates"

    def test_origin_dates_errors(self, weekly, annual):
        test = _ok(weekly, test_obs=52).test

        def errs(given):
            o, issues = place_origins(test, 3, annual, EVENT, origin_dates=given)
            assert o is None
            return " | ".join(i.message for i in issues)

        assert "declare 3 to 6" in errs(["2025-06-23", "2025-12-01"])
        assert "not a test-window date" in errs(["2025-06-23", "2025-12-03",
                                                 "2026-03-02"])
        assert "not a test-window date" in errs(["2025-06-23", "2025-12-01",
                                                 "2026-06-08"])
        assert "at least H=3 rows apart" in errs(["2025-06-23", "2025-06-30",
                                                  "2026-03-02"])

    def test_misuse_raises(self, weekly, annual):
        test = _ok(weekly, test_obs=52).test
        with pytest.raises(ValueError, match="3 to 6"):
            place_origins(test, 3, annual, EVENT, n_origins=2)
        with pytest.raises(TypeError, match="not both"):
            place_origins(test, 3, annual, EVENT, n_origins=4,
                          origin_dates=["2025-06-23"])

    def test_gap_wraps_the_year(self, annual):
        dates = ["2025-01-06", "2025-04-07", "2025-07-07", "2025-10-06"]
        # Phases 5, 96, 187, 278 days into the year; the wrap gap is 1 - 273/365.25.
        assert round(max_seasonal_gap(dates, annual), 4) == round(1 - 273 / 365.25, 4)


# ---------------------------------------------------------------------------
# seeds.py and the ARIMA period rule
# ---------------------------------------------------------------------------

class TestMoved:

    def test_derive_seed_moved_unchanged(self):
        from its2s.batch.seed_manager import derive_seed as old
        from its2s.seeds import derive_seed
        assert old is derive_seed
        # Pinned from batch/seed_manager.py at 8213060.
        assert derive_seed(42, "series_a") == 2976165223

    def test_arima_period(self):
        weekly_freq = SeriesFrequency.from_alias("W-SUN")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert arima_period(make_seasonality(["annual"], weekly_freq)) == 52
        assert arima_period(make_seasonality([], weekly_freq)) == 1
        daily = SeriesFrequency.from_alias("D")
        with pytest.raises(ValueError, match="one seasonal period"):
            arima_period(make_seasonality(["weekly", "annual"], daily))
        with pytest.raises(ValueError, match="cannot carry the annual cycle"):
            arima_period(make_seasonality(["annual"], daily))

    def test_layout_imports_no_heavy_library(self):
        code = ("import sys, its2s.windows, its2s.folds, its2s.origins, its2s.seeds, "
                "its2s.models.families; "
                "print([m for m in ('prophet', 'pmdarima', 'neuralprophet', "
                "'xgboost', 'torch') if m in sys.modules])")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                             text=True, check=True)
        assert out.stdout.strip() == "[]"
