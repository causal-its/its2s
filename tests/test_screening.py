# Description: Step 1's computation (its2s.screening): settings, folds as
#   fitted, seeds, recorded failures, criteria and ties. Synthetic data only.
#   Most tests run a cheap stand-in family so the suite stays fast; the tests
#   that are about real numerics (no outcome past the training window read,
#   candidate isolation) fit the real arima family.
# Usage: pytest tests/test_screening.py
# Dependencies: pytest, numpy, pandas, pmdarima

import copy
import warnings

import numpy as np
import pandas as pd
import pytest

import its2s.screening as screening
from its2s.design import DesignWarning, declare_design
from its2s.models.base import FitResult, PredictionResult
from its2s.models.families import FAMILIES
from its2s.records import canonical_json
from its2s.screening import mase_period, screen, screen_candidate
from its2s.search import nest_params, sample_settings
from its2s.settings import _deep_merge

from test_design import _design, _frame

TRAIN_END = "2025-06-23"   # sp1 below: 129 training weeks from 2023-01-02


def _declared(tmp, families=(("arima", 2),), n_folds=3, fold_eval_obs=6,
              cycles=("annual",), df=None):
    d = _design(df=df, tmp=str(tmp))
    d.set_seasonality(cycles=list(cycles))
    d.add_split("sp1", train_obs=129, test_obs=52, primary=True)
    d.add_cv("cv1", n_folds=n_folds, fold_eval_obs=fold_eval_obs)
    for name, n in families:
        d.add_family(name, n_trials=n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DesignWarning)
        return declare_design(d).to_dict()["handoff"]


# ---------------------------------------------------------------------------
# A stand-in family: forecasts the training mean plus a setting-driven shift
# ---------------------------------------------------------------------------

def _fake_family(fail=None, shift=None, log=None):
    """A family class that fits in microseconds.

    fail(kind, df, params) -> bool raises where it returns True; shift(params)
    moves the forecast so settings can score differently; log collects every
    fit and predict as (kind, first date, last date, n rows).
    """

    class Fake:
        spec = FAMILIES["prophet_xgb"]

        def __init__(self, params, seasonality):
            self.params = params
            self.seasonality = seasonality

        def _maybe_fail(self, kind, df):
            if fail is not None and fail(kind, df, self.params):
                raise RuntimeError(f"injected {kind} failure")

        def fit(self, train_df, target_col="y", date_col="ds", covariate_cols=None):
            if log is not None:
                log.append(("fit", train_df[date_col].iloc[0], train_df[date_col].iloc[-1],
                            len(train_df)))
            self._maybe_fail("fit", train_df)
            y = train_df[target_col].to_numpy(dtype=float)
            self.level = float(np.mean(y)) + (shift(self.params) if shift else 0.0)
            fitted = np.full(len(y), self.level)
            # The metadata an arima fit reports, for candidates named arima.
            return FitResult(fitted_values=fitted, residuals=y - fitted,
                             metadata={"order": (0, 0, 0), "seasonal_order": (0, 0, 0, 0)})

        def predict(self, target_df, target_col="y", date_col="ds", covariate_cols=None):
            if log is not None:
                log.append(("predict", target_df[date_col].iloc[0],
                            target_df[date_col].iloc[-1], len(target_df)))
            self._maybe_fail("predict", target_df)
            return PredictionResult(dates=target_df[date_col].to_numpy(),
                                    predicted=np.full(len(target_df), self.level))

        def seasonal_components(self):
            return {"fake": True}

    return Fake


@pytest.fixture
def use_family(monkeypatch):
    def install(cls):
        monkeypatch.setattr(screening, "load_family_class", lambda name: cls)
    return install


# ---------------------------------------------------------------------------
# The handoff and the data
# ---------------------------------------------------------------------------

class TestReads:

    def test_the_handoff_alone_gives_the_full_body(self, tmp_path, use_family):
        use_family(_fake_family())
        h = _declared(tmp_path, families=(("prophet_xgb", 3),))
        body = screen(copy.deepcopy(h), _frame())
        assert set(body) == {"first_fit_at", "combinations", "budgets", "candidates",
                             "multiplicity", "facts"}
        c = body["candidates"]["prophet_xgb/sp1/cv1"]
        assert c["status"] == "ok"
        for key in ("best", "whole_window", "default_reference", "search", "spread",
                    "criteria", "passed", "fitted_structure", "warnings"):
            assert c[key] is not None, key

    def test_no_outcome_past_the_training_window_is_read(self, tmp_path):
        """Real arima fits: blanking every outcome from the training window's
        end onward changes nothing in the body."""
        h = _declared(tmp_path)
        df = _frame()
        blanked = df.copy()
        blanked.loc[blanked["week"] >= pd.Timestamp(TRAIN_END), "deaths"] = np.nan
        a, b = screen(h, df), screen(h, blanked)
        a.pop("first_fit_at"), b.pop("first_fit_at")
        assert canonical_json(a) == canonical_json(b)
        # Control: the comparison can see a change. One outcome inside the
        # training window moved, and the body moves with it.
        moved = df.copy()
        moved.loc[moved["week"] == pd.Timestamp("2025-06-16"), "deaths"] += 50
        c = screen(h, moved)
        c.pop("first_fit_at")
        assert canonical_json(a) != canonical_json(c)

    def test_a_row_count_differing_from_the_seal_is_an_error(self, tmp_path, use_family):
        use_family(_fake_family())
        h = _declared(tmp_path)
        with pytest.raises(ValueError, match="holds 128 rows in the data and 129"):
            screen(h, _frame().drop(index=5))


# ---------------------------------------------------------------------------
# Folds as fitted
# ---------------------------------------------------------------------------

class TestFoldsAsFitted:

    def test_every_fit_and_forecast_is_on_the_sealed_rows(self, tmp_path, use_family):
        log = []
        use_family(_fake_family(log=log))
        h = _declared(tmp_path, families=(("prophet_xgb", 2),))
        screen(h, _frame())
        folds = h["combinations"]["sp1/cv1"]["folds"]
        dates = pd.date_range("2023-01-02", periods=129, freq="W-MON")
        expected = []
        for f in folds:
            n_fit = f["train"]["n"]
            expected.append(("fit", dates[0], dates[n_fit - 1], n_fit))
            expected.append(("predict", dates[n_fit], dates[n_fit + 5], 6))
        # Two settings and the default reference on the folds, then one
        # whole-window fit.
        assert log == expected * 3 + [("fit", dates[0], dates[-1], 129)]
        # The last evaluation slice ends at the last training row.
        assert expected[-1][2] == dates[-1]
        assert all(entry[2] < pd.Timestamp(TRAIN_END) for entry in log)


# ---------------------------------------------------------------------------
# Settings and seeds
# ---------------------------------------------------------------------------

class TestSettingsAndSeeds:

    def test_settings_are_drawn_with_the_candidate_seed(self, tmp_path, use_family):
        use_family(_fake_family())
        h = _declared(tmp_path, families=(("prophet_xgb", 4),))
        cand = h["candidates"]["prophet_xgb/sp1/cv1"]
        entry = screen(h, _frame())["candidates"]["prophet_xgb/sp1/cv1"]
        sample = {k: tuple(v) for k, v in cand["space"]["sample"].items()}
        drawn = sample_settings(sample, 4, cand["seed"])
        assert [t["setting"] for t in entry["search"]] == drawn

    def test_the_family_seed_is_the_candidate_seed(self, tmp_path, use_family):
        use_family(_fake_family())
        h = _declared(tmp_path, families=(("prophet_xgb", 2),))
        cand = h["candidates"]["prophet_xgb/sp1/cv1"]
        entry = screen(h, _frame())["candidates"]["prophet_xgb/sp1/cv1"]
        assert entry["best"]["params"]["xgb"]["random_state"] == cand["seed"]

    def test_params_are_complete(self, tmp_path, use_family):
        use_family(_fake_family())
        h = _declared(tmp_path, families=(("prophet_xgb", 2),))
        entry = screen(h, _frame())["candidates"]["prophet_xgb/sp1/cv1"]
        trial = entry["search"][entry["best"]["trial"]]["setting"]
        expected = _deep_merge(FAMILIES["prophet_xgb"].defaults, nest_params(trial))
        expected["xgb"]["random_state"] = entry["seed"]
        assert entry["best"]["params"] == expected

    def test_adding_a_family_leaves_another_candidate_unchanged(self, tmp_path_factory,
                                                                use_family):
        use_family(_fake_family(shift=lambda p: 0.0))
        alone = _declared(tmp_path_factory.mktemp("a"), families=(("arima", 2),))
        both = _declared(tmp_path_factory.mktemp("b"),
                         families=(("prophet_xgb", 2), ("arima", 2)))
        a = screen(alone, _frame())["candidates"]["arima/sp1/cv1"]
        b = screen(both, _frame())["candidates"]["arima/sp1/cv1"]
        assert canonical_json(a) == canonical_json(b)

    def test_a_candidate_screened_alone_matches_the_sweep(self, tmp_path):
        """Real arima fits, in a different order from the sweep's."""
        h = _declared(tmp_path, families=(("prophet_xgb", 0), ("arima", 2)))
        alone = screen_candidate(h, _frame(), "arima/sp1/cv1")
        swept = screen(h, _frame())["candidates"]["arima/sp1/cv1"]
        assert canonical_json(alone) == canonical_json(swept)
        assert alone["fitted_structure"]["whole_window"]["order"] is not None
        assert len(alone["fitted_structure"]["folds"]) == 3

    def test_no_search_scores_the_default(self, tmp_path, use_family):
        use_family(_fake_family())
        h = _declared(tmp_path, families=(("prophet_xgb", 0),))
        body = screen(h, _frame())
        entry = body["candidates"]["prophet_xgb/sp1/cv1"]
        assert entry["search"] is None and entry["default_reference"] is None
        assert entry["best"]["trial"] is None
        expected = copy.deepcopy(FAMILIES["prophet_xgb"].defaults)
        expected["xgb"]["random_state"] = entry["seed"]
        assert entry["best"]["params"] == expected
        assert len(entry["best"]["folds"]) == 3
        assert body["multiplicity"]["n_fits"] == 3 + 1


# ---------------------------------------------------------------------------
# Failures are recorded, never raised
# ---------------------------------------------------------------------------

def _second_fold(df):
    return len(df) == 129 - 2 * 6   # fold 2's training portion


class TestFailures:

    def test_a_setting_failing_on_one_fold_fails_the_trial(self, tmp_path, use_family):
        target = {}

        def fail(kind, df, params):
            # Fail trial 1 (the second drawn setting) on fold 2 only.
            return (kind == "fit" and _second_fold(df)
                    and params["xgb"]["max_depth"] == target["depth"])

        use_family(_fake_family(fail=fail, shift=lambda p: p["xgb"]["max_depth"] / 100))
        h = _declared(tmp_path, families=(("prophet_xgb", 3),))
        cand = h["candidates"]["prophet_xgb/sp1/cv1"]
        sample = {k: tuple(v) for k, v in cand["space"]["sample"].items()}
        target["depth"] = sample_settings(sample, 3, cand["seed"])[1]["xgb__max_depth"]
        entry = screen(h, _frame())["candidates"]["prophet_xgb/sp1/cv1"]
        failed = entry["search"][1]
        assert failed["status"] == "failed" and failed["fold"] == 2
        assert failed["mean_rmse"] is None
        assert "injected fit failure" in failed["error"]
        assert entry["best"]["trial"] != 1
        assert entry["spread"]["n_ok"] == 2

    def test_a_candidate_whose_every_setting_fails(self, tmp_path_factory, use_family):
        def fail(kind, df, params):
            return "random_state" in params.get("xgb", {})   # prophet_xgb only
        use_family(_fake_family(fail=fail))
        h = _declared(tmp_path_factory.mktemp("f"),
                      families=(("prophet_xgb", 2), ("arima", 2)))
        body = screen(h, _frame())
        bad = body["candidates"]["prophet_xgb/sp1/cv1"]
        assert bad["status"] == "failed"
        assert bad["failure"]["stage"] == "search"
        assert bad["passed"]["passed_all"] is False
        assert body["candidates"]["arima/sp1/cv1"]["status"] == "ok"
        assert body["multiplicity"]["n_failed"] == 1

    def test_a_failing_whole_window_fit(self, tmp_path, use_family):
        use_family(_fake_family(fail=lambda kind, df, p: kind == "fit" and len(df) == 129))
        h = _declared(tmp_path, families=(("prophet_xgb", 2),))
        entry = screen(h, _frame())["candidates"]["prophet_xgb/sp1/cv1"]
        assert entry["status"] == "failed"
        assert entry["failure"]["stage"] == "whole_window"
        assert "injected fit failure" in entry["failure"]["traceback"]


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

class TestCriteria:

    def test_s2_is_not_applicable_below_three_folds(self, tmp_path, use_family):
        use_family(_fake_family())
        h = _declared(tmp_path, families=(("prophet_xgb", 2),), n_folds=2)
        crit = screen(h, _frame())["candidates"]["prophet_xgb/sp1/cv1"]["criteria"]
        assert crit["S2"]["value"] is None
        assert "2 folds" in crit["S2"]["reason"]

    def test_s1_is_not_applicable_when_a_fold_is_shorter_than_m(self, tmp_path,
                                                                 use_family):
        use_family(_fake_family())
        # 129 - 3 x 26 = 51 rows in fold 1, not more than m = 52.
        h = _declared(tmp_path, families=(("prophet_xgb", 2),), fold_eval_obs=26)
        entry = screen(h, _frame())["candidates"]["prophet_xgb/sp1/cv1"]
        s1 = entry["criteria"]["S1"]
        assert s1["value"] is None and s1["verdict"] == "n/a"
        assert "51 rows, not more than m=52" in s1["reason"]
        assert entry["passed"]["n_applicable"] == 2   # S3 and S4 remain

    def test_a_deselected_criterion_keeps_its_value(self, tmp_path, use_family):
        use_family(_fake_family())
        h = _declared(tmp_path, families=(("prophet_xgb", 2),))
        crit = screen(h, _frame())["candidates"]["prophet_xgb/sp1/cv1"]["criteria"]
        assert crit["S2"]["selected"] is False
        assert crit["S2"]["value"] is not None
        assert crit["S2"]["verdict"] == "not-selected"
        assert all(c["defaulted"] is True for c in crit.values())

    def test_s4_and_s5_are_the_sealed_values(self, tmp_path, use_family):
        use_family(_fake_family())
        h = _declared(tmp_path, families=(("prophet_xgb", 2),))
        crit = screen(h, _frame())["candidates"]["prophet_xgb/sp1/cv1"]["criteria"]
        combo = h["combinations"]["sp1/cv1"]
        assert crit["S4"]["value"] == combo["S4"]["value"]
        assert crit["S5"]["value"] == combo["S5"]["value"]

    def test_mase_period_is_the_longest_cycle(self, tmp_path):
        from its2s.frequency import SeriesFrequency
        from its2s.seasonality import make_seasonality
        weekly = SeriesFrequency.from_alias("W-MON")
        daily = SeriesFrequency.from_alias("D")
        assert mase_period(make_seasonality(["annual"], weekly)) == 52
        assert mase_period(make_seasonality(["weekly", "annual"], daily)) == 365
        assert mase_period(make_seasonality(["weekly"], daily)) == 7
        assert mase_period(make_seasonality([], weekly)) == 1


# ---------------------------------------------------------------------------
# The choice
# ---------------------------------------------------------------------------

class TestChoice:

    def test_ties_go_to_the_lower_trial(self, tmp_path, use_family):
        use_family(_fake_family())   # every setting forecasts the same
        h = _declared(tmp_path, families=(("prophet_xgb", 4),))
        entry = screen(h, _frame())["candidates"]["prophet_xgb/sp1/cv1"]
        assert len({t["mean_rmse"] for t in entry["search"]}) == 1
        assert entry["best"]["trial"] == 0

    def test_lowest_mean_rmse_is_chosen(self, tmp_path, use_family):
        use_family(_fake_family(shift=lambda p: p["xgb"]["max_depth"]))
        h = _declared(tmp_path, families=(("prophet_xgb", 5),))
        entry = screen(h, _frame())["candidates"]["prophet_xgb/sp1/cv1"]
        means = [t["mean_rmse"] for t in entry["search"]]
        assert entry["best"]["trial"] == int(np.argmin(means))
        assert entry["best"]["mean_rmse"] == min(means)
