# Description: Tests for step 0's declaration (P11.04.03 C2): the Design
#   object and its API-misuse errors, the criteria table, the grouped
#   validation, the one rendering and preview. Reproduces the contract's
#   failed-declaration example (SN-13) on synthetic data (OI-16). Nothing
#   here fits a model.
# Usage: python -m pytest tests/test_design.py -v --tb=short
# Dependencies: pytest, numpy, pandas, its2s

import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from its2s.design import (CLOSING_LINE, CRITERIA, Design, FrozenDesignError,
                          render_text, resolve)

EVENT = "2026-06-22"


def _frame(start="2023-01-02", periods=201, freq="W-MON", seed=0):
    """The contract's example series: 181 pre-event weeks, then 20 more."""
    dates = pd.date_range(start, periods=periods, freq=freq)
    y = np.random.default_rng(seed).poisson(100, periods).astype(float)
    return pd.DataFrame({"week": dates, "deaths": y})


def _design(df=None, H=3, event=EVENT, tmp="results/", **kw):
    df = _frame() if df is None else df
    return Design(df, "week", "deaths", kw.pop("covariates", None), event, H, 42,
                  tmp, "euromomo_heat2026")


def _contract_example(d):
    """Contract 5.2 as amended by v5, without sp3."""
    d.set_seasonality(cycles=["annual"])
    d.add_split("sp1", train_obs=129, test_obs=52, primary=True)
    d.add_split("sp2", train_end="2025-10-06", test_end="2026-06-15")
    d.add_cv("cv1", n_folds=2, fold_eval_obs=11)
    d.add_cv("cv2", n_folds=3, fold_eval_obs=6)
    d.add_family("prophet_xgb", n_trials=100)
    d.add_family("arima", n_trials=40)
    d.set_criteria(S1=1.0, S3=2.0, C2=1.5)
    d.set_check(n_origins=4)
    d.add_sub_period("acute", "2026-06-22", "2026-07-06")
    return d


def _minimal(d):
    d.add_split("sp1", train_obs=129, test_obs=52)
    d.add_cv("cv1", n_folds=2, fold_eval_obs=3)
    d.add_family("prophet_xgb", n_trials=10)
    return d


def _findings(d, level=None, title=None):
    _, findings = resolve(d)
    return [f for f in findings
            if (level is None or f.level == level)
            and (title is None or f.title == title)]


def _text(fs):
    return " | ".join(f.message for f in fs)


# ---------------------------------------------------------------------------
# The contract's example, and SN-13's failed declaration
# ---------------------------------------------------------------------------

class TestContractExample:

    def test_seals_clean(self):
        d = _contract_example(_design())
        assert _findings(d, "error") == []

    def test_resolved_is_json(self):
        resolved, _ = resolve(_contract_example(_design()))
        assert json.loads(json.dumps(resolved)) == resolved

    def test_sn13_failed_declaration(self):
        # sp3 duplicates sp2; cv3 (4 x 13) leaves every split's smallest fold
        # too short for a seasonal arima and thin on cycles; 9 combinations
        # exceed the figure. The full account is 2 errors and 4 warnings,
        # all reported together.
        d = _contract_example(_design())
        d.add_split("sp3", train_frac=0.80)
        d.add_cv("cv3", n_folds=4, fold_eval_obs=13)
        assert len(_findings(d, "error")) == 2
        assert len(_findings(d, "warning")) == 4
        assert len(_findings(d, "warning", "figure")) == 1
        assert len(_findings(d, "warning", "thin folds")) == 3
        cap = _findings(d, "error", "family arima")
        assert len(cap) == 1 and "104" in cap[0].message
        dup = _findings(d, "error", "duplicate split")
        assert len(dup) == 1
        msg = dup[0].message
        assert "sp3" in msg and "sp2" in msg
        assert "181 pre-event" in msg and "145 training and 36 test" in msg
        thin = [f for f in _findings(d, "warning", "thin folds")
                if "on sp1" in f.message]
        assert len(thin) == 1
        assert "77 of 129" in thin[0].message and "1.48" in thin[0].message
        assert "2023-01-02 to 2024-06-17" in thin[0].message

    def test_failed_preview_ends_with_closing_line(self, tmp_path, capsys):
        d = _contract_example(_design(tmp=tmp_path))
        d.add_split("sp3", train_frac=0.80)
        findings = d.preview()
        out = capsys.readouterr().out
        assert any(f.level == "error" for f in findings)
        assert out.rstrip().endswith(CLOSING_LINE)
        assert list(tmp_path.iterdir()) == []

    def test_clean_preview_has_no_closing_line(self, tmp_path, capsys):
        _contract_example(_design(tmp=tmp_path)).preview()
        assert CLOSING_LINE not in capsys.readouterr().out
        assert list(tmp_path.iterdir()) == []

    def test_preview_prints_render_text(self, capsys):
        d = _contract_example(_design())
        d.preview()
        resolved, _ = resolve(d)
        assert capsys.readouterr().out == render_text(resolved) + "\n"

    def test_render_works_from_json_alone(self):
        resolved, _ = resolve(_contract_example(_design()))
        assert render_text(json.loads(json.dumps(resolved))) == render_text(resolved)


# ---------------------------------------------------------------------------
# API misuse raises at the call
# ---------------------------------------------------------------------------

class TestMisuse:

    def test_constructor_types(self):
        df = _frame()
        with pytest.raises(TypeError):
            Design(df.to_dict(), "week", "deaths", None, EVENT, 3, 42, "r", "x")
        with pytest.raises(TypeError):
            Design(df, "week", "deaths", None, EVENT, 3.0, 42, "r", "x")
        with pytest.raises(ValueError):
            Design(df, "week", "deaths", None, EVENT, 0, 42, "r", "x")
        with pytest.raises(TypeError):
            Design(df, "week", "deaths", "temp", EVENT, 3, 42, "r", "x")

    def test_data_is_copied(self):
        df = _frame()
        d = _minimal(_design(df))
        df.loc[10, "deaths"] = np.nan
        assert _findings(d, "error") == []

    def test_duplicate_and_absent_ids(self):
        d = _minimal(_design())
        with pytest.raises(ValueError, match="already declared"):
            d.add_split("sp1", train_obs=100)
        with pytest.raises(ValueError, match="already declared"):
            d.add_cv("cv1", n_folds=2, fold_eval_obs=3)
        with pytest.raises(ValueError, match="already declared"):
            d.add_family("prophet_xgb", n_trials=10)
        with pytest.raises(ValueError, match="No split"):
            d.remove_split("sp9")
        with pytest.raises(ValueError, match="No cv design"):
            d.remove_cv("cv9")
        with pytest.raises(ValueError, match="No family"):
            d.remove_family("arima")

    def test_ids_may_not_contain_slash(self):
        with pytest.raises(ValueError, match="'/'"):
            _design().add_split("sp/1", train_obs=100)

    def test_split_forms_may_not_mix(self):
        with pytest.raises(TypeError, match="exactly one form"):
            _design().add_split("sp1", train_obs=100, train_frac=0.5)

    def test_two_primaries(self):
        d = _design()
        d.add_split("sp1", train_obs=100, primary=True)
        with pytest.raises(ValueError, match="already primary"):
            d.add_split("sp2", train_obs=110, primary=True)

    def test_no_gap_on_add_cv(self):
        # V-9: there is no gap argument at launch.
        with pytest.raises(TypeError):
            _design().add_cv("cv1", n_folds=2, fold_eval_obs=3, gap=1)

    def test_check_arguments(self):
        d = _design()
        with pytest.raises(TypeError, match="not both"):
            d.set_check(n_origins=4, origin_dates=["2025-07-07"])
        with pytest.raises(ValueError, match="3 to 6"):
            d.set_check(n_origins=7)

    def test_unknown_criterion(self):
        with pytest.raises(TypeError, match="Unknown criterion"):
            _design().set_criteria(S9=1.0)

    def test_unknown_cycle(self):
        with pytest.raises(ValueError, match="Unknown cycle"):
            _design().set_seasonality(cycles=["monthly"])

    def test_frozen_design_refuses_every_mutator(self):
        d = _minimal(_design())
        d._frozen = True       # what declare_design sets once sealed
        calls = [
            lambda: d.set_seasonality(None),
            lambda: d.add_split("sp2", train_obs=100),
            lambda: d.remove_split("sp1"),
            lambda: d.add_cv("cv2", n_folds=2, fold_eval_obs=3),
            lambda: d.remove_cv("cv1"),
            lambda: d.add_family("arima", n_trials=10),
            lambda: d.remove_family("prophet_xgb"),
            lambda: d.set_criteria(S2=True),
            lambda: d.set_check(n_origins=3),
            lambda: d.add_sub_period("a", EVENT, EVENT),
            lambda: d.set_exploration(SimpleNamespace(boundary=EVENT)),
            lambda: d.set_plot(True),
        ]
        for call in calls:
            with pytest.raises(FrozenDesignError, match="frozen"):
                call()
        # Still a ValueError, so catching the API's misuse errors catches it.
        assert issubclass(FrozenDesignError, ValueError)


# ---------------------------------------------------------------------------
# Validation groups: findings, never raised
# ---------------------------------------------------------------------------

class TestData:

    def test_missing_column(self):
        d = _minimal(_design(covariates=["temp"]))
        assert "['temp']" in _text(_findings(d, "error", "data"))

    def test_event_off_grid(self):
        d = _minimal(_design(event="2026-06-24"))
        assert "2026-06-22 and 2026-06-29" in _text(_findings(d, "error", "data"))

    def test_fewer_than_H_rows_after_event(self):
        d = _minimal(_design(H=30))
        assert "fewer than H=30" in _text(_findings(d, "error", "data"))

    def test_missing_outcome_in_a_window_is_an_error(self):
        df = _frame()
        df.loc[[5, 150, 182], "deaths"] = np.nan
        msg = _text(_findings(_minimal(_design(df)), "error", "data"))
        assert "sp1 training: 1" in msg and "sp1 test: 1" in msg
        assert "sp1 post-event: 1" in msg

    def test_missing_outcome_before_train_start_is_not(self):
        df = _frame()
        df.loc[5, "deaths"] = np.nan
        d = _design(df)
        d.add_split("sp1", train_obs=100, train_start="2024-01-01")
        d.add_cv("cv1", n_folds=2, fold_eval_obs=3)
        d.add_family("prophet_xgb", n_trials=10)
        assert _findings(d, "error") == []

    def test_irregular_grid(self):
        df = _frame().drop(index=40)
        assert "regularly spaced" in _text(_findings(_minimal(_design(df)), "error"))


class TestNotChecked:
    # User ruling 2026-11-01 (C2 judgment call 5): skipped groups are named.

    def test_offgrid_event_names_what_was_skipped(self):
        d = _minimal(_design(event="2026-06-24"))
        d.add_split("bad", train_obs=500, test_obs=52)
        d.add_sub_period("acute", "2026-06-25", "2026-07-08")
        resolved, _ = resolve(d)
        groups = [g for nc in resolved["report_fields"]["not_checked"] for g in nc["groups"]]
        for g in ("splits", "cross-validation", "check origins", "sub-periods"):
            assert g in groups
        text = render_text(resolved)
        assert "NOT CHECKED" in text
        assert "event date" in text
        assert text.rstrip().endswith(CLOSING_LINE)

    def test_unresolvable_grid_names_seasonality_and_families(self):
        df = _frame().drop(index=[10, 11])          # a gap: no regular grid
        d = _minimal(_design(df=df))
        resolved, _ = resolve(d)
        groups = [g for nc in resolved["report_fields"]["not_checked"] for g in nc["groups"]]
        assert "seasonality" in groups and "family search spaces" in groups
        assert "date grid" in resolved["report_fields"]["not_checked"][0]["reason"]
        assert "NOT CHECKED" in render_text(resolved)

    def test_valid_design_has_no_section(self):
        resolved, _ = resolve(_minimal(_design()))
        assert resolved["report_fields"]["not_checked"] == []
        assert "NOT CHECKED" not in render_text(resolved)


class TestSeasonality:

    def test_default_is_annual_and_stamped(self):
        resolved, _ = resolve(_minimal(_design()))
        assert resolved["seasonality"]["cycles"] == ["annual"]
        assert resolved["seasonality"]["declared"] is False
        assert "DEFAULT, not declared" in render_text(resolved)

    def test_aseasonal(self):
        d = _minimal(_design())
        d.set_seasonality(None)
        resolved, findings = resolve(d)
        assert [f for f in findings if f.level == "error"] == []
        combo = resolved["combinations"]["sp1/cv1"]
        assert combo["S4"] == {"value": None, "verdict": "not applicable"}
        origins = resolved["check"]["splits"]["sp1"]
        assert origins["phase_origin"] is None and origins["max_gap"] is None
        # User ruling 2026-11-01 (C2 judgment call 6): the contract's phrasing.
        text = render_text(resolved)
        assert "not willing to assert a cycle" in text
        assert "aseasonal" not in text

    def test_empty_cycles_list_is_refused(self):
        with pytest.raises(ValueError, match="cycles=None.*not willing to assert"):
            _design().set_seasonality([])

    def test_daily_without_annual_warns(self):
        df = _frame(start="2025-01-01", periods=540, freq="D")
        d = Design(df, "week", "deaths", None, "2026-06-01", 7, 1, "r", "x")
        d.set_seasonality(cycles=["weekly"])
        d.add_split("sp1", test_obs=60)
        d.add_cv("cv1", n_folds=3, fold_eval_obs=7)
        d.add_family("prophet_xgb", n_trials=10)
        assert "no annual" in _text(_findings(d, "warning", "seasonality"))

    def test_uncarryable_cycle(self):
        d = _minimal(_design())
        d.set_seasonality(cycles=["weekly"])
        assert "cannot be carried" in _text(_findings(d, "error", "seasonality"))


class TestSplits:

    def test_none_declared(self):
        d = _design()
        d.add_cv("cv1", n_folds=2, fold_eval_obs=3)
        d.add_family("prophet_xgb", n_trials=10)
        assert "No split" in _text(_findings(d, "error", "splits"))

    def test_several_without_primary(self):
        d = _minimal(_design())
        d.add_split("sp2", train_obs=145)
        assert "none is primary" in _text(_findings(d, "error", "splits"))

    def test_one_split_is_primary_implicitly(self):
        resolved, _ = resolve(_minimal(_design()))
        assert resolved["splits"]["sp1"]["primary"] is True

    def test_off_grid_typed_date(self):
        d = _minimal(_design())
        d.add_split("sp2", train_end="2025-09-14", primary=True)
        msg = _text(_findings(d, "error", "split sp2"))
        assert "2025-09-08 and 2025-09-15" in msg

    def test_train_start_rows_listed_and_rendered(self):
        d = _design()
        d.add_split("sp1", train_end="2025-06-16", primary=True)
        d.add_split("sp2", train_end="2025-06-16", train_start="2024-01-01")
        d.add_cv("cv1", n_folds=2, fold_eval_obs=3)
        d.add_family("prophet_xgb", n_trials=10)
        resolved, findings = resolve(d)
        assert [f for f in findings if f.level == "error"] == []
        unused = resolved["splits"]["sp2"]["unused"]
        assert unused["start"] == "2023-01-02" and unused["n"] == 52
        assert resolved["splits"]["sp1"]["unused"] is None
        assert "before train_start" in render_text(resolved)

    def test_exploration_boundary_warning(self):
        d = _minimal(_design())
        d.set_exploration(SimpleNamespace(boundary=pd.Timestamp("2025-09-01"),
                                          rows_read=139))
        msg = _text(_findings(d, "warning", "exploration"))
        assert "sp1" in msg and "2025-09-01" in msg
        d.set_exploration(SimpleNamespace(boundary=pd.Timestamp("2025-06-23")))
        assert _findings(d, "warning", "exploration") == []


class TestCombinations:

    def test_fold_layout_too_long(self):
        d = _design()
        d.add_split("sp1", train_obs=129, test_obs=52)
        d.add_cv("cv1", n_folds=20, fold_eval_obs=13)
        d.add_family("prophet_xgb", n_trials=10)
        assert "crosses its start" in _text(_findings(d, "error", "cv cv1 on sp1"))

    def test_s2_with_two_folds_warns_only_when_selected(self):
        d = _minimal(_design())
        assert _findings(d, "warning", "fold stability") == []
        d.set_criteria(S2=True)
        assert "cv1" in _text(_findings(d, "warning", "fold stability"))

    def test_s5_band(self):
        d = _minimal(_design())
        d.add_cv("cv2", n_folds=2, fold_eval_obs=11)
        resolved, _ = resolve(d)
        assert resolved["combinations"]["sp1/cv2"]["S5"]["verdict"] == "not selected"
        d.set_criteria(S5=True)
        resolved, findings = resolve(d)
        assert resolved["combinations"]["sp1/cv2"]["S5"]["verdict"] == "FAIL"
        assert resolved["combinations"]["sp1/cv1"]["S5"]["verdict"] == "PASS"
        assert "cv2" in _text([f for f in findings if f.title == "fold horizon"])

    def test_s5_band_is_inclusive_and_printed_closed(self):
        # User ruling 2026-11-01: S5 passes at both ends; printed [lo, hi].
        d = _design(H=4)
        d.add_split("sp1", train_obs=129, test_obs=52)
        for cid, k in (("lo", 2), ("hi", 8), ("under", 1), ("over", 9)):
            d.add_cv(cid, n_folds=2, fold_eval_obs=k)
        d.add_family("prophet_xgb", n_trials=10)
        d.set_criteria(S5=True)
        resolved, _ = resolve(d)
        verdicts = {c: resolved["combinations"][f"sp1/{c}"]["S5"]["verdict"]
                    for c in ("lo", "hi", "under", "over")}
        assert verdicts == {"lo": "PASS", "hi": "PASS",
                            "under": "FAIL", "over": "FAIL"}
        text = render_text(resolved)
        assert "[0.5, 2.0]" in text
        assert "0.5 to 2.0" not in text

    def test_figure_limit(self):
        d = _minimal(_design())
        d.add_split("sp2", train_obs=140, primary=True)
        for k in range(2, 5):
            d.add_cv(f"cv{k}", n_folds=2, fold_eval_obs=k)
        assert len(_findings(d, "warning", "figure")) == 1
        d.set_plot(all_combinations=True)
        assert _findings(d, "warning", "figure") == []


class TestFamilies:

    def test_neuralprophet_withheld(self):
        d = _minimal(_design())
        d.add_family("neuralprophet", n_trials=10)
        assert "issue 53" in _text(_findings(d, "error", "family neuralprophet"))

    def test_unknown_family(self):
        d = _minimal(_design())
        d.add_family("lstm", n_trials=10)
        assert "Unknown model family" in _text(_findings(d, "error", "family lstm"))

    def test_n_trials_one_refused(self):
        d = _minimal(_design())
        d.add_family("arima", n_trials=1)
        assert "1 is refused" in _text(_findings(d, "error", "family arima"))
        d.remove_family("arima")
        d.add_family("arima", n_trials=0)
        assert _findings(d, "error", "family arima") == []

    def test_bad_space_shaping(self):
        d = _minimal(_design())
        d.add_family("arima", n_trials=10, fixed={"max_z": 1})
        assert "no dimension" in _text(_findings(d, "error", "family arima"))

    def test_arima_two_cycles_on_daily(self):
        df = _frame(start="2023-01-01", periods=1300, freq="D")
        d = Design(df, "week", "deaths", None, "2026-06-01", 7, 1, "r", "x")
        d.set_seasonality(cycles=["weekly", "annual"])
        d.add_split("sp1", test_obs=60)
        d.add_cv("cv1", n_folds=3, fold_eval_obs=7)
        d.add_family("arima", n_trials=10)
        assert "one seasonal period" in _text(_findings(d, "error", "family arima"))

    def test_arima_annual_on_daily(self):
        df = _frame(start="2023-01-01", periods=1300, freq="D")
        d = Design(df, "week", "deaths", None, "2026-06-01", 7, 1, "r", "x")
        d.add_split("sp1", test_obs=60)
        d.add_cv("cv1", n_folds=3, fold_eval_obs=7)
        d.add_family("arima", n_trials=10)
        assert "cannot carry the annual" in _text(_findings(d, "error", "family arima"))

    def test_arima_capacity(self):
        # OQ-5: a smallest fold under 2m = 104 rows at m=52 is an error.
        d = _design()
        d.add_split("sp1", train_obs=129, test_obs=52)
        d.add_cv("cv1", n_folds=2, fold_eval_obs=11)      # smallest fold 107
        d.add_cv("cv2", n_folds=4, fold_eval_obs=13)      # smallest fold 77
        d.add_family("arima", n_trials=10)
        msg = _text(_findings(d, "error", "family arima"))
        assert "sp1/cv2 (77 rows)" in msg and "sp1/cv1" not in msg

    def test_budgets_and_flags_rendered(self):
        d = _minimal(_design())
        d.add_family("arima", n_trials=40, bounds={"max_p": (1, 3)})
        text = render_text(resolve(d)[0])
        assert "UNEQUAL BUDGETS" in text
        assert "NON-DEFAULT  max_p bounds" in text


class TestCriteria:

    def test_defaults(self):
        crit, _ = resolve(_minimal(_design()))[0]["criteria"], None
        selected = [c for c, e in crit.items() if e["selected"]]
        assert selected == ["S1", "S3", "S4", "C1", "C2", "C3"]
        assert all(e["threshold_defaulted"] for e in crit.values())
        assert crit["S5"]["threshold"] == [0.5, 2.0]

    def test_value_true_none(self):
        d = _minimal(_design())
        d.set_criteria(S2=True, S3=1.8, S4=None)
        crit = resolve(d)[0]["criteria"]
        assert crit["S2"]["selected"] and crit["S2"]["threshold_defaulted"]
        assert crit["S3"]["threshold"] == 1.8 and not crit["S3"]["threshold_defaulted"]
        assert not crit["S4"]["selected"]
        # Deselecting S4 withholds its verdict, never its number.
        combo = resolve(d)[0]["combinations"]["sp1/cv1"]["S4"]
        assert combo["verdict"] == "not selected" and combo["value"] > 0

    def test_calls_accumulate(self):
        d = _minimal(_design())
        d.set_criteria(S2=True)
        d.set_criteria(S3=1.8)
        crit = resolve(d)[0]["criteria"]
        assert crit["S2"]["selected"] and crit["S3"]["threshold"] == 1.8

    def test_s5_pair(self):
        d = _minimal(_design())
        d.set_criteria(S5=(0.75, 1.5))
        assert resolve(d)[0]["criteria"]["S5"]["threshold"] == [0.75, 1.5]
        for bad in ((2.0, 0.5), (0, 1.0), 1.0):
            d.set_criteria(S5=bad)
            assert "two-sided" in _text(_findings(d, "error", "criteria"))

    def test_threshold_types(self):
        d = _minimal(_design())
        for bad in ("1.0", -1, False, float("inf")):
            d.set_criteria(S1=bad)
            assert "positive number" in _text(_findings(d, "error", "criteria"))

    def test_menu_matches_contract(self):
        ops = {c: (e.op, e.default, e.basis) for c, e in CRITERIA.items()}
        assert ops == {
            "S1": ("<", 1.0, "principled"), "S2": ("<=", 1.5, "convention"),
            "S3": ("<=", 2.0, "convention"), "S4": (">=", 2.0, "principled"),
            "S5": ("band", (0.5, 2.0), "convention"),
            "C1": ("<", 1.0, "principled"), "C2": ("<=", 1.5, "convention"),
            "C3": ("<=", 1.5, "convention"),
        }


class TestCheck:

    def test_sn6_origins_rendered(self):
        resolved, _ = resolve(_contract_example(_design()))
        o = resolved["check"]["splits"]["sp1"]
        assert o["phase_origin"] == "2025-06-23"
        assert round(o["max_gap"], 2) == 0.33
        assert "event-phase origin 2025-06-23" in render_text(resolved)

    def test_capacity(self):
        d = _design(H=11)
        d.add_split("sp1", train_obs=129, test_obs=52)
        d.add_cv("cv1", n_folds=2, fold_eval_obs=11)
        d.add_family("prophet_xgb", n_trials=10)
        d.set_check(n_origins=5)
        assert "at least 55 rows" in _text(_findings(d, "error", "check origins on sp1"))


class TestSubPeriods:

    def test_full_window_always_present(self):
        subs = resolve(_minimal(_design()))[0]["sub_periods"]
        assert subs["full_window"] == {"start": "2026-06-22", "end": "2026-07-13",
                                       "last": "2026-07-06", "n": 3, "cycles": None}

    def test_reserved_name(self):
        d = _minimal(_design())
        d.add_sub_period("full_window", EVENT, EVENT)
        assert "reserved" in _text(_findings(d, "error", "sub-period full_window"))

    def test_outside_and_reversed(self):
        d = _minimal(_design())
        d.add_sub_period("late", "2026-06-29", "2026-07-13")
        d.add_sub_period("back", "2026-07-06", "2026-06-22")
        d.add_sub_period("off", "2026-06-23", "2026-06-29")
        assert "not inside" in _text(_findings(d, "error", "sub-period late"))
        assert "after it ends" in _text(_findings(d, "error", "sub-period back"))
        assert "2026-06-22 and 2026-06-29" in _text(_findings(d, "error", "sub-period off"))

    def test_inclusive_end(self):
        d = _minimal(_design())
        d.add_sub_period("acute", "2026-06-22", "2026-06-29")
        assert resolve(d)[0]["sub_periods"]["acute"]["n"] == 2


class TestSweepAndSeeds:

    def test_sweep_size(self):
        d = _minimal(_design())
        d.add_family("arima", n_trials=0)
        resolved, _ = resolve(d)
        # prophet_xgb: 10 x 2 folds + 1; arima: max(0, 1) x 2 folds + 1.
        assert resolved["report_fields"]["sweep_fits"] == 21 + 3

    def test_seeds_are_derived_and_independent(self):
        from its2s.seeds import derive_seed
        d = _minimal(_design())
        before = resolve(d)[0]["seeds"]
        assert before["prophet_xgb/sp1/cv1"] == derive_seed(42, "prophet_xgb/sp1/cv1")
        assert before["bootstrap"] == derive_seed(42, "bootstrap")
        d.add_family("arima", n_trials=10)
        after = resolve(d)[0]["seeds"]
        assert {k: after[k] for k in before} == before


def test_preview_imports_no_heavy_library():
    code = (
        "import sys, numpy as np, pandas as pd\n"
        "from its2s.design import Design\n"
        "df = pd.DataFrame({'week': pd.date_range('2023-01-02', periods=201, "
        "freq='W-MON'), 'deaths': np.ones(201)})\n"
        "d = Design(df, 'week', 'deaths', None, '2026-06-22', 3, 42, 'r', 'x')\n"
        "d.add_split('sp1', train_obs=129, test_obs=52)\n"
        "d.add_cv('cv1', n_folds=2, fold_eval_obs=3)\n"
        "d.add_family('arima', n_trials=10)\n"
        "d.add_family('prophet_xgb', n_trials=10)\n"
        "d.preview()\n"
        "print([m for m in ('prophet', 'pmdarima', 'neuralprophet', 'xgboost', "
        "'torch') if m in sys.modules])\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip().splitlines()[-1] == "[]"
