# Description: Step 2's computation (its2s.check.check) and its public call
#   (its2s.check_selected_model): what is read, where each origin fits and
#   forecasts, the criteria arithmetic, overrides, single use, the refusals
#   made before anything is spent, earlier checks on the same data, a failed
#   origin, and the sealed check record and its report. Synthetic data;
#   stand-in families so no test waits on a real fit.
# Usage: pytest tests/test_check_selected_model.py
# Dependencies: pytest, numpy, pandas

import contextlib
import io
import json
import os
import stat
import warnings

import numpy as np
import pandas as pd
import pytest

import its2s.check as check_module
import its2s.screening as screening
from its2s import (CheckError, CheckRecord, SealError, check_selected_model)
from its2s.check import CheckWarning, check, resolve_thresholds
from its2s.check_record import RECORD_DIR, RECORD_FILE, REPORT_FILE
from its2s.design import Design, DesignWarning, declare_design

from test_design import EVENT, _frame
from test_screen_models import _quiet
from test_screening import _fake_family

SP1 = "prophet_xgb/sp1/cv1"
SP2 = "prophet_xgb/sp2/cv1"


def _declare(tmp, label="euromomo_heat2026", H=3, df=None,
             families=(("prophet_xgb", 3), ("arima", 2))):
    df = _frame() if df is None else df
    d = Design(df, "week", "deaths", None, EVENT, H, 42, str(tmp), label)
    d.set_seasonality(cycles=["annual"])
    d.add_split("sp1", train_obs=129, test_obs=52, primary=True)
    d.add_split("sp2", train_end="2025-10-06", test_end="2026-06-15")
    d.add_cv("cv1", n_folds=3, fold_eval_obs=6)
    for name, n in families:
        d.add_family(name, n_trials=n)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore", DesignWarning)
        return declare_design(d)


def _check(*args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return check_selected_model(*args, **kw)


@pytest.fixture
def families(monkeypatch):
    """Stand-in families: screening and the check each get their own, so a
    test can make the check's fail without touching screening."""
    fake = _fake_family()
    monkeypatch.setattr(screening, "load_family_class", lambda name: fake)
    monkeypatch.setattr(check_module, "load_family_class", lambda name: fake)

    def install(cls):
        monkeypatch.setattr(check_module, "load_family_class", lambda name: cls)
    return install


@pytest.fixture
def no_check_fits(monkeypatch):
    """Any family the check loads raises, so a test can prove none is reached."""
    def refuse(name):
        raise AssertionError(f"a {name} fit was reached")
    monkeypatch.setattr(check_module, "load_family_class", refuse)


@pytest.fixture
def screened(tmp_path, families):
    return _quiet(_declare(tmp_path), data=_frame())


def _label_dir(cands):
    return cands.handoff["output_path"]


def _spent(cands):
    return os.path.exists(os.path.join(_label_dir(cands), RECORD_DIR))


def _thresholds(cands, **overrides):
    return resolve_thresholds(cands.to_dict()["handoff"]["criteria"], overrides)


def _body_without_clock(body):
    return {k: v for k, v in body.items() if k != "first_fit_at"}


# ---------------------------------------------------------------------------
# What the computation reads
# ---------------------------------------------------------------------------

class TestReads:

    def test_the_handoff_alone_gives_the_full_body(self, screened):
        handoff = screened.to_dict()["handoff"]
        body = check(handoff, _frame(), SP1, _thresholds(screened))
        assert set(body) == {"first_fit_at", "n_fits", "origins", "aggregate",
                             "criteria", "verdict", "max_seasonal_gap"}
        record = _check(screened, SP1, data=_frame())
        assert body["origins"] == record.to_dict()["origins"]
        assert body["criteria"] == record.to_dict()["criteria"]

    def test_no_post_event_outcome_is_read(self, screened):
        handoff = screened.to_dict()["handoff"]
        df = _frame()
        blind = df.copy()
        blind.loc[blind["week"] >= pd.Timestamp(EVENT), "deaths"] = np.nan
        a = check(handoff, df, SP1, _thresholds(screened))
        b = check(handoff, blind, SP1, _thresholds(screened))
        assert _body_without_clock(a) == _body_without_clock(b)


# ---------------------------------------------------------------------------
# Where each origin fits and forecasts
# ---------------------------------------------------------------------------

class TestOrigins:

    @pytest.mark.parametrize("cid", [SP1, SP2])
    def test_fits_and_forecasts_are_the_sealed_origins(self, screened, families, cid):
        log = []
        families(_fake_family(log=log))
        h = screened.to_dict()["handoff"]
        split = h["splits"][h["candidates"][cid]["split"]]
        origins = h["origins"][h["candidates"][cid]["split"]]["dates"]
        H = h["question"]["H"]
        check(h, _frame(), cid, _thresholds(screened))
        fits = [e for e in log if e[0] == "fit"]
        preds = [e for e in log if e[0] == "predict"]
        assert len(fits) == len(preds) == len(origins)
        step = pd.Timedelta(weeks=1)
        for (_, f0, f1, _), (_, p0, p1, n), o in zip(fits, preds, origins):
            o = pd.Timestamp(o)
            assert f0 == pd.Timestamp(split["train"]["start"])
            assert f1 == o - step
            assert p0 == o and n == H and p1 == o + (H - 1) * step
            assert p1 < pd.Timestamp(split["test"]["end"])
        # The first fit holds no test data: it ends where the training window does.
        assert fits[0][2] == pd.Timestamp(split["train"]["end"]) - step

    def test_the_event_phase_origin_is_marked(self, screened):
        rec = _check(screened, SP1, data=_frame())
        phase = screened.handoff["origins"]["sp1"]["phase_origin"]
        marked = [o["date"] for o in rec.origins if o["event_phase"]]
        assert marked == ([phase] if phase is not None else [])

    def test_max_seasonal_gap_equals_the_design_records(self, tmp_path, families):
        declared = _declare(tmp_path)
        cands = _quiet(declared, data=_frame())
        for cid, sid in ((SP1, "sp1"), (SP2, "sp2")):
            body = check(cands.to_dict()["handoff"], _frame(), cid, _thresholds(cands))
            assert body["max_seasonal_gap"] == declared.check["splits"][sid]["max_gap"]


# ---------------------------------------------------------------------------
# Criteria arithmetic, computed independently of the package
# ---------------------------------------------------------------------------

def _by_hand(h, df, cid):
    """Per-origin RMSE and MASE for the stand-in, which forecasts the mean of
    its fit rows."""
    split = h["splits"][h["candidates"][cid]["split"]]
    H, m = h["question"]["H"], 52
    rmse, mase = [], []
    for o in h["origins"][h["candidates"][cid]["split"]]["dates"]:
        fit = df[(df.week >= split["train"]["start"]) & (df.week < o)]["deaths"].to_numpy()
        ahead = df[df.week >= o]["deaths"].to_numpy()[:H]
        err = ahead - fit.mean()
        rmse.append(np.sqrt(np.mean(err ** 2)))
        mase.append(np.mean(np.abs(err)) / np.mean(np.abs(fit[m:] - fit[:-m])))
    return rmse, mase


class TestCriteria:

    def test_c1_c2_c3_equal_hand_computed_values(self, screened):
        h = screened.to_dict()["handoff"]
        rmse, mase = _by_hand(h, _frame(), SP1)
        assert len(rmse) % 2 == 0      # C3's median is the mean of the middle two
        body = check(h, _frame(), SP1, _thresholds(screened))
        c = body["criteria"]
        assert c["C1"]["value"] == pytest.approx(np.mean(mase))
        assert c["C2"]["value"] == pytest.approx(
            np.mean(rmse) / h["candidates"][SP1]["cv_rmse"])
        s = sorted(rmse)
        mid = len(s) // 2
        assert c["C3"]["value"] == pytest.approx(max(rmse) / ((s[mid - 1] + s[mid]) / 2))

    def test_a_selected_criterion_not_computable_fails_the_check(self, screened,
                                                                 monkeypatch):
        # A seasonal-naive period longer than any fit span leaves MASE undefined.
        monkeypatch.setattr(check_module, "mase_period", lambda s: 1000)
        rec = _check(screened, SP1, data=_frame())
        assert rec.criteria["C1"]["value"] is None
        assert rec.criteria["C1"]["verdict"] == "n/a"
        assert "not more than m=1000" in rec.criteria["C1"]["reason"]
        assert rec.verdict == "FAIL"
        assert any(r.startswith("C1 is selected and not computable")
                   for r in rec.aggregate["verdict_reasons"])

    def test_a_deselected_criterion_not_computable_does_not_fail(self, screened,
                                                                monkeypatch):
        monkeypatch.setattr(check_module, "mase_period", lambda s: 1000)
        rec = _check(screened, SP1, data=_frame(), C1=None)
        assert rec.criteria["C1"]["selected"] is False
        assert not any(r.startswith("C1") for r in rec.aggregate["verdict_reasons"])


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------

class TestOverrides:

    def test_an_override_is_stamped_beside_step_0(self, screened):
        rec = _check(screened, SP1, data=_frame(), C2=2.0)
        c2 = rec.criteria["C2"]
        assert (c2["step0_threshold"], c2["threshold"], c2["overridden"]) == (1.5, 2.0, True)
        assert rec.handoff["stamps"]["overrides"]["C2"] == (1.5, 2.0)
        assert "OVERRIDDEN C2      step 0 <= 1.5  ->  applied <= 2.0" in rec.report()
        assert not rec.criteria["C1"]["overridden"]

    def test_none_deselects(self, screened):
        rec = _check(screened, SP1, data=_frame(), C3=None)
        assert rec.criteria["C3"]["selected"] is False
        assert rec.criteria["C3"]["verdict"] == "not-selected"
        assert rec.handoff["stamps"]["overrides"]["C3"] == (1.5, None)
        assert "applied deselected" in rec.report()

    @pytest.mark.parametrize("value, error", [
        ("2", TypeError), (True, TypeError), (0, CheckError), (-1.0, CheckError),
        (float("inf"), CheckError), (float("nan"), CheckError)])
    def test_a_bad_value_is_refused_before_any_fit(self, screened, no_check_fits,
                                                    value, error):
        with pytest.raises(error, match="C2"):
            _check(screened, SP1, data=_frame(), C2=value)
        assert not _spent(screened)

    def test_deselecting_all_three_is_refused_before_any_fit(self, screened,
                                                            no_check_fits):
        with pytest.raises(CheckError, match="No check criterion is selected"):
            _check(screened, SP1, data=_frame(), C1=None, C2=None, C3=None)
        assert not _spent(screened)


# ---------------------------------------------------------------------------
# Refusals made before anything is spent
# ---------------------------------------------------------------------------

class TestRefusals:

    def test_takes_only_a_candidates_record(self, screened, no_check_fits):
        with pytest.raises(TypeError, match="CandidatesRecord"):
            check_selected_model(screened.to_dict(), SP1, data=_frame())
        assert not _spent(screened)

    def test_a_changed_outcome_is_refused(self, screened, no_check_fits):
        df = _frame()
        df.loc[10, "deaths"] += 1
        with pytest.raises(CheckError, match="does not match the design's fingerprint"):
            check_selected_model(screened, SP1, data=df)
        assert not _spent(screened)

    def test_an_unknown_id_is_refused_naming_those_that_fit(self, screened,
                                                            no_check_fits):
        with pytest.raises(CheckError, match=f"No candidate 'nope'.*{SP1}"):
            check_selected_model(screened, "nope", data=_frame())
        assert not _spent(screened)

    def test_a_failed_candidate_is_refused(self, tmp_path, monkeypatch, no_check_fits):
        broken = _fake_family(fail=lambda kind, df, params: True)
        ok = _fake_family()
        monkeypatch.setattr(screening, "load_family_class",
                            lambda name: broken if name == "arima" else ok)
        cands = _quiet(_declare(tmp_path), data=_frame())
        with pytest.raises(CheckError, match="arima/sp1/cv1 failed at screening"):
            check_selected_model(cands, "arima/sp1/cv1", data=_frame())
        assert not _spent(cands)


# ---------------------------------------------------------------------------
# Single use
# ---------------------------------------------------------------------------

class TestSingleUse:

    @pytest.mark.parametrize("first_kw, verdict", [({}, "PASS"), ({"C1": 0.01}, "FAIL")])
    def test_a_second_call_fits_nothing_and_names_the_first(self, screened, monkeypatch,
                                                            first_kw, verdict):
        first = _check(screened, SP1, data=_frame(), **first_kw)
        assert first.verdict == verdict
        before = open(first.path).read()

        def refuse(name):
            raise AssertionError("a fit was reached")
        monkeypatch.setattr(check_module, "load_family_class", refuse)
        for cid in (SP1, SP2):
            with pytest.raises(CheckError, match=(
                    f"spent on {first.checked_at[:10]} by {SP1}, verdict {verdict}")):
                check_selected_model(screened, cid, data=_frame())
        assert open(first.path).read() == before


# ---------------------------------------------------------------------------
# Earlier checks on the same data
# ---------------------------------------------------------------------------

class TestPriorChecks:

    def test_a_second_label_on_the_same_data_is_warned_and_counted(self, tmp_path,
                                                                  families):
        a = _quiet(_declare(tmp_path, label="first"), data=_frame())
        _check(a, SP1, data=_frame())
        b = _quiet(_declare(tmp_path, label="second"), data=_frame())
        with pytest.warns(CheckWarning, match="first on"):
            rec = _check(b, SP1, data=_frame())
        assert rec.prior_checks["count"] == 1
        assert rec.prior_checks["records"][0]["label"] == "first"
        assert rec.handoff["stamps"]["prior_checks"] == 1
        assert "prior checks on this data   1" in rec.report()

    def test_a_label_differing_only_in_H_is_still_counted(self, tmp_path, families):
        a = _quiet(_declare(tmp_path, label="first", H=3), data=_frame())
        _check(a, SP1, data=_frame())
        b = _quiet(_declare(tmp_path, label="second", H=4), data=_frame())
        assert b.data_fingerprint["digest"] != a.data_fingerprint["digest"]
        with pytest.warns(CheckWarning):
            rec = _check(b, SP1, data=_frame())
        assert rec.prior_checks["count"] == 1

    def test_different_pre_event_values_are_not_counted(self, tmp_path, families):
        a = _quiet(_declare(tmp_path, label="first"), data=_frame())
        _check(a, SP1, data=_frame())
        df = _frame()
        df.loc[10, "deaths"] += 1
        b = _quiet(_declare(tmp_path, label="second", df=df), data=df)
        with warnings.catch_warnings():
            warnings.simplefilter("error", CheckWarning)
            rec = _check(b, SP1, data=df)
        assert rec.prior_checks["count"] == 0

    def test_a_record_with_a_broken_seal_is_still_counted(self, tmp_path, families):
        a = _quiet(_declare(tmp_path, label="first"), data=_frame())
        first = _check(a, SP1, data=_frame())
        os.chmod(first.path, 0o644)
        text = open(first.path).read()
        with open(first.path, "w") as fh:
            fh.write(text.replace('"verdict": "PASS"', '"verdict": "FAIL"', 1))
        b = _quiet(_declare(tmp_path, label="second"), data=_frame())
        with pytest.warns(CheckWarning, match="no longer matches its seal"):
            rec = _check(b, SP1, data=_frame())
        assert rec.prior_checks["records"][0]["seal_intact"] is False


# ---------------------------------------------------------------------------
# A failed origin
# ---------------------------------------------------------------------------

class TestOriginFailure:

    def test_one_failed_origin_fails_the_check_and_spends_the_window(self, screened,
                                                                     families):
        h = screened.handoff
        second = pd.Timestamp(h["origins"]["sp1"]["dates"][1])
        families(_fake_family(fail=lambda kind, df, params: (
            kind == "fit" and df["week"].iloc[-1] == second - pd.Timedelta(weeks=1))))
        rec = _check(screened, SP1, data=_frame())
        status = [o["status"] for o in rec.origins]
        assert status == ["ok", "failed", "ok", "ok"]
        assert "injected fit failure" in rec.origins[1]["error"]
        for cid in ("C1", "C2", "C3"):
            assert rec.criteria[cid]["value"] is None
            assert rec.criteria[cid]["reason"] == "origin 2 failed"
        assert rec.verdict == "FAIL"
        assert rec.test_window_spent is True and _spent(screened)


# ---------------------------------------------------------------------------
# The stamps
# ---------------------------------------------------------------------------

class TestStamps:

    @pytest.mark.parametrize("cid, after", [(SP1, False), (SP2, True)])
    def test_split_chosen_after_evidence(self, screened, cid, after):
        rec = _check(screened, cid, data=_frame())
        assert rec.split_chosen_after_evidence is after
        assert rec.handoff["stamps"]["split_chosen_after_evidence"] is after
        assert ("SPLIT CHOSEN AFTER EVIDENCE" in rec.report()) is after

    def test_rationale_is_stamped_or_said_absent(self, tmp_path, families):
        a = _quiet(_declare(tmp_path, label="a"), data=_frame())
        b = _quiet(_declare(tmp_path, label="b"), data=_frame())
        with_r = _check(a, SP1, data=_frame(), rationale="the only one on sp1")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", CheckWarning)
            without = _check(b, SP1, data=_frame())
        assert with_r.rationale == "the only one on sp1"
        assert "rationale          the only one on sp1" in with_r.report()
        assert without.rationale is None
        assert "rationale          none given" in without.report()


# ---------------------------------------------------------------------------
# What is written
# ---------------------------------------------------------------------------

class TestWritten:

    def test_a_fail_is_returned_and_written(self, screened):
        rec = _check(screened, SP1, data=_frame(), C1=0.01)
        assert isinstance(rec, CheckRecord) and rec.verdict == "FAIL"
        assert rec.handoff["verdict"] == "FAIL"
        assert os.path.exists(rec.path)
        assert rec.report().splitlines()[-2] == (
            "FAILED. No later step will run on euromomo_heat2026.")

    def test_two_read_only_files_and_the_printed_report(self, screened, capsys):
        rec = check_selected_model(screened, SP1, data=_frame())
        sub = os.path.join(_label_dir(screened), RECORD_DIR)
        assert sorted(os.listdir(sub)) == sorted([RECORD_FILE, REPORT_FILE])
        for name in os.listdir(sub):
            mode = os.stat(os.path.join(sub, name)).st_mode
            assert not mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        written = open(os.path.join(sub, REPORT_FILE)).read()
        assert capsys.readouterr().out == written
        assert rec.path == os.path.join(sub, RECORD_FILE)

    def test_the_record_round_trips_to_the_written_report(self, screened):
        rec = _check(screened, SP1, data=_frame())
        loaded = CheckRecord.load(_label_dir(screened))
        assert loaded.to_dict() == rec.to_dict()
        written = open(os.path.join(_label_dir(screened), RECORD_DIR, REPORT_FILE)).read()
        assert loaded.report() == written

    def test_a_one_character_edit_breaks_the_seal(self, screened):
        rec = _check(screened, SP1, data=_frame())
        text = open(rec.path).read()
        edited = text.replace('"verdict": "PASS"', '"verdict": "PASX"', 1)
        assert edited != text
        with pytest.raises(SealError, match="check record does not match its seal"):
            CheckRecord.from_dict(json.loads(edited))

    def test_the_record_carries_its_provenance(self, screened):
        rec = _check(screened, SP1, data=_frame())
        assert rec.candidate == screened.candidates[SP1]
        assert rec.provenance["candidates"]["seal"] == screened.seal["digest"]
        assert rec.provenance["design"]["seal"] == screened.design["seal"]["digest"]
        assert rec.multiplicity["n_check_fits"] == len(rec.origins)
        assert rec.screening_facts["passed_all"] == screened.handoff[
            "candidates"][SP1]["passed_all"]
