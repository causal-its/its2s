# Description: Step 1's public call (its2s.screen_models): the checks made
#   before any fit, the sealed candidates record and its handoff, the report
#   rendered from it, and what a second call on a screened label does.
#   Synthetic data; a stand-in family so no test waits on a real fit.
# Usage: pytest tests/test_screen_models.py
# Dependencies: pytest, numpy, pandas

import contextlib
import io
import json
import os
import re
import stat
import warnings

import pytest

import its2s
import its2s.screening as screening
from its2s import CandidatesRecord, DesignRecord, ScreeningError, SealError, screen_models
from its2s.candidates_record import RECORD_DIR, RECORD_FILE, REPORT_FILE
from its2s.design import DesignWarning, declare_design
from its2s.records import seal_record

from test_design import _design, _frame
from test_screening import _fake_family


def _declare(tmp, families=(("prophet_xgb", 3), ("arima", 2))):
    d = _design(tmp=str(tmp))
    d.set_seasonality(cycles=["annual"])
    d.add_split("sp1", train_obs=129, test_obs=52, primary=True)
    d.add_split("sp2", train_end="2025-10-06", test_end="2026-06-15")
    d.add_cv("cv1", n_folds=3, fold_eval_obs=6)
    for name, n in families:
        d.add_family(name, n_trials=n)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore", DesignWarning)
        return declare_design(d)


def _quiet(*args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return screen_models(*args, **kw)


@pytest.fixture
def families(monkeypatch):
    """Install stand-in families by name; returns a dict to replace them."""
    classes = {"prophet_xgb": _fake_family(), "arima": _fake_family()}
    monkeypatch.setattr(screening, "load_family_class", lambda name: classes[name])
    return classes


@pytest.fixture
def no_fits(monkeypatch):
    """Any family fit raises, so a test can prove that none is reached."""
    def refuse(name):
        raise AssertionError(f"a {name} fit was reached")
    monkeypatch.setattr(screening, "load_family_class", refuse)


@pytest.fixture
def screened(tmp_path, families):
    declared = _declare(tmp_path)
    return declared, _quiet(declared, data=_frame())


def _label_dir(declared):
    return declared.handoff["output_path"]


# ---------------------------------------------------------------------------
# Before any fit
# ---------------------------------------------------------------------------

class TestBeforeAnyFit:

    def test_takes_only_a_design_record(self, no_fits):
        with pytest.raises(TypeError, match="DesignRecord"):
            screen_models({"handoff": {}}, data=_frame())

    def test_a_changed_outcome_is_refused(self, tmp_path, families, no_fits):
        declared = _declare(tmp_path)
        df = _frame()
        df.loc[10, "deaths"] += 1
        with pytest.raises(ScreeningError, match="does not match the design's fingerprint"):
            screen_models(declared, data=df)
        assert not os.path.exists(os.path.join(_label_dir(declared), RECORD_DIR))

    def test_a_version_1_design_record_is_refused(self, tmp_path, no_fits):
        record = _declare(tmp_path).to_dict()
        record["schema_version"] = 1
        old = DesignRecord.from_dict(seal_record(record))
        with pytest.raises(ScreeningError, match="schema version 1"):
            screen_models(old, data=_frame())


# ---------------------------------------------------------------------------
# What is written
# ---------------------------------------------------------------------------

class TestWritten:

    def test_two_read_only_files(self, screened):
        declared, _ = screened
        sub = os.path.join(_label_dir(declared), RECORD_DIR)
        assert sorted(os.listdir(sub)) == sorted([RECORD_FILE, REPORT_FILE])
        for name in os.listdir(sub):
            mode = os.stat(os.path.join(sub, name)).st_mode
            assert not mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        assert not [n for n in os.listdir(_label_dir(declared)) if n.startswith(".")]

    def test_prints_the_report(self, tmp_path, families, capsys):
        declared = _declare(tmp_path)
        capsys.readouterr()
        screen_models(declared, data=_frame())
        written = open(os.path.join(_label_dir(declared), RECORD_DIR, REPORT_FILE)).read()
        assert capsys.readouterr().out == written

    def test_the_record_round_trips_to_the_written_report(self, screened):
        declared, cands = screened
        path = os.path.join(_label_dir(declared), RECORD_DIR, RECORD_FILE)
        loaded = CandidatesRecord.load(_label_dir(declared))
        assert loaded.path == path
        assert loaded.to_dict() == cands.to_dict()
        written = open(os.path.join(_label_dir(declared), RECORD_DIR, REPORT_FILE)).read()
        assert loaded.report() == written

    def test_an_edit_breaks_the_seal(self, screened, tmp_path):
        declared, _ = screened
        text = open(os.path.join(_label_dir(declared), RECORD_DIR, RECORD_FILE)).read()
        edited = text.replace('"n_trials": 3', '"n_trials": 4', 1)
        assert edited != text
        with pytest.raises(SealError, match="candidates record does not match its seal"):
            CandidatesRecord.from_dict(json.loads(edited))

    def test_the_design_is_embedded_verbatim(self, screened):
        declared, cands = screened
        assert cands.to_dict()["design"] == declared.to_dict()

    def test_timestamps_are_ordered_utc_microseconds(self, screened):
        declared, cands = screened
        stamp = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}\+00:00"
        for t in (cands.screening_started_at, cands.first_fit_at, cands.sealed_at):
            assert re.fullmatch(stamp, t)
        assert (declared.declared_at < cands.screening_started_at
                <= cands.first_fit_at < cands.sealed_at)


# ---------------------------------------------------------------------------
# The handoff step 2 reads
# ---------------------------------------------------------------------------

class TestHandoff:

    def test_sections(self, screened):
        _, cands = screened
        assert list(cands.handoff) == [
            "data", "question", "seasonality", "splits", "origins", "criteria",
            "sub_periods", "bootstrap_seed", "output_path", "candidates", "multiplicity"]
        assert set(cands.handoff["criteria"]) == {"C1", "C2", "C3"}

    def test_design_facts_are_carried_forward(self, screened):
        declared, cands = screened
        for key in ("data", "question", "seasonality", "splits", "origins",
                    "sub_periods", "bootstrap_seed", "output_path"):
            assert cands.to_dict()["handoff"][key] == declared.to_dict()["handoff"][key]

    def test_a_candidate_carries_what_step_2_needs(self, screened):
        _, cands = screened
        rec = cands.to_dict()
        h = rec["handoff"]["candidates"]["prophet_xgb/sp1/cv1"]
        full = rec["candidates"]["prophet_xgb/sp1/cv1"]
        assert h["params"] == full["best"]["params"]
        assert h["cv_rmse"] == full["best"]["mean_rmse"]
        assert h["seed"] == full["seed"] and h["primary"] is True
        assert h["passed_all"] == full["passed"]["passed_all"]

    def test_a_failed_candidate_is_ineligible_and_reported(self, tmp_path, families):
        def fail(kind, df, params):
            return "random_state" in params.get("xgb", {})
        families["prophet_xgb"] = _fake_family(fail=fail)
        declared = _declare(tmp_path)
        cands = _quiet(declared, data=_frame())
        h = cands.handoff["candidates"]["prophet_xgb/sp1/cv1"]
        assert h["status"] == "failed" and h["params"] is None and h["passed_all"] is False
        report = cands.report()
        assert "FAILED CANDIDATES  (2;" in report
        assert "prophet_xgb/sp1/cv1  at search:" in report


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

class TestReport:

    def test_every_family_row_is_followed_by_its_folds(self, screened):
        _, cands = screened
        lines = cands.report().splitlines()
        # Detail-table rows carry criterion verdicts; spread rows do not.
        rows = [i for i, l in enumerate(lines)
                if re.match(r"  (prophet_xgb|arima) +\d", l)
                and {"PASS", "FAIL", "--", "n/a"} & set(l.split())]
        # One row per candidate in the detail tables.
        assert len(rows) == 4
        for i in rows:
            assert [l.split()[:2] for l in lines[i + 1:i + 4]] == [
                ["fold", "1"], ["fold", "2"], ["fold", "3"]]

    def test_no_word_makes_a_choice(self, screened):
        _, cands = screened
        assert not re.search(r"\b(winner|best|recommended)\b", cands.report(), re.I)

    def test_the_multiplicity_is_stated(self, screened):
        _, cands = screened
        m = cands.multiplicity
        assert f"{m['n_candidates']} candidates over {m['n_fits']} fits" in cands.report()


# ---------------------------------------------------------------------------
# A second call on a screened label
# ---------------------------------------------------------------------------

class TestRerun:

    def test_without_rerun_nothing_is_fit(self, screened, monkeypatch):
        declared, cands = screened
        monkeypatch.setattr(screening, "load_family_class",
                            lambda name: (_ for _ in ()).throw(AssertionError("fit")))
        with pytest.raises(ScreeningError) as e:
            screen_models(declared, data=_frame())
        msg = str(e.value)
        assert f"was screened on {cands.sealed_at[:10]}" in msg
        assert f"4 candidates, {cands.multiplicity['n_fits']} fits" in msg
        assert re.search(r"\d+ minutes?\.", msg)
        assert "rerun=True" in msg and "Nothing has been fit." in msg

    def test_a_reproducing_rerun_keeps_the_record(self, screened):
        declared, cands = screened
        sub = os.path.join(_label_dir(declared), RECORD_DIR)
        before = {n: (open(os.path.join(sub, n), "rb").read(),
                      os.stat(os.path.join(sub, n)).st_mtime_ns) for n in os.listdir(sub)}
        again = _quiet(declared, data=_frame(), rerun=True)
        after = {n: (open(os.path.join(sub, n), "rb").read(),
                     os.stat(os.path.join(sub, n)).st_mtime_ns) for n in os.listdir(sub)}
        assert after == before
        assert again.seal == cands.seal

    def test_a_rerun_that_differs_writes_nothing(self, screened, families):
        declared, _ = screened
        sub = os.path.join(_label_dir(declared), RECORD_DIR)
        before = {n: open(os.path.join(sub, n), "rb").read() for n in os.listdir(sub)}
        families["arima"] = _fake_family(shift=lambda p: 0.5)
        with pytest.raises(ScreeningError, match=r"arima/sp1/cv1, arima/sp2/cv1 differ"):
            _quiet(declared, data=_frame(), rerun=True)
        assert {n: open(os.path.join(sub, n), "rb").read() for n in os.listdir(sub)} == before


def test_public_names():
    for name in ("screen_models", "CandidatesRecord", "ScreeningError"):
        assert name in its2s.__all__ and hasattr(its2s, name)
    assert issubclass(ScreeningError, ValueError)
