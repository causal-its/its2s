# Description: Step 3's computation (its2s.final_fit.fit) and its public call
#   (its2s.fit_final_model): a check that did not pass is refused before any
#   fit, the fit reads the check handoff alone and spans exactly the
#   pre-event rows, the model saved beside the sealed record loads back to
#   the same forecasts and is refused when a file changes, and a second call
#   is refused or, with rerun=True, must reproduce the sealed fit. Synthetic
#   data; stand-in families except where ARIMA's own order selection is
#   the point.
# Usage: pytest tests/test_fit_final_model.py
# Dependencies: pytest, numpy, pandas

import contextlib
import io
import json
import os
import stat

import numpy as np
import pandas as pd
import pytest

import its2s.check as check_module
import its2s.final_fit as final_fit
import its2s.fit_record as fit_record
import its2s.screening as screening
from its2s import (CheckRecord, FitError, FitRecord, SealError, fit_final_model)
from its2s.final_fit import fit
from its2s.fit_record import MODEL_DIR, RECORD_DIR, RECORD_FILE

from test_check_selected_model import SP1, SP2, _check, _declare
from test_design import EVENT, _frame
from test_screen_models import _quiet
from test_screening import _fake_family

LAST_PRE_EVENT = "2026-06-15"


def _saving_family(**kw):
    """A stand-in family that also saves and loads, as every real one does."""
    base = _fake_family(**kw)

    class Saving(base):
        _libraries = ()

        def save(self, directory):
            os.makedirs(directory, exist_ok=True)
            with open(os.path.join(directory, "level.json"), "w") as fh:
                json.dump({"level": self.level}, fh)

        @classmethod
        def load(cls, directory):
            with open(os.path.join(directory, "level.json")) as fh:
                model = cls(params=None, seasonality=None)
                model.level = json.load(fh)["level"]
            return model
    return Saving


@pytest.fixture
def families(monkeypatch):
    """Stand-in families for screening and the check, and an installer for
    the final fit (and loading its saved model)."""
    fake = _saving_family()
    for module in (screening, check_module, final_fit, fit_record):
        monkeypatch.setattr(module, "load_family_class", lambda name: fake)

    def install(cls):
        for module in (final_fit, fit_record):
            monkeypatch.setattr(module, "load_family_class", lambda name: cls)
    return install


@pytest.fixture
def no_fits(monkeypatch):
    """Any family the final fit loads raises, so a test can prove none is reached."""
    def refuse(name):
        raise AssertionError(f"a {name} fit was reached")
    monkeypatch.setattr(final_fit, "load_family_class", refuse)


def _screened(tmp, **kw):
    return _quiet(_declare(tmp, **kw), data=_frame())


@pytest.fixture
def checked(tmp_path, families):
    return _check(_screened(tmp_path), SP1, data=_frame())


@pytest.fixture
def failed(tmp_path, families):
    return _check(_screened(tmp_path), SP1, data=_frame(), C1=1e-9)


def _fit(*args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fit_final_model(*args, **kw)


def _target(check):
    return os.path.join(check.handoff["output_path"], RECORD_DIR)


def _files(directory):
    """{relative path: (bytes, mtime)} for every file under directory."""
    out = {}
    for root, _dirs, names in os.walk(directory):
        for n in names:
            p = os.path.join(root, n)
            with open(p, "rb") as fh:
                out[os.path.relpath(p, directory)] = (fh.read(), os.stat(p).st_mtime_ns)
    return out


def _writable(path):
    os.chmod(path, os.stat(path).st_mode | stat.S_IWUSR)


def _keys(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from _keys(v)
    elif isinstance(value, list):
        for v in value:
            yield from _keys(v)


# ---------------------------------------------------------------------------
# Refused before any fit
# ---------------------------------------------------------------------------

class TestRefusals:

    def test_a_failed_check_is_refused_before_any_fit(self, failed, no_fits):
        assert failed.verdict == "FAIL"
        with pytest.raises(FitError, match=r"The check of euromomo_heat2026 FAILED on "
                           r"\d{4}-\d\d-\d\d; no model is fit and no effect is estimated"):
            _fit(failed, data=_frame())
        assert not os.path.exists(_target(failed))

    def test_a_fail_edited_to_pass_on_disk_breaks_its_seal(self, failed):
        path = failed.path
        _writable(path)
        with open(path) as fh:
            text = fh.read()
        with open(path, "w") as fh:
            fh.write(text.replace('"verdict": "FAIL"', '"verdict": "PASS"'))
        assert '"verdict": "FAIL"' not in open(path).read()
        with pytest.raises(SealError, match="edited since it was checked"):
            CheckRecord.load(path)

    def test_a_dict_is_a_type_error_naming_the_loader(self, checked, no_fits):
        with pytest.raises(TypeError, match="CheckRecord.load"):
            _fit(checked.to_dict(), data=_frame())

    def test_other_data_is_refused_before_any_fit(self, checked, no_fits):
        df = _frame()
        df.loc[df.index[10], "deaths"] += 1
        with pytest.raises(FitError, match="does not match the design's fingerprint"):
            _fit(checked, data=df)
        assert not os.path.exists(_target(checked))


# ---------------------------------------------------------------------------
# What the fit reads, and the rows it fits on
# ---------------------------------------------------------------------------

def _facts(body):
    return {k: v for k, v in body.items() if k not in ("model", "fitted_at")}


class TestReads:

    def test_the_handoff_alone_gives_the_full_body(self, checked):
        body = fit(checked.to_dict()["handoff"], _frame())
        assert set(body) == {"model", "fitted_at", "span", "forecast_origin",
                             "selected_order", "seasonal_components", "fitted_values",
                             "warnings"}
        rec = _fit(checked, data=_frame()).to_dict()
        for key, value in _facts(body).items():
            assert rec[key] == value, key

    def test_no_post_event_outcome_is_read(self, checked):
        handoff = checked.to_dict()["handoff"]
        df = _frame()
        blind = df.copy()
        blind.loc[blind["week"] >= pd.Timestamp(EVENT), "deaths"] = np.nan
        assert _facts(fit(handoff, df)) == _facts(fit(handoff, blind))
        # Control: one pre-event outcome moved, and the fitted values move.
        moved = df.copy()
        moved.loc[moved["week"] == pd.Timestamp("2026-01-05"), "deaths"] += 50
        assert (fit(handoff, moved)["fitted_values"]
                != fit(handoff, df)["fitted_values"])

    @pytest.mark.parametrize("cid", [SP1, SP2])
    def test_the_fit_spans_every_row_before_the_event(self, tmp_path, families, cid):
        log = []
        check = _check(_screened(tmp_path), cid, data=_frame())
        families(_saving_family(log=log))
        rec = _fit(check, data=_frame())
        assert log == [("fit", pd.Timestamp("2023-01-02"), pd.Timestamp(LAST_PRE_EVENT),
                        181)]
        assert rec.span == {"first": "2023-01-02", "last": LAST_PRE_EVENT, "n": 181}
        assert rec.forecast_origin == LAST_PRE_EVENT


# ---------------------------------------------------------------------------
# The record and the saved model
# ---------------------------------------------------------------------------

class TestWritten:

    def test_record_and_model_are_written_read_only(self, checked):
        rec = _fit(checked, data=_frame())
        target = _target(checked)
        assert sorted(_files(target)) == [RECORD_FILE, os.path.join(MODEL_DIR, "level.json")]
        for root, _dirs, names in os.walk(target):
            for n in names:
                assert not os.stat(os.path.join(root, n)).st_mode & stat.S_IWUSR
        assert rec.path == os.path.join(target, RECORD_FILE)

    def test_the_record_on_disk_is_the_record_returned(self, checked):
        rec = _fit(checked, data=_frame())
        assert FitRecord.load(_target(checked)).to_dict() == rec.to_dict()
        path = rec.path
        _writable(path)
        text = open(path).read()
        with open(path, "w") as fh:
            fh.write(text.replace(LAST_PRE_EVENT, "2026-06-16", 1))
        with pytest.raises(SealError, match="edited since it was fitted"):
            FitRecord.load(path)

    def test_no_error_figure_of_any_kind(self, checked):
        rec = _fit(checked, data=_frame()).to_dict()
        keys = {k.lower() for k in _keys(rec)}
        assert not keys & {"rmse", "mae", "mase", "error", "errors", "residuals",
                           "predicted", "forecast"}

    def test_the_handoff_carries_what_step_4_reads(self, checked):
        rec = _fit(checked, data=_frame())
        h = rec.handoff
        assert set(h) == {"label", "data", "question", "seasonality", "candidate",
                          "fit_span", "forecast_origin", "model", "sub_periods",
                          "bootstrap_seed", "output_path", "stamps"}
        assert h["candidate"] == checked.handoff["candidate"]
        assert h["stamps"] == checked.handoff["stamps"] == rec.stamps
        assert h["model"] == rec.saved_model
        assert h["fit_span"] == rec.span
        assert rec.provenance["check"] == {"seal": checked.seal["digest"],
                                           "path": checked.path}

    def test_the_summary_says_where_the_model_is(self, checked, capsys):
        rec = fit_final_model(checked, data=_frame())
        out = capsys.readouterr().out
        assert "STEP 3 -- FINAL FIT, SEALED" in out
        assert f"2023-01-02 .. {LAST_PRE_EVENT}  (181 rows)" in out
        assert os.path.join(_target(checked), MODEL_DIR) in out
        assert rec.path in out


class TestSavedModel:

    def test_the_loaded_model_forecasts_as_the_fitted_one(self, checked):
        rec = _fit(checked, data=_frame())
        post = _frame().loc[lambda d: d["week"] >= pd.Timestamp(EVENT)]
        kw = {"target_col": "deaths", "date_col": "week"}
        held = rec.model().predict(post, **kw).predicted
        loaded = FitRecord.load(rec.path).model().predict(post, **kw).predicted
        np.testing.assert_array_equal(held, loaded)

    def test_the_returned_record_holds_the_model_it_fit(self, checked):
        rec = _fit(checked, data=_frame())
        assert rec.model() is rec.model()

    @pytest.mark.parametrize("change", ["edit", "add", "remove"])
    def test_a_changed_model_directory_is_refused(self, checked, change):
        rec = _fit(checked, data=_frame())
        model_dir = os.path.join(_target(checked), MODEL_DIR)
        os.chmod(model_dir, 0o755)
        saved = os.path.join(model_dir, "level.json")
        if change == "edit":
            _writable(saved)
            with open(saved, "a") as fh:
                fh.write(" ")
        elif change == "add":
            open(os.path.join(model_dir, "extra.txt"), "w").close()
        else:
            os.remove(saved)
        word = {"edit": "changed", "add": "added", "remove": "missing"}[change]
        with pytest.raises(SealError, match=f"does not match the fit record \\({word}"):
            FitRecord.load(rec.path).model()


# ---------------------------------------------------------------------------
# A second call on the same label
# ---------------------------------------------------------------------------

class TestSecondCall:

    def test_refused_without_rerun_and_nothing_is_fit(self, checked, monkeypatch):
        rec = _fit(checked, data=_frame())
        monkeypatch.setattr(final_fit, "load_family_class",
                            lambda name: (_ for _ in ()).throw(AssertionError("fit")))
        with pytest.raises(FitError, match=f"euromomo_heat2026 was fitted on "
                           f"{rec.fitted_at[:10]}; its record is at {rec.path}"):
            _fit(checked, data=_frame())

    def test_rerun_that_reproduces_returns_the_sealed_record(self, checked):
        rec = _fit(checked, data=_frame())
        before = _files(_target(checked))
        again = _fit(checked, data=_frame(), rerun=True)
        assert again.seal == rec.seal and again.to_dict() == rec.to_dict()
        assert _files(_target(checked)) == before

    def test_rerun_that_does_not_reproduce_writes_nothing(self, checked, families):
        _fit(checked, data=_frame())
        before = _files(_target(checked))
        families(_saving_family(shift=lambda params: 1.0))
        with pytest.raises(FitError, match="did not reproduce its sealed fit: "
                           "in-sample fitted values differ. Nothing was written."):
            _fit(checked, data=_frame(), rerun=True)
        assert _files(_target(checked)) == before


# ---------------------------------------------------------------------------
# ARIMA picks its order within the sealed bounds at every fit (real fits)
# ---------------------------------------------------------------------------

def _within(order, seasonal, params):
    p, d, q = order
    P, D, Q, _m = seasonal
    return (p <= params["max_p"] and d <= params["max_d"] and q <= params["max_q"]
            and P <= params["max_P"] and D <= params["max_D"] and Q <= params["max_Q"])


class TestArimaOrder:

    def test_every_origin_and_the_final_fit_record_an_order_within_bounds(self, tmp_path):
        cid = "arima/sp1/cv1"
        cands = _screened(tmp_path, families=(("arima", 0),))
        check = _check(cands, cid, data=_frame(), C1=100.0, C2=100.0, C3=100.0)
        rec = _fit(check, data=_frame())
        params = check.handoff["candidate"]["params"]
        orders = [o["structure"] for o in check.origins] + [rec.selected_order]
        assert len(orders) == len(check.origins) + 1
        for s in orders:
            assert set(s) == {"order", "seasonal_order", "with_intercept"}
            assert _within(s["order"], s["seasonal_order"], params)
        assert rec.selected_order["seasonal_order"][3] == check.handoff["candidate"]["m"]


# ---------------------------------------------------------------------------
# Public names
# ---------------------------------------------------------------------------

class TestPublicNames:

    def test_exported(self):
        import its2s
        for name in ("fit_final_model", "FitRecord", "FitError"):
            assert name in its2s.__all__ and hasattr(its2s, name)
        assert issubclass(FitError, ValueError)
