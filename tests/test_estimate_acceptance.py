# Description: Acceptance tests for step 4 (estimate_excess) on the real
#   families. The point forecast is the forecast of a fresh fit on exactly the
#   rows before the event, and not of a fit one row shorter; the computation
#   reads only the fit handoff (a static scan of the source); an estimate made
#   in the fitting session equals one made from the records on disk in a
#   fresh process, provenance included; the whole chain, run twice in fresh
#   processes under two labels, gives the same records, each estimate's
#   provenance digests equal to the seals of the records on disk; failed
#   replications are removed and counted end to end; every estimate is
#   labelled not validated; and a loaded model that does not reproduce the
#   sealed fitted values is refused before any bootstrap. Real arima and
#   prophet_xgb fits on synthetic weekly data.
# Usage: pytest tests/test_estimate_acceptance.py
# Dependencies: pytest, numpy, pandas, pmdarima, prophet, xgboost

import ast
import json
import os
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

import its2s.estimate as estimate_module
import its2s.final_fit as final_fit
import its2s.fitting as fitting
from its2s import (CandidatesRecord, CheckRecord, DesignRecord, EstimateError,
                   EstimateRecord, EstimateWarning, FitRecord, estimate_excess)
from its2s.estimate_record import RECORD_DIR, RECORD_FILE
from its2s.fitting import seasonality_of
from its2s.models.families import load_family_class
from its2s.records import canonical_json

from conftest import assert_labelled_not_validated
from test_check_refit_acceptance import (AR, LOOSE, PX, _CLOCK, _RECORD_NAMES, _fit,
                                         _functions, _handoff_source, _reachable,
                                         _violations, _without_clock, _without_file_bytes)
from test_check_selected_model import _check, _declare
from test_design import EVENT, _frame
import test_estimate_excess as step4_tests
from test_estimate_excess import (_est, _failing_at, _strip,
                                  assert_failures_counted_and_survivors_used,
                                  no_bootstrap)  # noqa: F401 (a fixture)
from test_screen_models import _quiet

# Replications per real estimate: enough to form an interval, few enough to
# keep a real refit per replication affordable.
N_REAL = 20
FAMILIES = (("prophet_xgb", PX), ("arima", AR))
COLS = {"target_col": "deaths", "date_col": "week"}


def _label_dir(fit):
    return os.path.dirname(os.path.dirname(os.path.abspath(fit.path)))


def _run_file(fit, run_label):
    return os.path.join(_label_dir(fit), RECORD_DIR, run_label, RECORD_FILE)


@pytest.fixture(scope="module", params=FAMILIES, ids=[f for f, _ in FAMILIES])
def real(request, tmp_path_factory):
    """A real family's step 3 fit, as fit_final_model returned it (its model
    in memory), on the contract's example series."""
    family, cid = request.param
    cands = _quiet(_declare(tmp_path_factory.mktemp(family), families=((family, 2),)),
                   data=_frame())
    return _fit(_check(cands, cid, data=_frame(), **LOOSE), data=_frame())


# ---------------------------------------------------------------------------
# The point forecast is a fresh fit on exactly the rows before the event
# ---------------------------------------------------------------------------

def _fresh_forecast(handoff, rows, target):
    """Fit a new model of the sealed family, parameters and seasonality on
    rows, and forecast target."""
    c = handoff["candidate"]
    model = load_family_class(c["family"])(params=c["params"],
                                           seasonality=seasonality_of(handoff))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(rows, **COLS)
        return np.asarray(model.predict(target, **COLS).predicted, dtype=float)


def test_the_point_forecast_is_a_fresh_fit_on_the_rows_before_the_event(real):
    est = _est(real, run_label="a1", n_replications=N_REAL)
    h = real.to_dict()["handoff"]
    data = _frame()
    event = pd.Timestamp(EVENT)
    # Selected by date here, independently of the package's own row selection.
    before = data.loc[data["week"] < event]
    target = data.loc[data["week"] >= event].iloc[:h["candidate"]["post_event"]["n"]]
    point = np.array([row["forecast"] for row in est.per_step])

    np.testing.assert_array_equal(point, _fresh_forecast(h, before, target))
    # The last pre-event row left out: a one-row slip at the forecast origin.
    shorter = _fresh_forecast(h, before.iloc[:-1], target)
    assert not np.allclose(point, shorter)

    origin = before["week"].iloc[-1].strftime("%Y-%m-%d")
    assert real.to_dict()["forecast_origin"] == h["forecast_origin"] == origin
    assert pd.Timestamp(origin) + pd.Timedelta(weeks=1) == event
    assert est.per_step[0]["date"] == EVENT
    assert [r["date"] for r in est.per_step] == [
        d.strftime("%Y-%m-%d") for d in target["week"]]


# ---------------------------------------------------------------------------
# The computation reads only the fit handoff (static scan)
# ---------------------------------------------------------------------------

# The step 3 name set, plus the name the public step binds the fit record to.
ESTIMATE_RECORD_NAMES = _RECORD_NAMES | {"fit"}


def _estimate_defs():
    """Every top-level function of estimate.py and of the modules its
    computation calls into for rows and forecasts; no name is defined twice
    among those the computation reaches."""
    return _functions(estimate_module, fitting, final_fit)


class TestReadsOnlyTheFitHandoff:

    def test_the_computation_touches_no_record(self):
        defs = _estimate_defs()
        names = _reachable("estimate", defs)
        assert {"estimate", "fit_rows", "target_rows", "pre_event_rows",
                "predict_family"} <= names
        for name in names:
            owners = [m.__name__ for m in (estimate_module, fitting, final_fit)
                      if name in _functions(m)]
            assert len(owners) == 1, (name, owners)
        assert _violations(names, defs, "estimate", ESTIMATE_RECORD_NAMES) == []

    def test_the_public_step_passes_the_handoff_alone(self):
        defs = _estimate_defs()
        calls = [n for n in ast.walk(defs["estimate_excess"]) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "estimate"]
        assert len(calls) == 1
        assert ast.unparse(calls[0].args[0]) == "handoff"
        assert _handoff_source(defs["estimate_excess"], defs) == "fit.to_dict()['handoff']"

    def test_the_scan_sees_a_fit_record_read(self):
        """Control: a computation that names the fit record is caught, by the
        name step 3's set lacked."""
        defs = _estimate_defs()
        bad = ast.parse("def estimate(handoff, data):\n    return fit.span\n")
        defs["estimate"] = bad.body[0]
        assert _violations({"estimate"}, defs, "estimate", _RECORD_NAMES) == []
        assert _violations({"estimate"}, defs, "estimate", ESTIMATE_RECORD_NAMES) != []


# ---------------------------------------------------------------------------
# The same fit record, in the fitting session and from disk
# ---------------------------------------------------------------------------

_FROM_DISK = """
import contextlib, io, sys, warnings
sys.path.insert(0, sys.argv[1])
warnings.simplefilter("ignore")
from test_design import _frame
from its2s import FitRecord, estimate_excess
label_dir, run_label, n = sys.argv[2], sys.argv[3], int(sys.argv[4])
with contextlib.redirect_stdout(io.StringIO()):
    estimate_excess(FitRecord.load(label_dir), data=_frame(), n_replications=n,
                    run_label=run_label)
"""

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _python(script, *args):
    subprocess.run([sys.executable, "-c", script, TESTS_DIR, *map(str, args)],
                   check=True, capture_output=True, text=True)


def test_an_estimate_from_disk_equals_one_in_the_fitting_session(real):
    session = _est(real, run_label="session", n_replications=N_REAL)
    _python(_FROM_DISK, _label_dir(real), "disk", N_REAL)
    disk = EstimateRecord.load(os.path.dirname(_run_file(real, "disk")))
    assert canonical_json(_strip(session)) == canonical_json(_strip(disk))
    # Control: the provenance is compared, and names the four records.
    assert set(disk.to_dict()["provenance"]) == {"design", "candidates", "check", "fit"}
    assert dict(disk.provenance["fit"])["seal"] == real.seal["digest"]


# ---------------------------------------------------------------------------
# The whole chain twice, each in a fresh process, under two labels
# ---------------------------------------------------------------------------

_CHAIN = """
import contextlib, io, sys, warnings
sys.path.insert(0, sys.argv[1])
warnings.simplefilter("ignore")
from test_check_selected_model import _declare
from test_design import _frame
from its2s import (screen_models, check_selected_model, fit_final_model,
                   estimate_excess)
out, label, n = sys.argv[2], sys.argv[3], int(sys.argv[4])
data = _frame()
loose = {"C1": 100.0, "C2": 100.0, "C3": 100.0}
with contextlib.redirect_stdout(io.StringIO()):
    for family, cid in (("prophet_xgb", "prophet_xgb/sp1/cv1"), ("arima", "arima/sp1/cv1")):
        declared = _declare(f"{out}/{family}", label=label, families=((family, 2),))
        cands = screen_models(declared, data=data)
        check = check_selected_model(cands, cid, data=data, **loose)
        fit = fit_final_model(check, data=data)
        estimate_excess(fit, data=data, n_replications=n)
"""

# Each record's file under the label directory.
CHAIN_RECORDS = {name: os.path.join(cls.RECORD_DIR, cls.RECORD_FILE) for name, cls in (
    ("design", DesignRecord), ("candidates", CandidatesRecord), ("check", CheckRecord),
    ("fit", FitRecord))}
CHAIN_RECORDS["estimate"] = os.path.join(RECORD_DIR, "main", RECORD_FILE)
ESTIMATE_CLOCK = _CLOCK | {"estimated_at"}


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _relabelled(value, label_dir, label):
    """value with this run's label directory and label replaced by
    placeholders, so two runs differ only where they should."""
    if isinstance(value, dict):
        return {k: _relabelled(v, label_dir, label) for k, v in value.items()}
    if isinstance(value, list):
        return [_relabelled(v, label_dir, label) for v in value]
    if isinstance(value, str):
        return value.replace(label_dir, "<label_dir>").replace(label, "<label>")
    return value


def _comparable(record, name, label_dir, label):
    out = _without_clock(_relabelled(record, label_dir, label), ESTIMATE_CLOCK)
    if name == "fit":
        out = _without_file_bytes(out)
    return out


def _assert_provenance_is_on_disk(estimate, label_dir):
    """Every digest in the estimate's provenance is the seal of the record at
    the path it names, under this run's own label directory."""
    provenance = estimate["provenance"]
    assert set(provenance) == {"design", "candidates", "check", "fit"}
    for name, entry in provenance.items():
        assert entry["path"] == os.path.join(label_dir, CHAIN_RECORDS[name]), name
        assert entry["seal"] == _read(entry["path"])["seal"]["digest"], name


def test_the_whole_chain_reproduces_in_fresh_processes(tmp_path):
    labels = ("heat_a", "heat_b")
    for label in labels:
        _python(_CHAIN, tmp_path / label, label, N_REAL)
    for family, _cid in FAMILIES:
        runs = []
        for label in labels:
            label_dir = str(tmp_path / label / family / label)
            records = {name: _read(os.path.join(label_dir, file))
                       for name, file in CHAIN_RECORDS.items()}
            _assert_provenance_is_on_disk(records["estimate"], label_dir)
            runs.append({name: _comparable(r, name, label_dir, label)
                         for name, r in records.items()})
        for name in CHAIN_RECORDS:
            assert canonical_json(runs[0][name]) == canonical_json(runs[1][name]), (
                family, name)
        # Control: the comparison sees the estimate and its relabelled paths.
        estimate = runs[0]["estimate"]
        assert estimate["label"] == "<label>" and estimate["bootstrap"]["n_succeeded"] > 0
        assert estimate["provenance"]["fit"]["path"].startswith("<label_dir>")


# ---------------------------------------------------------------------------
# Failed replications on a real family, end to end
# ---------------------------------------------------------------------------

def test_failed_replications_on_a_real_family(real, monkeypatch):
    captured = []
    _failing_at(monkeypatch, step4_tests.TestFailures.FAILING, captured)
    with pytest.warns(EstimateWarning, match=f"4 of {N_REAL} bootstrap replications failed"):
        est = estimate_excess(real, data=_frame(), n_replications=N_REAL, run_label="a6")
    point = np.array([row["forecast"] for row in est.per_step])
    assert_failures_counted_and_survivors_used(est, captured, real, point, n=N_REAL)


# ---------------------------------------------------------------------------
# Every estimate is labelled not validated (the suite-wide check lives in
# conftest.py; here, both interval types on each real family)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("interval_type", ["prediction", "confidence"])
def test_every_interval_type_is_labelled_not_validated(real, interval_type):
    est = _est(real, run_label=f"a7_{interval_type}", interval_type=interval_type,
               n_replications=N_REAL)
    assert est.interval["validated"] is False
    assert_labelled_not_validated(est.path)


# ---------------------------------------------------------------------------
# A loaded model off the sealed fitted values is refused
# ---------------------------------------------------------------------------

def test_a_loaded_model_off_by_one_ulp_is_refused(real, no_bootstrap, monkeypatch):
    loaded = FitRecord.load(_label_dir(real))
    handoff = loaded.to_dict()["handoff"]
    # Control: unchanged, the loaded model reproduces the sealed fitted values.
    estimate_module._confirmed_model(loaded, handoff, _frame())

    cls = type(loaded.model())
    original = cls.in_sample

    def moved(self, *args, **kw):
        values = np.array(original(self, *args, **kw), dtype=float)
        values[7] = np.nextafter(values[7], np.inf)
        return values
    monkeypatch.setattr(cls, "in_sample", moved)
    with pytest.raises(EstimateError, match="does not reproduce.*Fit with:.*Installed:"):
        _est(loaded, run_label="a8")
    assert not os.path.exists(os.path.dirname(_run_file(real, "a8")))
