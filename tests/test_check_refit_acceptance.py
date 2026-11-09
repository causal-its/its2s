# Description: Acceptance tests for steps 2 and 3 (check_selected_model and
#   fit_final_model) on the real families: a failed check cannot reach a
#   fitted model; the check and final-fit computations read only their
#   predecessor's handoff (a static scan of the source); every fold fit,
#   origin fit and final fit carries the declared seasonal structure; and the
#   whole chain (declare, screen, check, fit), run twice in fresh processes,
#   gives the same check and fit records and saved models that behave
#   identically. Real arima and prophet_xgb fits on synthetic weekly data.
# Usage: pytest tests/test_check_refit_acceptance.py
# Dependencies: pytest, numpy, pandas, pmdarima, prophet, xgboost

import ast
import contextlib
import inspect
import io
import json
import os
import pickle
import shutil
import stat
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import its2s.check as check_module
import its2s.final_fit as final_fit
import its2s.fitting as fitting
import its2s.screening as screening
from its2s import CheckRecord, FitError, FitRecord, SealError, fit_final_model
from its2s.models.families import load_family_class
from its2s.records import canonical_json

from test_check_selected_model import _check, _declare
from test_design import EVENT, _frame
from test_screen_models import _quiet

PX = "prophet_xgb/sp1/cv1"
AR = "arima/sp1/cv1"
# Thresholds loose enough that a real candidate passes, so the fit is reached.
LOOSE = {"C1": 100.0, "C2": 100.0, "C3": 100.0}


def _fit(*args, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fit_final_model(*args, **kw)


# ---------------------------------------------------------------------------
# A1: a failed check cannot reach a fitted model (real families)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def failed_real(tmp_path_factory):
    """A real prophet_xgb check forced to FAIL by an unreachable C1."""
    cands = _quiet(_declare(tmp_path_factory.mktemp("a1"),
                            families=(("prophet_xgb", 0),)), data=_frame())
    return _check(cands, PX, data=_frame(), C1=1e-9)


class TestFailedCheckCannotBeFit:

    def test_a_failed_check_is_refused_before_any_fit(self, failed_real, monkeypatch):
        assert failed_real.verdict == "FAIL"
        def refuse(name):
            raise AssertionError(f"a {name} fit was reached")
        monkeypatch.setattr(final_fit, "load_family_class", refuse)
        with pytest.raises(FitError, match="FAILED on .*no model is fit"):
            _fit(failed_real, data=_frame())
        assert not os.path.exists(os.path.join(failed_real.handoff["output_path"],
                                               "03_fit"))

    def test_a_fail_edited_to_pass_does_not_load(self, failed_real, tmp_path):
        copy = tmp_path / "02-check.json"
        shutil.copy(failed_real.path, copy)
        os.chmod(copy, stat.S_IRUSR | stat.S_IWUSR)
        copy.write_text(copy.read_text().replace('"verdict": "FAIL"', '"verdict": "PASS"'))
        with pytest.raises(SealError, match="edited since it was checked"):
            CheckRecord.load(copy)


# ---------------------------------------------------------------------------
# A2: the computations read only their predecessor's handoff (static scan)
# ---------------------------------------------------------------------------

# Names under which a record object is bound anywhere in these modules, and
# the record classes themselves. None may appear inside a computation.
_RECORD_NAMES = {"candidates", "check", "existing", "rec", "record", "CandidatesRecord",
                 "CheckRecord", "FitRecord", "DesignRecord"}
# Methods that turn a record object into its content or read one from disk.
_RECORD_METHODS = {"to_dict", "load", "from_dict"}


def _functions(*modules):
    out = {}
    for module in modules:
        for node in ast.parse(inspect.getsource(module)).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                out[node.name] = node
    return out


def _reachable(root, defs):
    """root and every function of defs it calls, directly or not."""
    seen, todo = set(), [root]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        for node in ast.walk(defs[name]):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in defs):
                todo.append(node.func.id)
    return seen


def _violations(names, defs, root, record_names=_RECORD_NAMES):
    found = []
    for name in sorted(names):
        for node in ast.walk(defs[name]):
            if isinstance(node, ast.Name) and node.id in record_names and \
                    not (name == root and node.id == root):
                found.append(f"{name}: name {node.id}")
            if isinstance(node, ast.Attribute) and node.attr in _RECORD_METHODS:
                found.append(f"{name}: .{node.attr}")
    return found


def _handoff_source(fn, defs):
    """Where the handoff passed to the computation comes from, as source."""
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and [ast.unparse(t) for t in node.targets] == \
                ["handoff"]:
            value = node.value
            if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
                return _handoff_source(defs[value.func.id], defs)
            return ast.unparse(value)
    return None


@pytest.mark.parametrize("module, root, public, record", [
    (check_module, "check", "check_selected_model", "candidates"),
    (final_fit, "fit", "fit_final_model", "check"),
])
class TestReadsOnlyTheHandoff:

    def test_the_computation_touches_no_record(self, module, root, public, record):
        defs = _functions(module, fitting)
        names = _reachable(root, defs)
        assert root in names and len(names) > 3
        assert _violations(names, defs, root) == []

    def test_the_public_step_passes_the_handoff_alone(self, module, root, public, record):
        defs = _functions(module, fitting)
        calls = [n for n in ast.walk(defs[public]) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == root]
        assert len(calls) == 1
        assert ast.unparse(calls[0].args[0]) == "handoff"
        assert _handoff_source(defs[public], defs) == f"{record}.to_dict()['handoff']"

    def test_the_scan_sees_a_record_read(self, module, root, public, record):
        """Control: a helper that reads a record is caught."""
        defs = _functions(module, fitting)
        bad = ast.parse(f"def {root}(handoff, data):\n    return {record}.to_dict()\n")
        defs[root] = bad.body[0]
        assert _violations({root}, defs, root) != []


# ---------------------------------------------------------------------------
# A3: every fit carries the declared seasonal structure (real families)
# ---------------------------------------------------------------------------

DECLARED = {
    "prophet_xgb": {"prophet": ["yearly"],
                    "xgb_time_features": ["day_of_year", "month", "week_of_year"]},
    "arima": {"m": 52, "seasonal_search": True},
}


def _recording(log):
    """load_family_class, with every fit logging its seasonal structure."""
    def load(name):
        cls = load_family_class(name)

        class Recording(cls):
            def fit(self, *args, **kw):
                result = super().fit(*args, **kw)
                log.append(self.seasonal_components())
                return result
        return Recording
    return load


@pytest.mark.parametrize("family, cid", [("prophet_xgb", PX), ("arima", AR)])
def test_every_fit_carries_the_declared_seasonality(tmp_path, monkeypatch, family, cid):
    logs = {"screen": [], "check": [], "fit": []}
    for step, module in (("screen", screening), ("check", check_module),
                         ("fit", final_fit)):
        monkeypatch.setattr(module, "load_family_class", _recording(logs[step]))
    cands = _quiet(_declare(tmp_path, families=((family, 2),)), data=_frame())
    check = _check(cands, cid, data=_frame(), **LOOSE)
    fit = _fit(check, data=_frame())
    n_folds = len(cands.combinations["sp1/cv1"]["folds"])
    assert len(logs["screen"]) >= 3 * n_folds + 1     # 2 settings, default, whole window
    assert len(logs["check"]) == len(check.origins)
    assert len(logs["fit"]) == 1
    expected = DECLARED[family]
    for step, entries in logs.items():
        for components in entries:
            components = dict(components)
            components.pop("fitted_seasonal_order", None)
            assert components == expected, step
    recorded = fit.to_dict()["seasonal_components"]
    recorded.pop("fitted_seasonal_order", None)
    assert recorded == expected


# ---------------------------------------------------------------------------
# A4: the whole chain twice, each in a fresh process
# ---------------------------------------------------------------------------

_CHAIN = """
import contextlib, io, pickle, sys, warnings
sys.path.insert(0, sys.argv[1])
warnings.simplefilter("ignore")
from test_check_selected_model import _declare
from its2s import screen_models, check_selected_model, fit_final_model
out, data_file = sys.argv[2:4]
with open(data_file, "rb") as fh:
    data = pickle.load(fh)
loose = {"C1": 100.0, "C2": 100.0, "C3": 100.0}
with contextlib.redirect_stdout(io.StringIO()):
    for family, cid in (("prophet_xgb", "prophet_xgb/sp1/cv1"), ("arima", "arima/sp1/cv1")):
        declared = _declare(f"{out}/{family}", families=((family, 2),))
        cands = screen_models(declared, data=data)
        check = check_selected_model(cands, cid, data=data, **loose)
        fit_final_model(check, data=data)
"""

# Every key that holds a time, or a seal that hashes one.
_CLOCK = {"declared_at", "screening_started_at", "first_fit_at", "checked_at",
          "fitted_at", "sealed_at", "seal"}


def _without_clock(value, clock=_CLOCK):
    if isinstance(value, dict):
        return {k: _without_clock(v, clock) for k, v in value.items() if k not in clock}
    if isinstance(value, list):
        return [_without_clock(v, clock) for v in value]
    return value


def _without_file_bytes(record):
    """A fit record with the saved model's file hashes reduced to file names.

    A pickled ARIMA model is the same model in every coefficient and forecast
    from one process to the next, but not the same bytes: the pickle's
    internal object numbering depends on which objects happen to share memory.
    The hashes guard a saved model against later edits; reproducibility is
    judged on what the saved models do (_same_behaviour).
    """
    for holder in (record.get("saved_model"), record.get("handoff", {}).get("model")):
        if holder is not None:
            holder["files"] = sorted(holder["files"])
    return record


def _same_behaviour(a, b):
    """The two saved models, each loaded after its hashes are verified, give
    identical forecasts, and for ARIMA identical coefficients and in-sample
    values."""
    data = _frame()
    kw = {"target_col": "deaths", "date_col": "week"}
    post = data.loc[data["week"] >= pd.Timestamp(EVENT)]
    year = data.iloc[:52]
    ma, mb = FitRecord.load(a).model(), FitRecord.load(b).model()
    for rows in (post, year):
        np.testing.assert_array_equal(ma.predict(rows, **kw).predicted,
                                      mb.predict(rows, **kw).predicted)
    if hasattr(ma, "_model") and hasattr(ma._model, "predict_in_sample"):
        np.testing.assert_array_equal(ma._model.params(), mb._model.params())
        np.testing.assert_array_equal(ma._model.predict_in_sample(),
                                      mb._model.predict_in_sample())


def _records(out):
    found = {}
    for family in ("prophet_xgb", "arima"):
        label = os.path.join(out, family, "euromomo_heat2026")
        for step, file in (("check", "02_check/02-check.json"), ("fit", "03_fit/03-fit.json")):
            with open(os.path.join(label, file)) as fh:
                found[(family, step)] = json.load(fh)
    return found


def test_the_chain_reproduces_in_fresh_processes(tmp_path):
    data_file = tmp_path / "data.pkl"
    with open(data_file, "wb") as fh:
        pickle.dump(_frame(), fh)
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    out = str(tmp_path / "out")
    runs = []
    for i in range(2):
        # The same output path both times, so every path in the records is the
        # same; the first run is moved aside, out of reach of the prior-check scan.
        subprocess.run([sys.executable, "-c", _CHAIN, tests_dir, out, str(data_file)],
                       check=True, capture_output=True, text=True)
        runs.append(_records(out))
        shutil.move(out, str(tmp_path / f"run{i}"))
    assert set(runs[0]) == set(runs[1])
    for key in runs[0]:
        a, b = (_without_file_bytes(_without_clock(r[key])) for r in runs)
        assert canonical_json(a) == canonical_json(b), key
    for family in ("prophet_xgb", "arima"):
        _same_behaviour(*(os.path.join(tmp_path, f"run{i}", family, "euromomo_heat2026")
                          for i in range(2)))
    # Control: the records do hold what the comparison is meant to see.
    fit = runs[0][("prophet_xgb", "fit")]
    assert fit["saved_model"]["files"] and fit["fitted_values"]["n"] == 181
    assert runs[0][("arima", "check")]["verdict"] == "PASS"
