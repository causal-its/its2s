# Description: Acceptance tests for step 1 (its2s.screen_models) that need a
#   fresh Python process, the written records on disk, or the old tuning path:
#   a candidate re-screened alone in a new interpreter reproduces its sealed
#   entry byte for byte; the written design record's declared_at precedes the
#   candidates record's first_fit_at, and no family fit can run before that
#   stamp; and screening agrees float for float with tune_model on the
#   equivalent start-anchored fold layout. Real arima and prophet_xgb fits on
#   synthetic data, at a few settings each.
# Usage: pytest tests/test_screening_acceptance.py
# Dependencies: pytest, numpy, pandas, pmdarima, prophet, xgboost

import ast
import contextlib
import inspect
import io
import json
import os
import re
import subprocess
import sys
import warnings

import pandas as pd
import pytest

import its2s.screening as screening
from its2s import CandidatesRecord, screen_models
from its2s.candidates_record import RECORD_DIR
from its2s.design import DesignWarning, declare_design
from its2s.design_record import RECORD_DIR as DESIGN_DIR
from its2s.design_record import RECORD_FILE as DESIGN_FILE
from its2s.models.families import get_family
from its2s.records import canonical_json
from its2s.screening import screen_candidate, utc_now
from its2s.search import nest_params
from its2s.tuning import tune_model

from test_design import EVENT, _design, _frame
from test_screening import _fake_family

FAMILIES = (("arima", 2), ("prophet_xgb", 3))


def _declare(tmp, families=FAMILIES):
    d = _design(tmp=str(tmp))
    d.set_seasonality(cycles=["annual"])
    d.add_split("sp1", train_obs=129, test_obs=52, primary=True)
    d.add_cv("cv1", n_folds=3, fold_eval_obs=6)
    for name, n in families:
        d.add_family(name, n_trials=n)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore", DesignWarning)
        return declare_design(d)


@pytest.fixture(scope="module")
def screened_real(tmp_path_factory):
    """One real screening run, written to disk: (label directory, data)."""
    df = _frame()
    declared = _declare(tmp_path_factory.mktemp("real"))
    with contextlib.redirect_stdout(io.StringIO()):
        screen_models(declared, data=df)
    return declared.handoff["output_path"], df


# ---------------------------------------------------------------------------
# A candidate re-screened alone in a fresh process
# ---------------------------------------------------------------------------

_ALONE = """
import sys, pickle
from its2s import DesignRecord
from its2s.records import canonical_json
from its2s.screening import screen_candidate
label_dir, data_file, cid = sys.argv[1:4]
handoff = DesignRecord.load(label_dir).to_dict()["handoff"]
with open(data_file, "rb") as fh:
    data = pickle.load(fh)
sys.stdout.buffer.write(canonical_json(screen_candidate(handoff, data, cid)))
"""


@pytest.mark.parametrize("family", [name for name, _ in FAMILIES])
def test_a_candidate_alone_in_a_fresh_process_reproduces_its_entry(
        screened_real, tmp_path, family):
    label_dir, df = screened_real
    sealed = CandidatesRecord.load(label_dir).to_dict()["candidates"]
    (cid,) = [c for c, e in sealed.items() if e["family"] == family]
    assert sealed[cid]["status"] == "ok"
    data_file = tmp_path / "data.pkl"
    df.to_pickle(data_file)
    # A new interpreter with the threading pinned, reading the design record
    # from disk: nothing is shared with the process that wrote the record.
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    out = subprocess.run([sys.executable, "-c", _ALONE, label_dir, str(data_file), cid],
                         capture_output=True, env=env, timeout=900)
    assert out.returncode == 0, out.stderr.decode()[-2000:]
    assert out.stdout == canonical_json(sealed[cid])


# ---------------------------------------------------------------------------
# The declaration predates the first fit
# ---------------------------------------------------------------------------

def test_the_written_declaration_predates_the_first_fit(screened_real):
    label_dir, _ = screened_real
    with open(os.path.join(label_dir, DESIGN_DIR, DESIGN_FILE), encoding="utf-8") as fh:
        declared_at = json.load(fh)["declared_at"]
    with open(os.path.join(label_dir, RECORD_DIR, CandidatesRecord.RECORD_FILE),
              encoding="utf-8") as fh:
        first_fit_at = json.load(fh)["first_fit_at"]
    stamp = r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{6}\+00:00"
    assert re.fullmatch(stamp, declared_at) and re.fullmatch(stamp, first_fit_at)
    assert declared_at < first_fit_at


def _calls_to(tree, attr):
    """(enclosing function name, line) of every call to <something>.<attr>(...)."""
    found = []
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            for node in ast.walk(fn):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == attr):
                    found.append((fn.name, node.lineno))
    return found


def test_only_the_stamping_helper_reaches_a_family_fit():
    """Structural: every .fit( call in screening is inside _fit, after the tick."""
    tree = ast.parse(inspect.getsource(screening))
    fits = _calls_to(tree, "fit")
    assert fits and {name for name, _ in fits} == {"_fit"}
    (fit_line,) = [line for _, line in fits]
    ticks = [line for name, line in _calls_to(tree, "tick") if name == "_fit"]
    assert ticks and max(ticks) < fit_line


def test_no_fit_starts_before_first_fit_at(tmp_path, monkeypatch):
    """Behavioural: each family fit notes the clock on entry; none precedes the stamp."""
    entered = []
    base = _fake_family()

    class Timed(base):
        def fit(self, *args, **kw):
            entered.append(utc_now())
            return super().fit(*args, **kw)

    monkeypatch.setattr(screening, "load_family_class", lambda name: Timed)
    declared = _declare(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        cands = screen_models(declared, data=_frame())
    assert len(entered) == cands.multiplicity["n_fits"]
    assert declared.declared_at < cands.first_fit_at <= min(entered)


# ---------------------------------------------------------------------------
# Agreement with the old tuning path
# ---------------------------------------------------------------------------

def _old_path(handoff, df, cid):
    """tune_model on the start-anchored layout that equals the sealed folds.

    With no skip, a first training portion the size of the sealed first
    fold's, evaluation slices of the sealed width, and the CV capped at the
    training window's end, the old layout cuts exactly the sealed folds,
    provided the training window starts at the series' first row.
    """
    cand = handoff["candidates"][cid]
    family, spec = cand["family"], get_family(cand["family"])
    split = handoff["splits"][cand["split"]]
    folds = handoff["combinations"][f"{cand['split']}/{cand['cv']}"]["folds"]
    assert pd.Timestamp(split["train"]["start"]) == pd.to_datetime(df["week"]).min()
    overrides = {"data": {"date_col": "week", "target_col": "deaths",
                          "covariate_cols": []}}
    if spec.seed_param is not None:
        # The family's own random state, which screening sets to the
        # candidate's seed, reaches the old path through config_overrides.
        node = overrides.setdefault("models", {}).setdefault(family, {})
        *path, leaf = spec.seed_param
        for key in path:
            node = node.setdefault(key, {})
        node[leaf] = cand["seed"]
    return tune_model(df, EVENT, family, n_trials=cand["n_trials"], n_folds=len(folds),
                      split_method="observations", test_obs=folds[0]["evaluate"]["n"],
                      min_train_obs=folds[0]["train"]["n"], skip_obs=0,
                      cv_end_date=split["train"]["end"], config_overrides=overrides,
                      seed=cand["seed"])


@pytest.mark.parametrize("family", [name for name, _ in FAMILIES])
def test_screening_agrees_with_tune_model(tmp_path, family):
    df = _frame()
    handoff = _declare(tmp_path).to_dict()["handoff"]
    (cid,) = [c for c, e in handoff["candidates"].items() if e["family"] == family]
    new = screen_candidate(handoff, df, cid)
    old = _old_path(handoff, df, cid)
    # Every trial, not only the best: identical floats in the same order.
    assert [t["mean_rmse"] for t in new["search"]] == list(old.trials_df["mean_rmse"])
    assert new["best"]["trial"] == int(old.trials_df["mean_rmse"].idxmin())
    assert new["best"]["mean_rmse"] == old.best_rmse
    chosen = new["search"][new["best"]["trial"]]["setting"]
    assert nest_params(chosen) == old.best_params
