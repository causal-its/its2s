# Description: Step 1's computation: score every declared candidate on its
#   sealed folds. For each candidate it draws the declared number of settings
#   from the family's sealed search space with the candidate's own seed, fits
#   each setting on every fold's training portion and scores the fold's
#   evaluation slice, keeps the setting with the lowest mean fold RMSE, fits
#   that setting once on the whole training window, and evaluates the
#   screening criteria S1-S5. A failure is recorded, never raised.
#   Everything is read from the design record's HANDOFF section and the data;
#   no other field of the design is reachable from here, and nothing here
#   touches the file system.
#   screen_models is the public step: it verifies the sealed design and the
#   data, refuses or checks a repeat run, then screens, seals and writes the
#   candidates record and its report.
# Usage: candidates = its2s.screen_models(declared, data=df)
#        from its2s.screening import screen, screen_candidate
# Dependencies: numpy, pandas

import logging
import os
import time
import traceback
import warnings
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd

from . import __version__
from .candidates_record import (RECORD_DIR, REPORT_FILE, CandidatesRecord,
                                build_record)
from .design import _count, _verdict
from .design_record import SCHEMA_VERSION as DESIGN_SCHEMA_VERSION
from .design_record import DesignRecord
from .fitting import (Columns, FitClock, error_line, fit_family, mase_period,
                      predict_family, rows_in, seasonality_of, selected_structure,
                      utc_now, verify_data)
from .metrics.error_metrics import compute_metrics
from .models.families import get_family, load_family_class, resolve_space
from .records import atomic_directory, canonical_json, record_json
from .search import nest_params, sample_settings
from .settings import _deep_merge

logger = logging.getLogger(__name__)

SCREENING_CRITERIA = ("S1", "S2", "S3", "S4", "S5")

# The step-0 verdict words, as the candidates record spells them.
_VERDICT_WORDS = {"PASS": "pass", "FAIL": "fail", "not applicable": "n/a",
                  "not selected": "not-selected"}


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

def _with_seed(params, seed_param, seed):
    if seed_param is None:
        return params
    *path, leaf = seed_param
    node = params
    for key in path:
        node = node.setdefault(key, {})
    node[leaf] = seed
    return params


def _settings(spec, cand):
    """The settings a candidate scores: (default, [(flat, params), ...]).

    Every setting is complete: the family defaults with the sealed fixed
    values and the drawn values merged over them, and the family's own seed
    set to the candidate's. The default is the family defaults with the
    fixed values applied, which is what the candidate fits with no search.
    """
    space, seed = cand["space"], cand["seed"]
    fixed = dict(space["fixed"])

    def complete(flat):
        params = _deep_merge(spec.defaults, nest_params({**flat, **fixed}))
        return _with_seed(params, spec.seed_param, seed)

    default = complete({})
    if cand["n_trials"] == 0:
        return default, []
    sample = {k: tuple(v) for k, v in space["sample"].items()}
    drawn = sample_settings(sample, cand["n_trials"], seed)
    return default, [(flat, complete(flat)) for flat in drawn]


def _non_default_space(spec, cand, seasonality):
    shipped = resolve_space(spec, seasonality)
    sample = {k: list(v) for k, v in cand["space"]["sample"].items()}
    return bool(cand["space"]["fixed"]) or sample != {
        k: list(v) for k, v in shipped.sample.items()}


# ---------------------------------------------------------------------------
# Scoring one setting on the folds
# ---------------------------------------------------------------------------

def _fold_scores(fold, train, evaluate, predicted, cols, m):
    actual = evaluate[cols.y].to_numpy(dtype=float)
    with warnings.catch_warnings():
        # MAPE is not used here; its zero-count warning is not this step's.
        warnings.simplefilter("ignore")
        metrics = compute_metrics(actual, predicted,
                                  training_actual=train[cols.y].to_numpy(dtype=float),
                                  seasonality=m)
    mase = metrics.mase
    return {"k": fold["k"], "n_train": len(train),
            "evaluate": {"start": fold["evaluate"]["start"],
                         "end": fold["evaluate"]["end"]},
            "rmse": metrics.rmse, "mae": metrics.mae,
            "mase": None if mase is None or not np.isfinite(mase) else mase}


def _score(family, params, seasonality, folds, cols, m, clock, counter):
    """Fit params on every fold; a failure on any fold fails the setting."""
    cls = load_family_class(family)
    per_fold, structure = [], []
    for fold, train, evaluate in folds:
        model = cls(params=params, seasonality=seasonality)
        try:
            fit = fit_family(model, train, cols, clock, counter)
            predicted = predict_family(model, evaluate, cols, counter)
        except Exception as e:
            return {"status": "failed", "fold": fold["k"], "error": error_line(e)}
        per_fold.append(_fold_scores(fold, train, evaluate, predicted, cols, m))
        structure.append(selected_structure(family, fit))
    return {"status": "ok", "folds": per_fold, "structure": structure,
            "mean_rmse": float(np.mean([f["rmse"] for f in per_fold])),
            "mean_mae": float(np.mean([f["mae"] for f in per_fold]))}


def _mean_mase(per_fold):
    values = [f["mase"] for f in per_fold]
    return None if any(v is None for v in values) else float(np.mean(values))


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

def _criterion_values(per_fold, mean_rmse, insample_rmse, combo, m):
    """(value, reason when not computable) for S1-S5."""
    out = {}
    short = [f for f in per_fold if f["mase"] is None]
    out["S1"] = ((None, f"fold {short[0]['k']}'s training portion holds "
                        f"{short[0]['n_train']} rows, not more than m={m}")
                 if short else (_mean_mase(per_fold), None))
    rmses = [f["rmse"] for f in per_fold]
    median = float(np.median(rmses))
    if len(rmses) < 3:
        out["S2"] = (None, f"{len(rmses)} folds; S2 needs at least 3")
    elif median == 0:
        out["S2"] = (None, "the median fold RMSE is zero")
    else:
        out["S2"] = (float(max(rmses)) / median, None)
    out["S3"] = ((None, "the whole-window fit's in-sample RMSE is zero")
                 if insample_rmse == 0 else (mean_rmse / insample_rmse, None))
    s4 = combo["S4"]["value"]
    out["S4"] = (s4, None if s4 is not None else "no seasonal cycle declared")
    out["S5"] = (combo["S5"]["value"], None)
    return out


def _criteria(handoff_criteria, values):
    criteria = {}
    for cid in SCREENING_CRITERIA:
        entry = handoff_criteria[cid]
        value, reason = values[cid]
        criteria[cid] = {
            "value": value, "threshold": entry["threshold"], "op": entry["op"],
            "selected": entry["selected"],
            "defaulted": entry["threshold_defaulted"],
            "verdict": _VERDICT_WORDS[_verdict(entry, value)],
            "reason": reason,
        }
    applicable = [c for c in criteria.values() if c["verdict"] in ("pass", "fail")]
    n_passed = sum(c["verdict"] == "pass" for c in applicable)
    passed = {"n_applicable": len(applicable), "n_passed": n_passed,
              "passed_all": n_passed == len(applicable)}
    return criteria, passed


# ---------------------------------------------------------------------------
# One candidate
# ---------------------------------------------------------------------------

def screen_candidate(handoff, data, cid, clock=None):
    """Screen one candidate; return its entry in the candidates record.

    Depends only on the handoff and the data, so the same candidate screened
    alone, in any order or in a fresh process, gives the same entry.
    """
    clock = clock if clock is not None else FitClock()
    cand = handoff["candidates"][cid]
    family, spec = cand["family"], get_family(cand["family"])
    split = handoff["splits"][cand["split"]]
    combo = handoff["combinations"][f"{cand['split']}/{cand['cv']}"]
    cols = Columns(handoff["data"])
    seasonality = seasonality_of(handoff)
    m = mase_period(seasonality)

    # The training window only: no row at or after its end reaches a family.
    train = rows_in(data, cols.date, split["train"], f"training window {cand['split']}")
    folds = [(f, rows_in(train, cols.date, f["train"], f"fold {f['k']} training portion"),
              rows_in(train, cols.date, f["evaluate"], f"fold {f['k']} evaluation slice"))
             for f in combo["folds"]]

    entry = {
        "status": "ok", "family": family, "split": cand["split"], "cv": cand["cv"],
        "primary": split["primary"], "seed": cand["seed"], "n_trials": cand["n_trials"],
        "m": cand["m"], "mase_m": m,
        "non_default_space": _non_default_space(spec, cand, seasonality),
    }
    warn = {"search": Counter(), "default_reference": Counter(),
            "whole_window": Counter()}

    default, trials = _settings(spec, cand)
    if trials:
        scored = []
        for i, (flat, params) in enumerate(trials):
            s = _score(family, params, seasonality, folds, cols, m, clock, warn["search"])
            scored.append({"trial": i, "flat": flat, "params": params, **s})
        ok = [t for t in scored if t["status"] == "ok"]
        reference = _score(family, default, seasonality, folds, cols, m, clock,
                           warn["default_reference"])
        entry["search"] = [
            {"trial": t["trial"], "setting": t["flat"], "status": t["status"],
             "mean_rmse": t.get("mean_rmse"),
             **({"fold": t["fold"], "error": t["error"]} if t["status"] == "failed" else {})}
            for t in scored]
        means = [t["mean_rmse"] for t in ok]
        entry["spread"] = (None if not ok else
                           {"best": float(min(means)), "median": float(np.median(means)),
                            "worst": float(max(means)), "n_ok": len(ok)})
        entry["default_reference"] = (
            {"status": "failed", "fold": reference["fold"], "error": reference["error"]}
            if reference["status"] == "failed" else
            {"status": "ok", "mean_rmse": reference["mean_rmse"],
             "mean_mase": _mean_mase(reference["folds"]),
             "folds": [{"k": f["k"], "rmse": f["rmse"], "mase": f["mase"]}
                       for f in reference["folds"]]})
        if not ok:
            return _failed(entry, warn, "search",
                           f"all {len(scored)} settings failed; first: "
                           f"fold {scored[0]['fold']}, {scored[0]['error']}", None)
        best = min(ok, key=lambda t: (t["mean_rmse"], t["trial"]))
    else:
        entry["search"], entry["spread"], entry["default_reference"] = None, None, None
        s = _score(family, default, seasonality, folds, cols, m, clock, warn["search"])
        if s["status"] == "failed":
            return _failed(entry, warn, "search",
                           f"fold {s['fold']}, {s['error']}", None)
        best = {"trial": None, "params": default, **s}

    model = load_family_class(family)(params=best["params"], seasonality=seasonality)
    try:
        fit = fit_family(model, train, cols, clock, warn["whole_window"])
    except Exception as e:
        return _failed(entry, warn, "whole_window", error_line(e), traceback.format_exc())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        insample = compute_metrics(train[cols.y].to_numpy(dtype=float),
                                   np.asarray(fit.fitted_values, dtype=float)).rmse

    entry["best"] = {
        "trial": best["trial"], "params": best["params"],
        "mean_rmse": best["mean_rmse"], "mean_mae": best["mean_mae"],
        "mean_mase": _mean_mase(best["folds"]), "folds": best["folds"],
    }
    entry["whole_window"] = {"insample_rmse": insample, "n": len(train)}
    values = _criterion_values(best["folds"], best["mean_rmse"], insample, combo, m)
    entry["criteria"], entry["passed"] = _criteria(handoff["criteria"], values)
    entry["fitted_structure"] = _fitted_structure(family, best["structure"], fit, model)
    entry["warnings"] = _warnings(warn)
    return entry


def _fitted_structure(family, fold_structure, fit, model):
    if family == "arima":
        return {"folds": fold_structure, "whole_window": selected_structure(family, fit)}
    return {"whole_window": model.seasonal_components()}


def _warnings(warn):
    return {kind: dict(sorted(c.items())) for kind, c in warn.items()}


def _failed(entry, warn, stage, error, tb):
    entry.update(status="failed", best=None, whole_window=None, criteria=None,
                 passed={"n_applicable": 0, "n_passed": 0, "passed_all": False},
                 fitted_structure=None, warnings=_warnings(warn),
                 failure={"stage": stage, "error": error, "traceback": tb})
    return entry


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def screen(handoff, data):
    """Screen every candidate in the handoff's order; return the record body.

    The body is unsealed and carries no step timestamps except first_fit_at;
    the caller stamps, seals and writes it.
    """
    clock = FitClock()
    candidates = {}
    ids = list(handoff["candidates"])
    for n, cid in enumerate(ids, 1):
        started = time.monotonic()
        candidates[cid] = screen_candidate(handoff, data, cid, clock)
        c = candidates[cid]
        logger.info("screened %s (%d of %d): %s, %d settings, %.1fs", cid, n, len(ids),
                    c["status"], c["n_trials"], time.monotonic() - started)

    budgets = {}
    for c in handoff["candidates"].values():
        budgets[c["family"]] = c["n_trials"]
    ok = {cid: c for cid, c in candidates.items() if c["status"] == "ok"}
    passing = [cid for cid, c in ok.items() if c["passed"]["passed_all"]]
    lowest = None
    if ok:
        low = min(c["best"]["mean_rmse"] for c in ok.values())
        lowest = {"mean_rmse": low,
                  "ids": [cid for cid, c in ok.items() if c["best"]["mean_rmse"] == low]}
    return {
        "first_fit_at": clock.first_fit_at,
        "combinations": {
            key: {"split": c["split"], "cv": c["cv"], "n_folds": len(c["folds"]),
                  "folds": [dict(f) for f in c["folds"]],
                  "S4": dict(c["S4"]), "S5": dict(c["S5"])}
            for key, c in handoff["combinations"].items()},
        "budgets": {"n_trials": budgets, "unequal": len(set(budgets.values())) > 1},
        "candidates": candidates,
        "multiplicity": {"n_candidates": len(candidates), "n_ok": len(ok),
                         "n_failed": len(candidates) - len(ok), "n_fits": clock.n_fits,
                         "n_passed_all": len(passing)},
        "facts": {"lowest_mean_cv_rmse": lowest, "passing_all_applicable": passing},
    }


# ---------------------------------------------------------------------------
# The public step
# ---------------------------------------------------------------------------

class ScreeningError(ValueError):
    """screen_models stopped before fitting anything, or a rerun did not
    reproduce the sealed record. A ValueError, like DesignError."""


def _verify_design(declared):
    if not isinstance(declared, DesignRecord):
        raise TypeError(
            f"screen_models takes the DesignRecord that declare_design returned, or "
            f"one read with DesignRecord.load; got {type(declared).__name__}.")
    if declared.schema_version != DESIGN_SCHEMA_VERSION:
        raise ScreeningError(
            f"This design record is schema version {declared.schema_version}; this "
            f"package screens version {DESIGN_SCHEMA_VERSION}, whose handoff carries "
            "the S4, S5 and coverage facts screening reads. Declare the design again "
            "under a new label.")


def _minutes(start, end):
    seconds = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    return max(1, round(seconds / 60))


def _refuse_rerun(existing, label):
    m = existing.multiplicity
    raise ScreeningError(
        f"{label} was screened on {existing.sealed_at[:10]}: "
        f"{m['n_candidates']} candidates, {m['n_fits']} fits, "
        f"{_count(_minutes(existing.screening_started_at, existing.sealed_at), 'minute')}. To "
        "recompute every candidate and check that each reproduces its sealed entry, "
        "call again with rerun=True. Nothing has been fit.")


def _check_rerun(existing, body, label):
    """Keep the sealed record if every candidate reproduced; else raise."""
    sealed = existing.to_dict()["candidates"]
    differ = [cid for cid in sorted(set(sealed) | set(body["candidates"]))
              if cid not in sealed or cid not in body["candidates"]
              or canonical_json(sealed[cid]) != canonical_json(body["candidates"][cid])]
    if differ:
        raise ScreeningError(
            f"Rerunning {label} did not reproduce its sealed candidates record: "
            f"{', '.join(differ)} differ. Nothing was written. The same sealed design "
            "on the same data must give the same candidates, so this is a "
            "reproducibility failure to investigate, not a newer answer.")
    logger.info("rerun of %s reproduced all %d candidates; the sealed record is kept",
                label, len(sealed))
    return existing


def screen_models(declared, data, rerun=False):
    """Step 1: screen every declared candidate on the training window only.

    In order: check the design record and that the data is the data it was
    sealed against; if this label was already screened, refuse (rerun=False)
    or recompute and require every candidate to reproduce its sealed entry
    (rerun=True); otherwise screen, seal the candidates record, print its
    report, and write both read-only under the label's 01_screening/
    directory in one move.

    Parameters
    ----------
    declared : DesignRecord
        From declare_design, or DesignRecord.load.
    data : pd.DataFrame
        The series the design was declared on.
    rerun : bool
        Only matters when the label was already screened; see above.

    Returns
    -------
    CandidatesRecord

    Raises
    ------
    ScreeningError
        Before any fit when the record or the data does not match, or when
        the label was already screened and rerun is False; after the
        recomputation when a rerun does not reproduce.
    """
    _verify_design(declared)
    design = declared.to_dict()
    handoff = design["handoff"]
    label = design["label"]
    fp = verify_data(handoff, data, ScreeningError)

    target = os.path.join(handoff["output_path"], RECORD_DIR)
    existing = CandidatesRecord.load(target) if os.path.exists(target) else None
    if existing is not None and not rerun:
        _refuse_rerun(existing, label)

    started_at = utc_now()
    body = screen(handoff, data)
    if existing is not None:
        return _check_rerun(existing, body, label)

    record = build_record(design, fp, body, started_at, utc_now(), __version__)
    result = CandidatesRecord.from_dict(record)
    report = result.report()
    with atomic_directory(target) as tmp:
        with open(os.path.join(tmp, CandidatesRecord.RECORD_FILE), "w",
                  encoding="utf-8") as fh:
            fh.write(record_json(record))
        with open(os.path.join(tmp, REPORT_FILE), "w", encoding="utf-8") as fh:
            fh.write(report)
    print(report, end="")
    return CandidatesRecord.from_dict(
        record, path=os.path.join(target, CandidatesRecord.RECORD_FILE))
