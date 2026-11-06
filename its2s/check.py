# Description: Step 2's computation and its public call. check() refits the
#   chosen candidate at each origin sealed for its split, on every row before
#   the origin, forecasts H rows into the test window, scores each forecast,
#   and applies the check criteria C1-C3 to the aggregate. It reads the
#   candidates record's HANDOFF section and the data, nothing else, and
#   touches no file. check_selected_model is the public step: it verifies
#   the candidates record, the data, the id and any threshold overrides,
#   refuses a label whose test window is already spent, counts earlier checks
#   on the same pre-event data, then runs the check once, seals and writes
#   the check record and its report with the test window marked spent.
# Usage: check = its2s.check_selected_model(candidates, "prophet_xgb/sp1/cv1", data=df)
#        from its2s.check import check
# Dependencies: numpy, pandas

import json
import math
import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from . import __version__
from .candidates_record import RECORD_DIR as CANDIDATES_DIR
from .candidates_record import RECORD_FILE as CANDIDATES_FILE
from .candidates_record import SCHEMA_VERSION as CANDIDATES_SCHEMA_VERSION
from .candidates_record import CandidatesRecord
from .check_record import RECORD_DIR, REPORT_FILE, CheckRecord, build_record
from .design import _verdict
from .design_record import RECORD_DIR as DESIGN_DIR
from .design_record import RECORD_FILE as DESIGN_FILE
from .design_record import _iso, fingerprint
from .fitting import (Columns, FitClock, error_line, fit_family, mase_period,
                      predict_family, rows_in, seasonality_of,
                      selected_structure, utc_now, verify_data)
from .metrics.error_metrics import compute_metrics
from .models.families import load_family_class
from .origins import max_seasonal_gap
from .records import atomic_directory, record_json, seal_matches

CHECK_CRITERIA = ("C1", "C2", "C3")

# The step-0 verdict words, as the check record spells them.
_VERDICT_WORDS = {"PASS": "pass", "FAIL": "fail", "not applicable": "n/a",
                  "not selected": "not-selected"}

# An argument left out, as distinct from None, which deselects a criterion.
_UNSET = object()


class CheckError(ValueError):
    """check_selected_model stopped before fitting anything. A ValueError,
    like ScreeningError; nothing has been fit and the test window is unspent."""


class CheckWarning(UserWarning):
    """An earlier check was run on the same pre-event data under another label."""


# ---------------------------------------------------------------------------
# One origin
# ---------------------------------------------------------------------------

def _span(df, date_col):
    dates = pd.to_datetime(df[date_col])
    return {"first": _iso(dates.iloc[0]), "last": _iso(dates.iloc[-1]), "n": len(df)}


def _origin_rows(train, test, date_col, origin, H):
    """(fit rows, forecast rows) for one origin.

    The fit takes every row of the split before the origin; the forecast the
    H rows of the test window from the origin on. Nothing at or after the
    test window's end is in either.
    """
    o = pd.Timestamp(origin)
    before = test.loc[pd.to_datetime(test[date_col]) < o]
    fit_rows = pd.concat([train, before])
    ahead = test.loc[pd.to_datetime(test[date_col]) >= o].iloc[:H]
    if len(ahead) != H:
        raise ValueError(f"origin {origin} leaves {len(ahead)} test-window rows to "
                         f"forecast; the design sealed H={H}.")
    return fit_rows, ahead


def _scores(fit_rows, ahead, predicted, cols, m):
    actual = ahead[cols.y].to_numpy(dtype=float)
    with warnings.catch_warnings():
        # MAPE is not used here; its zero-count warning is not this step's.
        warnings.simplefilter("ignore")
        metrics = compute_metrics(actual, predicted,
                                  training_actual=fit_rows[cols.y].to_numpy(dtype=float),
                                  seasonality=m)
    mase, reason = metrics.mase, None
    if mase is None or not np.isfinite(mase):
        mase = None
        reason = (f"its fit rows hold {len(fit_rows)}, not more than m={m}"
                  if len(fit_rows) <= m else "the seasonal naive's error is zero")
    return {
        "forecast": {"dates": [_iso(d) for d in pd.to_datetime(ahead[cols.date])],
                     "actual": actual.tolist(), "predicted": predicted.tolist(),
                     "error": (actual - predicted).tolist()},
        "rmse": metrics.rmse, "mae": metrics.mae, "mase": mase, "mase_reason": reason,
    }


def _check_origin(k, origin, phase_origin, rows, cand, seasonality, cols, m, clock):
    """Fit, forecast and score one origin; a failure is recorded, never raised."""
    fit_rows, ahead = rows
    entry = {"k": k, "date": origin, "event_phase": origin == phase_origin,
             "fit": _span(fit_rows, cols.date)}
    counter = Counter()
    model = load_family_class(cand["family"])(params=cand["params"],
                                              seasonality=seasonality)
    try:
        fit = fit_family(model, fit_rows, cols, clock, counter)
        predicted = predict_family(model, ahead, cols, counter)
    except Exception as e:
        entry.update(status="failed", error=error_line(e),
                     warnings=dict(sorted(counter.items())))
        return entry
    entry["status"] = "ok"
    entry.update(_scores(fit_rows, ahead, predicted, cols, m))
    entry["structure"] = (selected_structure("arima", fit) if cand["family"] == "arima"
                          else model.seasonal_components())
    entry["warnings"] = dict(sorted(counter.items()))
    return entry


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------

def _criterion_values(origins, cv_rmse):
    """(value, reason when not computable) for C1-C3, and the aggregate."""
    failed = [o for o in origins if o["status"] == "failed"]
    if failed:
        reason = f"origin {failed[0]['k']} failed"
        aggregate = {"n_origins": len(origins), "n_failed": len(failed),
                     "mean_mase": None, "mean_rmse": None, "cv_rmse": cv_rmse,
                     "worst_rmse": None, "median_rmse": None}
        return {cid: (None, reason) for cid in CHECK_CRITERIA}, aggregate

    rmses = [o["rmse"] for o in origins]
    short = [o for o in origins if o["mase"] is None]
    mean_mase = None if short else float(np.mean([o["mase"] for o in origins]))
    mean_rmse = float(np.mean(rmses))
    worst, median = float(max(rmses)), float(np.median(rmses))
    out = {}
    out["C1"] = ((None, f"origin {short[0]['k']}'s MASE is n/a: {short[0]['mase_reason']}")
                 if short else (mean_mase, None))
    out["C2"] = ((None, "the step 1 tuning metric (cv_rmse) is zero")
                 if cv_rmse == 0 else (mean_rmse / cv_rmse, None))
    out["C3"] = ((None, "the median origin RMSE is zero")
                 if median == 0 else (worst / median, None))
    aggregate = {"n_origins": len(origins), "n_failed": 0, "mean_mase": mean_mase,
                 "mean_rmse": mean_rmse, "cv_rmse": cv_rmse, "worst_rmse": worst,
                 "median_rmse": median}
    return out, aggregate


def _judge(thresholds, values):
    """Each criterion's entry, the verdict, and every reason it is not a pass.

    A selected criterion that cannot be computed fails the check: a gate
    that could not be evaluated has not been passed.
    """
    criteria, reasons = {}, []
    for cid in CHECK_CRITERIA:
        t = thresholds[cid]
        value, reason = values[cid]
        word = _VERDICT_WORDS[_verdict(t, value)]
        criteria[cid] = {**t, "value": value, "verdict": word, "reason": reason}
        if t["selected"] and word == "fail":
            reasons.append(f"{cid} {value:.3f} fails {t['op']} {t['threshold']}")
        elif t["selected"] and word == "n/a":
            reasons.append(f"{cid} is selected and not computable: {reason}")
    return criteria, ("FAIL" if reasons else "PASS"), reasons


# ---------------------------------------------------------------------------
# The computation
# ---------------------------------------------------------------------------

def check(handoff, data, cid, thresholds, clock=None):
    """Check one candidate at every sealed origin of its split.

    Parameters
    ----------
    handoff : dict
        The candidates record's HANDOFF section; nothing else is read.
    data : pd.DataFrame
    cid : str
        A candidate id in the handoff whose status is ok.
    thresholds : dict
        C1-C3, each with selected, op and threshold (and whatever else the
        caller records beside them), as resolve_thresholds returns.
    clock : FitClock, optional

    Returns
    -------
    dict
        The record body: origins, aggregate, criteria, verdict, the reasons
        for a FAIL, the maximum seasonal gap, first_fit_at and the fit count.
    """
    clock = clock if clock is not None else FitClock()
    cand = handoff["candidates"][cid]
    split = handoff["splits"][cand["split"]]
    sealed = handoff["origins"][cand["split"]]
    cols = Columns(handoff["data"])
    seasonality = seasonality_of(handoff)
    m = mase_period(seasonality)
    H = handoff["question"]["H"]

    # The split's training and test windows only: no row at or after the
    # test window's end reaches a family or a metric.
    train = rows_in(data, cols.date, split["train"], f"training window {cand['split']}")
    test = rows_in(data, cols.date, split["test"], f"test window {cand['split']}")

    origins = []
    for k, o in enumerate(sealed["dates"], 1):
        rows = _origin_rows(train, test, cols.date, o, H)
        origins.append(_check_origin(k, o, sealed["phase_origin"], rows, cand,
                                     seasonality, cols, m, clock))

    values, aggregate = _criterion_values(origins, cand["cv_rmse"])
    criteria, verdict, reasons = _judge(thresholds, values)
    aggregate["mase_m"] = m
    return {
        "first_fit_at": clock.first_fit_at,
        "n_fits": clock.n_fits,
        "origins": origins,
        "aggregate": {**aggregate, "verdict_reasons": reasons},
        "criteria": criteria,
        "verdict": verdict,
        "max_seasonal_gap": max_seasonal_gap(sealed["dates"], seasonality),
    }


# ---------------------------------------------------------------------------
# Checks made before any fit
# ---------------------------------------------------------------------------

def _verify_candidates(candidates):
    if not isinstance(candidates, CandidatesRecord):
        raise TypeError(
            "check_selected_model takes the CandidatesRecord that screen_models "
            f"returned, or one read with CandidatesRecord.load; got "
            f"{type(candidates).__name__}.")
    if candidates.schema_version != CANDIDATES_SCHEMA_VERSION:
        raise CheckError(
            f"This candidates record is schema version {candidates.schema_version}; "
            f"this package checks version {CANDIDATES_SCHEMA_VERSION}. Declare the "
            "design again under a new label.")


def _verify_id(handoff, cid):
    cands = handoff["candidates"]
    ok = [c for c, v in cands.items() if v["status"] == "ok"]
    if cid not in cands:
        raise CheckError(f"No candidate {cid!r} in this record. The candidates that "
                         f"fit: {', '.join(ok) or 'none'}.")
    if cands[cid]["status"] != "ok":
        raise CheckError(f"{cid} failed at screening and cannot be checked. The "
                         f"candidates that fit: {', '.join(ok) or 'none'}.")


def _override(cid, value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{cid} must be a positive number, or None to deselect it; "
                        f"got {value!r}.")
    if not math.isfinite(value) or value <= 0:
        raise CheckError(f"{cid} must be a positive finite number, or None to "
                         f"deselect it; got {value!r}.")
    return float(value)


def resolve_thresholds(handoff_criteria, overrides):
    """C1-C3 as step 0 sealed them, with any override applied and stamped.

    overrides maps a criterion to a number (apply this threshold, selecting
    the criterion) or None (deselect it); a criterion absent from it keeps
    its step 0 setting.
    """
    out = {}
    for cid in CHECK_CRITERIA:
        c = handoff_criteria[cid]
        entry = {"op": c["op"], "step0_threshold": c["threshold"],
                 "step0_selected": c["selected"], "defaulted": c["threshold_defaulted"],
                 "overridden": cid in overrides, "override": None,
                 "threshold": c["threshold"], "selected": c["selected"]}
        if cid in overrides:
            value = _override(cid, overrides[cid])
            entry["override"] = value
            entry["selected"] = value is not None
            if value is not None:
                entry["threshold"] = value
        out[cid] = entry
    if not any(e["selected"] for e in out.values()):
        raise CheckError(
            "No check criterion is selected. The check has no override to proceed "
            "without a criterion; keep at least one of C1, C2, C3. Nothing has been fit.")
    return out


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _refuse_spent(target, label):
    rec = _read_json(os.path.join(target, CheckRecord.RECORD_FILE)) or {}
    when = (rec.get("checked_at") or "an unknown date")[:10]
    raise CheckError(
        f"The test window of {label} was spent on {when} by "
        f"{rec.get('candidate_id', 'an unreadable record')}, verdict "
        f"{rec.get('verdict', 'unknown')}. A second check is not offered; declare "
        "again under a new label. Nothing has been fit.")


def _pre_event_fingerprint(handoff, data):
    d = handoff["data"]
    return fingerprint(data, d["date_col"], [d["outcome_col"], *d["covariate_cols"]],
                       handoff["question"]["event_date"])


def _prior_checks(output_path, digest):
    """Every check record under the same output directory whose pre-event
    fingerprint matches. A record with a broken seal is still counted, and
    named as such: it is still evidence that the test window was seen."""
    output_dir = os.path.dirname(output_path) or "."
    found = []
    if not os.path.isdir(output_dir):
        return found
    for label in sorted(os.listdir(output_dir)):
        rec = _read_json(os.path.join(output_dir, label, RECORD_DIR,
                                      CheckRecord.RECORD_FILE))
        if not isinstance(rec, dict):
            continue
        pre = rec.get("pre_event_fingerprint")
        if isinstance(pre, dict) and pre.get("digest") == digest:
            found.append({"label": label, "checked_at": rec.get("checked_at"),
                          "candidate_id": rec.get("candidate_id"),
                          "verdict": rec.get("verdict"),
                          "seal_intact": seal_matches(rec)})
    return found


def _warn_prior(found):
    if not found:
        return
    named = "; ".join(
        f"{p['label']} on {(p['checked_at'] or 'an unknown date')[:10]}"
        + ("" if p["seal_intact"] else " (its record no longer matches its seal)")
        for p in found)
    warnings.warn(
        f"The test window of this data was already seen by {len(found)} earlier "
        f"check(s): {named}. This check is weaker evidence than a first one; the "
        "count is stamped into the record and every report from here on.",
        CheckWarning, stacklevel=3)


# ---------------------------------------------------------------------------
# The public step
# ---------------------------------------------------------------------------

def _record_fields(candidates, cid, rationale, body, fp, pre_fp, prior, checked_at,
                   sealed_at):
    rec = candidates.to_dict()
    h = rec["handoff"]
    c = h["candidates"][cid]
    out = h["output_path"]
    return {
        "package_version": __version__,
        "checked_at": checked_at,
        "first_fit_at": body["first_fit_at"],
        "sealed_at": sealed_at,
        "candidate_id": cid,
        "rationale": rationale,
        "verdict": body["verdict"],
        "criteria": body["criteria"],
        "origins": body["origins"],
        "aggregate": body["aggregate"],
        "max_seasonal_gap": body["max_seasonal_gap"],
        "candidate": rec["candidates"][cid],
        "screening_facts": {"passed_all": c["passed_all"],
                            "n_applicable": c["n_applicable"], "n_passed": c["n_passed"],
                            "n_candidates_passed_all": h["multiplicity"]["n_passed_all"]},
        "multiplicity": {**h["multiplicity"], "n_check_fits": body["n_fits"]},
        "split_chosen_after_evidence": not c["primary"],
        "prior_checks": {"count": len(prior), "records": prior},
        "test_window_spent": True,
        "provenance": {
            "design": {"seal": rec["design"]["seal"]["digest"],
                       "path": os.path.join(out, DESIGN_DIR, DESIGN_FILE)},
            "candidates": {"seal": rec["seal"]["digest"],
                           "path": candidates.path
                           or os.path.join(out, CANDIDATES_DIR, CANDIDATES_FILE)},
        },
        "data_fingerprint": fp,
        "pre_event_fingerprint": pre_fp,
    }


def check_selected_model(candidates, candidate_id, data, rationale=None,
                         C1=_UNSET, C2=_UNSET, C3=_UNSET):
    """Step 2: check one chosen candidate on the test window, once.

    In order, all before the first fit: check the candidates record, that
    the data is the data it was sealed against, that the id names a
    candidate that fit, and the overrides; refuse if this label's test
    window was already spent; count earlier checks on the same pre-event
    data under the same output directory, warning when there are any. Then
    refit the candidate at every sealed origin of its split, score each
    H-row forecast, apply C1-C3, seal and write the check record and its
    report read-only under the label's 02_check/ directory, and print the
    report. A FAIL is returned, not raised; no later step runs on it.

    Parameters
    ----------
    candidates : CandidatesRecord
        From screen_models, or CandidatesRecord.load.
    candidate_id : str
        One id from the candidate index, of a candidate that fit.
    data : pd.DataFrame
        The series the design was declared on.
    rationale : str, optional
        One sentence on why this candidate; stamped into the record.
    C1, C2, C3 : float or None, optional
        A threshold to apply in place of step 0's, stamped beside it; None
        deselects the criterion. At least one must stay selected.

    Returns
    -------
    CheckRecord

    Raises
    ------
    CheckError
        Before any fit, when the record, the data, the id or an override is
        not acceptable, when no criterion is selected, or when this label's
        test window has already been spent. Nothing is spent.
    """
    _verify_candidates(candidates)
    handoff = candidates.to_dict()["handoff"]
    fp = verify_data(handoff, data, CheckError)
    if not isinstance(candidate_id, str):
        raise TypeError(f"candidate_id must be a string; got {candidate_id!r}.")
    _verify_id(handoff, candidate_id)
    if rationale is not None and not isinstance(rationale, str):
        raise TypeError(f"rationale must be a string or None; got {rationale!r}.")
    overrides = {cid: v for cid, v in (("C1", C1), ("C2", C2), ("C3", C3))
                 if v is not _UNSET}
    thresholds = resolve_thresholds(handoff["criteria"], overrides)

    label = os.path.basename(os.path.normpath(handoff["output_path"]))
    target = os.path.join(handoff["output_path"], RECORD_DIR)
    if os.path.exists(target):
        _refuse_spent(target, label)

    pre_fp = _pre_event_fingerprint(handoff, data)
    prior = _prior_checks(handoff["output_path"], pre_fp["digest"])
    _warn_prior(prior)

    checked_at = utc_now()
    body = check(handoff, data, candidate_id, thresholds)
    fields = _record_fields(candidates, candidate_id, rationale, body, fp, pre_fp, prior,
                            checked_at, utc_now())
    record = build_record(fields, handoff)
    result = CheckRecord.from_dict(record)
    report = result.report()
    with atomic_directory(target) as tmp:
        with open(os.path.join(tmp, CheckRecord.RECORD_FILE), "w", encoding="utf-8") as fh:
            fh.write(record_json(record))
        with open(os.path.join(tmp, REPORT_FILE), "w", encoding="utf-8") as fh:
            fh.write(report)
    print(report, end="")
    return CheckRecord.from_dict(record, path=os.path.join(target, CheckRecord.RECORD_FILE))
