# Description: Step 3's computation and its public call. fit() fits the
#   checked candidate once, with its sealed parameters and the declared
#   seasonality, on every row of its split before the event; it forecasts
#   nothing and scores nothing. It reads the check record's HANDOFF section
#   and the data, nothing else, and touches no file. fit_final_model is the
#   public step: it refuses any check that did not pass, verifies the data,
#   refuses a label already fitted (or, with rerun=True, refits and requires
#   the sealed fit to reproduce), then fits, saves the model and seals the
#   fit record beside it.
# Usage: fit = its2s.fit_final_model(check, data=df)
#        model = fit.model()
# Dependencies: numpy

import hashlib
import os
from collections import Counter

import numpy as np

from . import __version__
from .check_record import SCHEMA_VERSION as CHECK_SCHEMA_VERSION
from .check_record import CheckRecord
from .fit_record import (MODEL_DIR, MODEL_PATH, RECORD_DIR, FitRecord,
                         build_record, model_files)
from .fitting import (Columns, FitClock, fit_family, rows_in, seasonality_of,
                      selected_structure, span_of, utc_now, verify_data)
from .models.base import _versions
from .models.families import load_family_class
from .records import atomic_directory, canonical_json, record_json


class FitError(ValueError):
    """fit_final_model stopped before fitting anything, or a rerun did not
    reproduce the sealed fit. A ValueError, like CheckError."""


# ---------------------------------------------------------------------------
# The computation
# ---------------------------------------------------------------------------

def fitted_values_digest(values):
    """A hash of the in-sample fitted values, as little-endian 64-bit floats.

    Two fits with identical fitted values give the same digest, so a rerun
    can be compared with the sealed fit while the values themselves are
    never written, scored or shown.
    """
    v = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return {"algorithm": "sha256", "digest": hashlib.sha256(v.tobytes()).hexdigest(),
            "n": int(v.size)}


def pre_event_rows(handoff, data):
    """Every row of the candidate's split before the event: its training
    window and its test window, which ends where the event begins."""
    c = handoff["candidate"]
    event = handoff["question"]["event_date"]
    if c["test"]["end"] != event:
        raise ValueError(f"The test window of {c['split']} ends on {c['test']['end']}, "
                         f"not on the event date {event}.")
    bounds = {"start": c["train"]["start"], "end": event,
              "n": c["train"]["n"] + c["test"]["n"]}
    return rows_in(data, handoff["data"]["date_col"], bounds,
                   f"pre-event window of {c['split']}")


def fit(handoff, data, clock=None):
    """Fit the checked candidate once on every pre-event row of its split.

    Parameters
    ----------
    handoff : dict
        The check record's HANDOFF section; nothing else is read.
    data : pd.DataFrame
    clock : FitClock, optional

    Returns
    -------
    dict
        The fitted model (under "model") and the facts the fit record
        states about it: when it was fit, the span, the forecast origin,
        ARIMA's selected order, the seasonal components, the digest of the
        fitted values and the library warnings, counted by category.
    """
    clock = clock if clock is not None else FitClock()
    c = handoff["candidate"]
    cols = Columns(handoff["data"])
    seasonality = seasonality_of(handoff)
    rows = pre_event_rows(handoff, data)

    counter = Counter()
    model = load_family_class(c["family"])(params=c["params"], seasonality=seasonality)
    result = fit_family(model, rows, cols, clock, counter)
    span = span_of(rows, cols.date)
    return {
        "model": model,
        "fitted_at": clock.first_fit_at,
        "span": span,
        # The last row fitted; step 4 forecasts from the next date, the event.
        "forecast_origin": span["last"],
        "selected_order": selected_structure(c["family"], result),
        "seasonal_components": model.seasonal_components(),
        "fitted_values": fitted_values_digest(result.fitted_values),
        "warnings": dict(sorted(counter.items())),
    }


# ---------------------------------------------------------------------------
# Checks made before any fit
# ---------------------------------------------------------------------------

def _verify_check(check):
    if not isinstance(check, CheckRecord):
        raise TypeError(
            "fit_final_model takes the CheckRecord that check_selected_model "
            f"returned, or one read with CheckRecord.load; got {type(check).__name__}.")
    if check.schema_version != CHECK_SCHEMA_VERSION:
        raise FitError(
            f"This check record is schema version {check.schema_version}; this "
            f"package fits from version {CHECK_SCHEMA_VERSION}. Declare the design "
            "again under a new label.")
    handoff = check.to_dict()["handoff"]
    if handoff["verdict"] != "PASS":
        raise FitError(
            f"The check of {handoff['label']} FAILED on {check.checked_at[:10]}; no "
            "model is fit and no effect is estimated on this label. Nothing has been fit.")
    return handoff


def _refuse_rerun(existing, label):
    raise FitError(
        f"{label} was fitted on {existing.fitted_at[:10]}; its record is at "
        f"{existing.path}, and FitRecord.load reads it with its model. To refit and "
        "check that the fit reproduces, call again with rerun=True. Nothing has been fit.")


def _check_rerun(existing, handoff, body, label):
    """Keep the sealed fit if the refit reproduced it; else raise."""
    sealed = existing.to_dict()
    refit = {"params": handoff["candidate"]["params"], "span": body["span"],
             "selected_order": body["selected_order"],
             "fitted_values": body["fitted_values"]}
    names = {"params": "parameters", "span": "data span",
             "selected_order": "ARIMA's selected order",
             "fitted_values": "in-sample fitted values"}
    differ = [names[k] for k in refit
              if canonical_json(sealed[k]) != canonical_json(refit[k])]
    if differ:
        raise FitError(
            f"Refitting {label} did not reproduce its sealed fit: {', '.join(differ)} "
            "differ. Nothing was written. The same checked candidate on the same data "
            "must give the same fit, so this is a reproducibility failure to "
            "investigate, not a newer model.")
    return existing


# ---------------------------------------------------------------------------
# The public step
# ---------------------------------------------------------------------------

def _record_fields(check, body, model, fp, sealed_at):
    h = check.to_dict()["handoff"]
    c = h["candidate"]
    return {
        "package_version": __version__,
        "fitted_at": body["fitted_at"],
        "sealed_at": sealed_at,
        "label": h["label"],
        "candidate_id": c["id"],
        "family": c["family"],
        "params": c["params"],
        "seed": c["seed"],
        "span": body["span"],
        "forecast_origin": body["forecast_origin"],
        "selected_order": body["selected_order"],
        "seasonal_components": body["seasonal_components"],
        "fitted_values": body["fitted_values"],
        "library_versions": _versions(getattr(body["model"], "_libraries", ())),
        "warnings": body["warnings"],
        "stamps": h["stamps"],
        "provenance": {"check": {"seal": check.seal["digest"], "path": check.path
                                 or os.path.join(h["output_path"], CheckRecord.RECORD_DIR,
                                                 CheckRecord.RECORD_FILE)}},
        "saved_model": model,
        "data_fingerprint": fp,
    }


def _summary(rec, record_path):
    lines = [f"STEP 3 -- FINAL FIT, SEALED          series: {rec['label']}",
             f"  candidate          {rec['candidate_id']}",
             f"  fitted on          {rec['span']['first']} .. {rec['span']['last']}  "
             f"({rec['span']['n']} rows)",
             f"  forecast origin    {rec['forecast_origin']}; step 4 forecasts from "
             f"{rec['handoff']['question']['event_date']}"]
    if rec["selected_order"] is not None:
        s = rec["selected_order"]
        lines.append(f"  ARIMA order        {tuple(s['order'])} x "
                     f"{tuple(s['seasonal_order'])}")
    lines += [f"  model              {os.path.join(os.path.dirname(record_path), MODEL_DIR)}",
              f"  record             {record_path}"]
    return "\n".join(lines) + "\n"


def fit_final_model(check, data, rerun=False):
    """Step 3: fit the checked candidate once on every pre-event row.

    In order, all before the fit: check the record is a check record that
    PASSED, and that the data is the data it was sealed against; if this
    label was already fitted, refuse (rerun=False) or refit and require the
    sealed fit to reproduce (rerun=True). Then fit the candidate with its
    sealed parameters on its split's training and test windows, the row
    before the event being the last; save the model and seal the fit
    record, both read-only under the label's 03_fit/ directory in one move;
    print a short summary. Nothing is forecast and nothing is scored.

    Parameters
    ----------
    check : CheckRecord
        From check_selected_model, or CheckRecord.load.
    data : pd.DataFrame
        The series the design was declared on.
    rerun : bool
        Only matters when the label was already fitted; see above. The
        fitted values are compared in memory only, as a digest.

    Returns
    -------
    FitRecord
        Holding the fitted model: FitRecord.model().

    Raises
    ------
    FitError
        Before any fit, when the check did not pass, the data does not
        match, or the label was already fitted and rerun is False; after
        the refit, when a rerun does not reproduce. Nothing is written.
    """
    handoff = _verify_check(check)
    fp = verify_data(handoff, data, FitError)
    label = handoff["label"]

    target = os.path.join(handoff["output_path"], RECORD_DIR)
    existing = FitRecord.load(target) if os.path.exists(target) else None
    if existing is not None and not rerun:
        _refuse_rerun(existing, label)

    body = fit(handoff, data)
    if existing is not None:
        return _check_rerun(existing, handoff, body, label)

    record_path = os.path.join(target, FitRecord.RECORD_FILE)
    with atomic_directory(target) as tmp:
        body["model"].save(os.path.join(tmp, MODEL_DIR))
        model = {"path": MODEL_PATH, "files": model_files(os.path.join(tmp, MODEL_DIR))}
        record = build_record(_record_fields(check, body, model, fp, utc_now()), handoff)
        with open(os.path.join(tmp, FitRecord.RECORD_FILE), "w", encoding="utf-8") as fh:
            fh.write(record_json(record))
    print(_summary(record, record_path), end="")
    return FitRecord.from_dict(record, path=record_path, _fitted=body["model"])
