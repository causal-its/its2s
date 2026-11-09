# Description: Step 4's computation and its public call. estimate() forecasts
#   the post-event window from the step 3 model, runs the moving block
#   bootstrap and forms per-step and per-period excess with intervals; it
#   reads the fit record's HANDOFF section, the data, the model and its
#   confirmed fitted values, nothing else, and touches no file.
#   estimate_excess is the public step: it checks every argument and refuses
#   before any bootstrap, confirms the saved model reproduces the fitted
#   values step 3 sealed, then estimates, seals the estimate record and
#   writes it with its report, two CSVs and the figure under
#   04_estimate/<run_label>/.
# Usage: est = its2s.estimate_excess(fit, data=df, run_label="main")
# Dependencies: numpy, pandas, matplotlib

import numbers
import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from . import __version__
from .bootstrap.mbb import (ReplicateColumns, check_block_length, replicate,
                            replication_seeds, residual_pool)
from .candidates_record import CandidatesRecord
from .check_record import CheckRecord
from .design import _suggest_label
from .design_record import DesignRecord, _iso
from .final_fit import fitted_values_digest, pre_event_rows
from .fit_record import SCHEMA_VERSION as FIT_SCHEMA_VERSION
from .fit_record import FitRecord
from .fitting import Columns, predict_family, rows_in, span_of, utc_now, verify_data
from .metrics.excess import excess_from_paths
from .models.base import _versions
from .models.families import get_family, startup_rows_of
from .estimate_record import (RECORD_DIR, RECORD_FILE, REPORT_FILE, EstimateRecord,
                              build_record, caveats)
from .estimate_tables import (COUNTERFACTUAL_FILE, EXCESS_TABLE_FILE, counterfactual_table,
                              excess_table, to_csv_text)
from .outputs.estimate_plot import (PLOT_FILE, STYLE_COLORS, STYLE_FONTS, STYLE_KEYS,
                                    counterfactual_figure, resolve_style)
from .records import atomic_directory, record_json, stamped_on

FULL_WINDOW = "full_window"
INTERVAL_TYPES = ("prediction", "confidence")
INTERVAL_METHOD = "quantile"
# block_length="auto" until the interval rebuild: a fixed 14 rows, named so
# the record states the rule as well as the integer.
AUTO_BLOCK_LENGTH = 14
AUTO_BLOCK_RULE = "fixed-14-observations"
GIVEN_BLOCK_RULE = "as given"
# Rules of thumb, stated as such: fewer replications beyond a bound, or
# fewer whole blocks in the pool, than these, and the step warns.
MIN_BEYOND_BOUND = 5
MIN_BLOCKS = 10
NOISE_POOL = "in-sample residuals, start-up rows excluded, centred"
NOTHING = "Nothing has been estimated."


class EstimateError(ValueError):
    """estimate_excess stopped before estimating anything. A ValueError,
    like FitError."""


class EstimateWarning(UserWarning):
    """A step 4 result is computed but one of its settings or its
    replications makes an interval less trustworthy than it reads."""


# ---------------------------------------------------------------------------
# The computation
# ---------------------------------------------------------------------------

def fit_rows(handoff, data):
    """The rows step 3 fit on: train.start <= date < event, which must span
    exactly the sealed fit span. A mismatch is an error, not a correction."""
    rows = pre_event_rows(handoff, data)
    span = span_of(rows, handoff["data"]["date_col"])
    if span != dict(handoff["fit_span"]):
        raise EstimateError(
            f"The pre-event rows of the data span {span['first']} .. {span['last']} "
            f"({span['n']} rows); the model was fit on {handoff['fit_span']['first']} .. "
            f"{handoff['fit_span']['last']} ({handoff['fit_span']['n']} rows). {NOTHING}")
    return rows


def target_rows(handoff, data):
    """The post-event rows to forecast; their count must be the sealed one."""
    return rows_in(data, handoff["data"]["date_col"], handoff["candidate"]["post_event"],
                   "post-event window")


def _check_origin(handoff, fit_df, target_df):
    """The forecast starts at the event, from the grid date before it."""
    date_col = handoff["data"]["date_col"]
    origin = pd.Timestamp(handoff["forecast_origin"])
    event = pd.Timestamp(handoff["question"]["event_date"])
    step = pd.tseries.frequencies.to_offset(handoff["data"]["freq"])
    last_fit = pd.Timestamp(fit_df[date_col].iloc[-1])
    first_target = pd.Timestamp(target_df[date_col].iloc[0])
    if origin != last_fit or origin + step != event or first_target != event:
        raise EstimateError(
            f"The forecast origin {_iso(origin)} is not the grid date before the event "
            f"{_iso(event)} (last fit row {_iso(last_fit)}, first forecast row "
            f"{_iso(first_target)}). {NOTHING}")


def _period_masks(handoff, dates, names):
    """One mask over the post-event rows per period, in the order of names."""
    masks = []
    for name in names:
        w = handoff["sub_periods"][name]
        mask = np.asarray((dates >= pd.Timestamp(w["start"]))
                          & (dates < pd.Timestamp(w["end"])))
        if int(mask.sum()) != w["n"]:
            raise EstimateError(
                f"The period {name} holds {int(mask.sum())} post-event rows in the data "
                f"and {w['n']} in the sealed design. {NOTHING}")
        masks.append(mask)
    return masks


def _run_replications(model, fit_df, target_df, fitted, cols, settings, seed, k):
    """Every replication in order: their outcomes and the residual pool."""
    residuals = fit_df[cols.target].to_numpy(dtype=float) - fitted
    pool = residual_pool(residuals, k)
    check_block_length(len(pool), settings["block_length"])
    noise = settings["interval_type"] == "prediction"
    seeds = replication_seeds(seed, settings["n_replications"])
    outcomes = [replicate(r, model, fit_df, target_df, fitted, pool,
                          settings["block_length"], seeds[r], cols,
                          startup_rows=k, noise=noise)
                for r in range(settings["n_replications"])]
    return outcomes, pool


def _failures(outcomes):
    """Failed replications counted by kind; exceptions by exception type."""
    exceptions = Counter(o.exception for o in outcomes if o.failure == "raised")
    kinds = Counter(o.failure for o in outcomes if not o.ok)
    return {"exception": dict(sorted(exceptions.items())),
            "non_finite": kinds["non_finite"], "wrong_length": kinds["wrong_length"]}


def _warn_on_failures(n_failed, n, level):
    beyond = n * (1 - level) / 2
    if n_failed > beyond:
        warnings.warn(
            f"{n_failed} of {n} bootstrap replications failed, more than the "
            f"{beyond:g} expected beyond each bound at level {level:g}. The failed "
            "replications could all have belonged to one tail, so the bounds are not "
            "trustworthy.", EstimateWarning, stacklevel=3)


def _null_counts(rows):
    return dict(sorted(Counter(r["excess_pct_null"] for r in rows
                               if r["excess_pct_null"] is not None).items()))


def _step_rows(dates, steps):
    return [{"date": _iso(d), "observed": s["observed"], "forecast": s["expected"],
             "counterfactual_lo": s["counterfactual_lo"],
             "counterfactual_hi": s["counterfactual_hi"],
             "excess": s["excess"], "excess_lo": s["excess_lo"], "excess_hi": s["excess_hi"],
             "excess_pct": s["excess_pct"], "excess_pct_lo": s["excess_pct_lo"],
             "excess_pct_hi": s["excess_pct_hi"], "excess_pct_null": s["excess_pct_null"]}
            for d, s in zip(dates, steps)]


def _period_rows(dates, names, masks, periods):
    """One row per period; first and last are the inclusive dates of its rows."""
    out = []
    for name, mask, p in zip(names, masks, periods):
        inside = dates[mask]
        out.append({"name": name, "n": p["n_obs"], "first": _iso(inside.iloc[0]),
                    "last": _iso(inside.iloc[-1]),
                    "observed": p["total_observed"], "forecast": p["total_expected"],
                    "excess": p["total_excess"], "excess_lo": p["excess_lo"],
                    "excess_hi": p["excess_hi"], "excess_pct": p["excess_pct"],
                    "excess_pct_lo": p["excess_pct_lo"], "excess_pct_hi": p["excess_pct_hi"],
                    "excess_pct_null": p["excess_pct_null"],
                    "point_outside_interval": p["point_outside_interval"],
                    "median_path_minus_point": p["median_path_minus_point"]})
    return out


def estimate(handoff, data, model, fitted, settings):
    """Forecast the post-event window, bootstrap it, and form the excess.

    Parameters
    ----------
    handoff : dict
        The fit record's HANDOFF section; nothing else of any record is read.
    data : pd.DataFrame
    model : BaseModel
        The step 3 model, fitted or loaded.
    fitted : np.ndarray
        Its in-sample fitted values on the fit rows, already confirmed
        against the sealed digest; residuals are observed minus these.
    settings : dict
        interval_type, level, n_replications, block_length (an integer),
        block_rule, block_argument and periods (the period names, full_window
        first).

    Returns
    -------
    dict
        The record body: interval, bootstrap, per_step, periods and
        percent_null.

    Raises
    ------
    EstimateError
        When the data's rows do not match the sealed spans, the forecast
        origin is not the grid date before the event, the point forecast is
        not finite, or every replication fails.
    """
    cols = Columns(handoff["data"])
    rcols = ReplicateColumns(cols.y, cols.date, cols.cov)
    fit_df = fit_rows(handoff, data)
    target_df = target_rows(handoff, data)
    _check_origin(handoff, fit_df, target_df)
    fitted = np.asarray(fitted, dtype=float)
    if len(fitted) != len(fit_df):
        raise EstimateError(f"{len(fitted)} fitted values for {len(fit_df)} fit rows. "
                            + NOTHING)
    try:
        point = predict_family(model, target_df, cols, Counter())
    except ValueError as e:
        raise EstimateError(f"The point forecast failed: {e}. {NOTHING}") from None

    k = model.startup_rows()
    n = settings["n_replications"]
    outcomes, pool = _run_replications(model, fit_df, target_df, fitted, rcols,
                                       settings, handoff["bootstrap_seed"], k)
    paths = [o.path for o in outcomes if o.ok]
    failures = _failures(outcomes)
    if not paths:
        raise EstimateError(
            f"All {n} bootstrap replications failed ({failures}); no interval can be "
            f"formed. {NOTHING}")
    _warn_on_failures(n - len(paths), n, settings["level"])
    refit_warnings = Counter()
    for o in outcomes:
        refit_warnings.update(o.warnings)

    dates = pd.to_datetime(target_df[cols.date]).reset_index(drop=True)
    names = settings["periods"]
    masks = _period_masks(handoff, dates, names)
    steps, periods = excess_from_paths(
        target_df[cols.y].to_numpy(dtype=float), point, np.column_stack(paths),
        settings["level"], masks, percent="within_path")
    per_step = _step_rows(dates, steps)
    per_period = _period_rows(dates, names, masks, periods)

    noise = None
    if settings["interval_type"] == "prediction":
        noise = {"pool": NOISE_POOL, "n_pool": len(pool),
                 "block_length": settings["block_length"], "propagation": "none"}
    return {
        "interval": {"type": settings["interval_type"], "level": settings["level"],
                     "method": INTERVAL_METHOD, "validated": False},
        "bootstrap": {
            "n_requested": n, "n_succeeded": len(paths), "failures": failures,
            "refit_warnings": dict(sorted(refit_warnings.items())),
            "block_length": {"argument": settings["block_argument"],
                             "rule": settings["block_rule"],
                             "value": settings["block_length"]},
            "start_up_rows": k, "n_pool": len(pool), "pool_centred": True,
            "noise": noise, "seed": handoff["bootstrap_seed"],
        },
        "per_step": per_step,
        "periods": per_period,
        "percent_null": {"per_step": _null_counts(per_step),
                         "periods": _null_counts(per_period)},
    }


# ---------------------------------------------------------------------------
# Checks made before any bootstrap, in the order estimate_excess runs them
# ---------------------------------------------------------------------------

def _verify_fit(fit):
    if not isinstance(fit, FitRecord):
        raise TypeError(
            "estimate_excess takes the FitRecord that fit_final_model returned, or "
            f"one read with FitRecord.load; got {type(fit).__name__}.")
    if fit.schema_version != FIT_SCHEMA_VERSION:
        raise EstimateError(
            f"This fit record is schema version {fit.schema_version}; this package "
            f"estimates from version {FIT_SCHEMA_VERSION}. Declare the design again "
            f"under a new label. {NOTHING}")
    if fit.path is None:
        raise EstimateError(
            "This fit record was not read from disk, so there is no label directory "
            f"to write the estimate under; read it with FitRecord.load. {NOTHING}")


def _label_dir(fit):
    """The directory holding 03_fit/, from where the fit record was read; not
    the handoff's output_path, which may be relative to another directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(fit.path)))


def _run_dir(fit, run_label):
    """The run's directory; refused when run_label is not one plain directory
    name, or names a run that exists."""
    seps = {os.sep, "/"} | ({os.altsep} if os.altsep else set())
    if (not isinstance(run_label, str) or not run_label or run_label in (".", "..")
            or any(s in run_label for s in seps)):
        raise EstimateError(
            f"run_label must be a non-empty string usable as one directory name (no "
            f"path separator, not '.' or '..'); got {run_label!r}. {NOTHING}")
    run_dir = os.path.join(_label_dir(fit), RECORD_DIR, run_label)
    if os.path.exists(run_dir):
        on = stamped_on(os.path.join(run_dir, RECORD_FILE), "estimated_at", run_dir)
        raise EstimateError(
            f"this estimate cannot be sealed: the run_label {run_label!r} already exists "
            f"at {run_dir}, estimated on {on}. One run_label is one sealed estimate; "
            f"estimate again under a new run_label, for example "
            f"{_suggest_label(run_label)!r}. {NOTHING}")
    return run_dir


def _refuse_withheld(family):
    spec = get_family(family)
    if spec.status == "withheld":
        raise EstimateError(
            f"{family} is not supported in this release: {spec.withheld_reason} "
            f"(GitHub issue {spec.withheld_issue}). {NOTHING}")


def _resolve_periods(handoff, sub_periods):
    """full_window, then the chosen sub-periods in declaration order."""
    declared = [n for n in handoff["sub_periods"] if n != FULL_WINDOW]
    if sub_periods is None:
        chosen = set(declared)
    else:
        if isinstance(sub_periods, str) or not isinstance(sub_periods, (list, tuple)):
            raise EstimateError(
                "sub_periods must be None (every declared sub-period), [] (full_window "
                f"only) or a list of declared names; got {sub_periods!r}. {NOTHING}")
        unknown = [n for n in sub_periods if n != FULL_WINDOW and n not in declared]
        if unknown:
            raise EstimateError(
                f"sub_periods names {unknown}, not declared at step 0. Declared: "
                f"{declared or 'none'}; full_window is always reported. {NOTHING}")
        chosen = set(sub_periods)
    return [FULL_WINDOW] + [n for n in declared if n in chosen]


def _check_interval_type(interval_type):
    if interval_type not in INTERVAL_TYPES:
        raise EstimateError(
            f"interval_type must be 'prediction' or 'confidence'; got "
            f"{interval_type!r}. {NOTHING}")


def _is_number(value):
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _check_level(level):
    if not _is_number(level) or not 0 < level < 1:
        raise EstimateError(
            f"level must be a number strictly between 0 and 1; got {level!r}. {NOTHING}")


def _check_n_replications(n):
    if not isinstance(n, numbers.Integral) or isinstance(n, bool) or n < 1:
        raise EstimateError(
            f"n_replications must be a positive integer; got {n!r}. {NOTHING}")


def _warn_on_level(level, n):
    beyond = n * (1 - level) / 2
    if beyond < MIN_BEYOND_BOUND:
        warnings.warn(
            f"With {n} replications at level {level:g}, about {beyond:g} lie beyond each "
            f"bound, fewer than {MIN_BEYOND_BOUND}; each bound rests on that many "
            "replications.", EstimateWarning, stacklevel=3)


def _resolve_block_length(block_length, fit):
    """(value, rule) for block_length, checked against the residual pool."""
    if block_length == "auto":
        value, rule = AUTO_BLOCK_LENGTH, AUTO_BLOCK_RULE
    elif (isinstance(block_length, numbers.Integral) and not isinstance(block_length, bool)
          and block_length >= 1):
        value, rule = int(block_length), GIVEN_BLOCK_RULE
    else:
        raise EstimateError(
            f"block_length must be 'auto' or a positive integer; got {block_length!r}. "
            + NOTHING)
    s = fit.selected_order
    k = startup_rows_of(None if s is None else s["order"],
                        None if s is None else s["seasonal_order"])
    n_pool = fit.span["n"] - k
    if value >= n_pool:
        raise EstimateError(
            f"block_length {value} must be shorter than the residual pool: {n_pool} "
            f"rows ({fit.span['n']} fit rows less {k} start-up rows). {NOTHING}")
    if n_pool // value < MIN_BLOCKS:
        warnings.warn(
            f"The residual pool of {n_pool} rows holds {n_pool // value} whole blocks of "
            f"{value}, fewer than {MIN_BLOCKS}; so few blocks give the resampling few "
            "distinct arrangements.", EstimateWarning, stacklevel=3)
    return value, rule


def _positive(value, integer=False):
    if integer:
        return (isinstance(value, numbers.Integral) and not isinstance(value, bool)
                and value > 0)
    return _is_number(value) and value > 0


def _check_style(style):
    if style is None:
        return
    if not isinstance(style, dict):
        raise EstimateError(f"style must be None or a dict; got {style!r}. {NOTHING}")
    problems = [f"unknown key {k!r}" for k in style if k not in STYLE_KEYS]
    for key, names, ok, what in (("colors", STYLE_COLORS, lambda v: isinstance(v, str),
                                  "a colour string"),
                                 ("fonts", STYLE_FONTS, _positive, "a positive size")):
        if key not in style:
            continue
        value = style[key]
        if not isinstance(value, dict):
            problems.append(f"{key} must be a dict")
            continue
        problems += [f"unknown {key} key {k!r}" for k in value if k not in names]
        problems += [f"{key}[{k!r}] must be {what}" for k, v in value.items()
                     if k in names and not ok(v)]
    if "figsize" in style:
        f = style["figsize"]
        if not (isinstance(f, (list, tuple)) and len(f) == 2 and all(map(_positive, f))):
            problems.append("figsize must be two positive numbers")
    if "dpi" in style and not _positive(style["dpi"], integer=True):
        problems.append("dpi must be a positive integer")
    if problems:
        raise EstimateError(
            f"style: {'; '.join(problems)}. Keys: {list(STYLE_KEYS)}; colors: "
            f"{list(STYLE_COLORS)}; fonts: {list(STYLE_FONTS)}. {NOTHING}")


def _confirmed_model(fit, handoff, data):
    """The model, and its fitted values recomputed and checked against the
    digest step 3 sealed.

    A model read from disk holds no in-sample fit, and a reloaded model is
    not guaranteed to reproduce one; the sealed digest is the reference, so
    a failure to reproduce is refused, never used. Fitted values always come
    from the model itself, never from what a fit left in memory.
    """
    model = fit.model()
    cols = Columns(handoff["data"])
    rows = fit_rows(handoff, data)
    try:
        fitted = np.asarray(model.in_sample(rows, target_col=cols.y, date_col=cols.date,
                                            covariate_cols=cols.cov), dtype=float)
    except ValueError as e:
        raise EstimateError(f"The model's in-sample fit failed: {e}. {NOTHING}") from None
    if fitted_values_digest(fitted) != dict(fit.fitted_values):
        installed = _versions(getattr(model, "_libraries", ()))
        raise EstimateError(
            "The model does not reproduce the in-sample fitted values sealed at step 3, "
            "so it is not the model that was fit. A changed library is the likely cause. "
            f"Fit with: {dict(fit.library_versions)}. Installed: {installed}. {NOTHING}")
    return model, fitted


def _under(label_dir, record_class):
    return os.path.join(label_dir, record_class.RECORD_DIR, record_class.RECORD_FILE)


def _provenance(fit, label_dir):
    """The seal and path of every record the estimate descends from: the fit
    and the check directly, the design and candidates from the check's own
    provenance once its seal is verified."""
    check = CheckRecord.load(label_dir)
    if check.seal["digest"] != fit.provenance["check"]["seal"]:
        raise EstimateError(
            f"The check record at {check.path} is not the one this fit descends from: "
            "its seal differs from the one the fit record holds. " + NOTHING)
    cp = check.provenance
    return {
        "design": {"seal": cp["design"]["seal"], "path": _under(label_dir, DesignRecord)},
        "candidates": {"seal": cp["candidates"]["seal"],
                       "path": _under(label_dir, CandidatesRecord)},
        "check": {"seal": check.seal["digest"], "path": check.path},
        "fit": {"seal": fit.seal["digest"], "path": os.path.abspath(fit.path)},
    }


def _write_run(run_dir, record, handoff, data):
    """Write the record, report, both CSVs and the figure, read-only, under
    a new run directory in one move, so a failure part-way leaves nothing
    behind. Returns the report text."""
    report = EstimateRecord.from_dict(record).report()
    table = counterfactual_table(record, handoff, data)
    fig = counterfactual_figure(record, table, record["presentation"]["resolved"],
                                handoff["data"]["outcome_col"])
    texts = ((RECORD_FILE, record_json(record)), (REPORT_FILE, report),
             (COUNTERFACTUAL_FILE, to_csv_text(table)),
             (EXCESS_TABLE_FILE, to_csv_text(excess_table(record))))
    with atomic_directory(run_dir) as tmp:
        for name, text in texts:
            with open(os.path.join(tmp, name), "w", encoding="utf-8") as fh:
                fh.write(text)
        fig.savefig(os.path.join(tmp, PLOT_FILE), dpi=fig.dpi, bbox_inches="tight")
    return report


# ---------------------------------------------------------------------------
# The public step
# ---------------------------------------------------------------------------

def estimate_excess(fit, data, *, sub_periods=None, interval_type="prediction",
                    level=0.95, n_replications=1000, block_length="auto",
                    run_label="main", style=None):
    """Step 4: estimate the excess over the post-event window, with intervals.

    In order, all before any bootstrap: check the fit record; refuse a
    run_label that is not one directory name or that already exists under
    this label; refuse a withheld family; check the data is the data the
    design was sealed against; check sub_periods, interval_type, level,
    n_replications, block_length and style; load the model (its files
    checked against their sealed hashes) and confirm it reproduces the
    fitted values step 3 sealed. Then forecast the post-event window, run
    the moving block bootstrap and form per-step and per-period excess and
    percent excess with intervals of the requested type; seal the estimate
    record; write it, its report, 04-counterfactual.csv, 04-excess_table.csv
    and 04-counterfactual.png read-only under 04_estimate/<run_label>/; print
    the report.

    Every interval is labelled not validated: its coverage has not been
    measured and it is known to undercover.

    Expected runtime: one refit of the model per replication, so about
    n_replications times as long as step 3's fit.

    Parameters
    ----------
    fit : FitRecord
        From fit_final_model, or FitRecord.load.
    data : pd.DataFrame
        The series the design was declared on; it supplies the post-event
        observations.
    sub_periods : None, or list of str
        None: full_window and every sub-period declared at step 0. []:
        full_window only. A list: full_window and those named, each
        declared at step 0.
    interval_type : str
        "prediction" (the default) or "confidence".
    level : float
        Strictly between 0 and 1.
    n_replications : int
    block_length : "auto" or int
        In rows. "auto" resolves to 14 (rule fixed-14-observations) until
        the interval rebuild.
    run_label : str
        Names this run's directory under 04_estimate/.
    style : dict, optional
        Figure presentation: colors, fonts, figsize, dpi.

    Returns
    -------
    EstimateRecord

    Raises
    ------
    TypeError
        When fit is not a FitRecord.
    EstimateError
        Before any bootstrap, for any check above; after it, when every
        replication failed. Nothing is written.
    """
    _verify_fit(fit)
    run_dir = _run_dir(fit, run_label)
    handoff = fit.to_dict()["handoff"]
    family = handoff["candidate"]["family"]
    _refuse_withheld(family)
    fp = verify_data(handoff, data, EstimateError, NOTHING)
    periods = _resolve_periods(handoff, sub_periods)
    _check_interval_type(interval_type)
    _check_level(level)
    _check_n_replications(n_replications)
    _warn_on_level(level, n_replications)
    block_value, block_rule = _resolve_block_length(block_length, fit)
    _check_style(style)
    model, fitted = _confirmed_model(fit, handoff, data)
    label_dir = _label_dir(fit)
    provenance = _provenance(fit, label_dir)

    estimated_at = utc_now()
    settings = {"interval_type": interval_type, "level": float(level),
                "n_replications": int(n_replications), "block_length": block_value,
                "block_rule": block_rule, "block_argument": block_length
                if block_length == "auto" else int(block_length), "periods": periods}
    body = estimate(handoff, data, model, fitted, settings)

    record = build_record({
        "package_version": __version__,
        "estimated_at": estimated_at,
        "sealed_at": utc_now(),
        "label": handoff["label"],
        "run_label": run_label,
        "candidate_id": handoff["candidate"]["id"],
        "family": family,
        **body,
        "caveats": caveats(interval_type, family),
        "presentation": {"style": style, "resolved": resolve_style(style)},
        "stamps": fit.to_dict()["stamps"],
        "provenance": provenance,
        "data_fingerprint": fp,
    })
    report = _write_run(run_dir, record, handoff, data)
    print(report, end="")
    return EstimateRecord.from_dict(record, path=os.path.join(run_dir, RECORD_FILE))
