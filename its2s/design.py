# Description: Step 0's declaration (contract P11.04 section 5; spec_P11.04.03
#   4.4). Design is the mutable declaration the analyst builds piece by piece;
#   resolve() turns it into a plain, JSON-ready description of what was
#   declared plus every finding of the grouped validation; render_text() is
#   THE ONE rendering, used by preview() here and by declare_design, so what
#   the analyst previews is what is sealed (contract 5.2). Misuse of the API
#   raises at the call; anything that depends on the data or on how the
#   pieces combine is a finding, reported as a group. Nothing here fits a
#   model or imports a model library.
# Usage: from its2s.design import Design, render_text
# Dependencies: pandas

import math
import os
import textwrap
import warnings
from dataclasses import dataclass

import pandas as pd

from .folds import cycles_seen, layout_folds, longest_cycle
from .frequency import SeriesFrequency, resolve_frequency
from .models.families import FAMILIES, arima_period, resolve_space
from .origins import N_ORIGINS_RANGE, place_origins
from .seasonality import CYCLES, _sampling_interval, make_seasonality
from .seeds import derive_seed
from .windows import Window, _grid_position, event_position, resolve_split, split_form

# The figure stays readable up to this many combinations (SN-3).
MAX_PLOTTED_COMBINATIONS = 6

# Reserved name of the whole post-event window (contract 5.10).
FULL_WINDOW = "full_window"

CLOSING_LINE = "Nothing has been written. No model has been fit."


@dataclass(frozen=True)
class Criterion:
    """One entry of the closed criteria menu (contract 6.4, 7.4).

    op : "<", "<=", ">=" or "band" (two-sided, inclusive, threshold a pair)
    """

    id: str
    stage: str              # "screening" | "check"
    asks: str
    statistic: str
    op: str
    default: object
    basis: str              # "principled" | "convention"
    selected_by_default: bool


CRITERIA = {c.id: c for c in (
    Criterion("S1", "screening", 'does it beat "same period last year"?',
              "mean cross-validated MASE", "<", 1.0, "principled", True),
    Criterion("S2", "screening", "is any fold catastrophically worse?",
              "worst fold RMSE / median fold RMSE", "<=", 1.5, "convention", False),
    Criterion("S3", "screening", "is out-of-fold error far above in-sample?",
              "mean CV RMSE / training-fit RMSE", "<=", 2.0, "convention", True),
    Criterion("S4", "screening", "seen the seasonal cycle enough times?",
              "seasonal cycles in the smallest fold", ">=", 2.0, "principled", True),
    Criterion("S5", "screening", "tuned at the distance it will be used?",
              "fold horizon / H", "band", (0.5, 2.0), "convention", False),
    Criterion("C1", "check", 'does it beat "same period last year"?',
              "mean MASE across origins", "<", 1.0, "principled", True),
    Criterion("C2", "check", "did error blow up out of sample?",
              "mean RMSE across origins / step 1 tuning metric", "<=", 1.5,
              "convention", True),
    Criterion("C3", "check", "is any origin catastrophically worse?",
              "worst origin RMSE / median origin RMSE", "<=", 1.5, "convention", True),
)}


@dataclass(frozen=True)
class Finding:
    """One result of the grouped validation.

    level : "error" (the design cannot be sealed) or "warning" (recorded)
    title : the short heading it is reported under
    message : what is wrong and, where it helps, what to change
    """

    level: str
    title: str
    message: str

    def to_dict(self):
        return {"level": self.level, "title": self.title, "message": self.message}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _iso(ts):
    """A timestamp as an ISO date, or date and time when it has one."""
    ts = pd.Timestamp(ts)
    return ts.date().isoformat() if ts == ts.normalize() else ts.isoformat()


def _jsonable(value):
    """A typed argument in a form JSON holds, kept as close to verbatim as can be."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (pd.Timestamp,)) or hasattr(value, "isoformat"):
        return _iso(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _check_id(kind, value):
    if not isinstance(value, str) or not value:
        raise TypeError(f"A {kind} id must be a non-empty string; got {value!r}.")
    if "/" in value:
        raise ValueError(f"{kind} id {value!r} may not contain '/', which separates "
                         "the parts of a candidate id (family/split/cv).")


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fmt_threshold(op, threshold):
    if op == "band":
        lo, hi = threshold
        return f"{lo} to {hi}"
    return f"{op} {threshold}"


def _passes(op, value, threshold):
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == ">=":
        return value >= threshold
    lo, hi = threshold
    return lo <= value <= hi


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------

class Design:
    """The declaration of one analysis, built piece by piece (contract 5.2).

    Mutable while drafting; frozen once declared. preview() shows exactly
    what declare_design would seal, and writes nothing.

    Parameters
    ----------
    data : pd.DataFrame
        One row per observation. Copied here, so a later change to the
        caller's frame cannot change what is declared.
    date_col, outcome_col : str
    covariate_cols : list of str or None
    event_date : date-like
        The first date of the post-event window; a date of the series.
    horizon : int
        H, the post-event window in observations. Required.
    seed : int
        The master seed. Required.
    output_dir : str or path
    series_label : str
        Names this analysis; output_dir/series_label is where it is written.
    """

    def __init__(self, data, date_col, outcome_col, covariate_cols, event_date,
                 horizon, seed, output_dir, series_label):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"data must be a pandas DataFrame; got {type(data).__name__}.")
        for name, value in (("date_col", date_col), ("outcome_col", outcome_col),
                            ("series_label", series_label)):
            if not isinstance(value, str) or not value:
                raise TypeError(f"{name} must be a non-empty string; got {value!r}.")
        if covariate_cols is not None and (
                isinstance(covariate_cols, str)
                or not all(isinstance(c, str) for c in covariate_cols)):
            raise TypeError("covariate_cols must be None or a list of column names.")
        if not isinstance(horizon, int) or isinstance(horizon, bool):
            raise TypeError(f"horizon must be an integer; got {horizon!r}.")
        if horizon < 1:
            raise ValueError(f"horizon must be at least 1; got {horizon}.")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise TypeError(f"seed must be an integer; got {seed!r}.")
        if not isinstance(output_dir, (str, os.PathLike)):
            raise TypeError(f"output_dir must be a path; got {output_dir!r}.")
        try:
            event = pd.Timestamp(event_date)
        except (TypeError, ValueError) as e:
            raise TypeError(f"event_date {event_date!r} is not a date: {e}") from None

        self.data = data.copy(deep=True)
        self.date_col = date_col
        self.outcome_col = outcome_col
        self.covariate_cols = list(covariate_cols or [])
        self.event_date = event
        self.horizon = horizon
        self.seed = seed
        self.output_dir = os.fspath(output_dir)
        self.series_label = series_label

        self._cycles = None                 # None: not declared, annual assumed
        self._splits = {}
        self._cvs = {}
        self._families = {}
        self._criteria = {}
        self._check = {"n_origins": None, "origin_dates": None}
        self._sub_periods = {}
        self._exploration = None
        self._plot_all = False
        self._starter_used = False
        self._frozen = False

    # -- guards -------------------------------------------------------------

    def _mutable(self):
        if self._frozen:
            raise ValueError("This design has been declared and is frozen; build a "
                             "new Design to change it.")

    # -- 5.4 seasonality ----------------------------------------------------

    def set_seasonality(self, cycles=("annual",)):
        """Declare the outcome's seasonal cycles; None asserts there are none."""
        self._mutable()
        if cycles is None:
            self._cycles = ()
            return self
        if isinstance(cycles, str):
            raise TypeError("cycles must be a list of cycle names, or None; "
                            f"got the string {cycles!r}.")
        cycles = list(cycles)
        unknown = [c for c in cycles if c not in CYCLES]
        if unknown:
            raise ValueError(f"Unknown cycle(s) {unknown}; expected a subset of "
                             f"{list(CYCLES)}, or None.")
        if not cycles:
            raise ValueError("cycles=[] is ambiguous; pass cycles=None to assert the "
                             "series is aseasonal.")
        self._cycles = tuple(c for c in CYCLES if c in cycles)
        return self

    # -- 5.5 splits ---------------------------------------------------------

    def add_split(self, split_id, *, train_obs=None, test_obs=None, train_end=None,
                  test_end=None, train_frac=None, test_frac=None, train_start=None,
                  primary=False):
        """Add a split in one of three forms (counts, dates, fractions)."""
        self._mutable()
        _check_id("split", split_id)
        if split_id in self._splits:
            raise ValueError(f"A split {split_id!r} is already declared; remove it "
                             "first to replace it.")
        if not isinstance(primary, bool):
            raise TypeError(f"primary must be True or False; got {primary!r}.")
        typed = {"train_obs": train_obs, "test_obs": test_obs, "train_end": train_end,
                 "test_end": test_end, "train_frac": train_frac,
                 "test_frac": test_frac, "train_start": train_start}
        typed = {k: v for k, v in typed.items() if v is not None}
        form = split_form(typed)
        if primary:
            other = next((s for s, d in self._splits.items() if d["primary"]), None)
            if other is not None:
                raise ValueError(f"Split {other!r} is already primary; exactly one "
                                 "split is primary.")
        self._splits[split_id] = {"form": form, "typed": typed, "primary": primary}
        return self

    def remove_split(self, split_id):
        self._mutable()
        self._remove(self._splits, "split", split_id)
        return self

    # -- 5.6 cross-validation designs ----------------------------------------

    def add_cv(self, cv_id, *, n_folds, fold_eval_obs):
        """Add a cross-validation design: n_folds folds of fold_eval_obs rows."""
        self._mutable()
        _check_id("cv", cv_id)
        if cv_id in self._cvs:
            raise ValueError(f"A cv design {cv_id!r} is already declared; remove it "
                             "first to replace it.")
        for name, value in (("n_folds", n_folds), ("fold_eval_obs", fold_eval_obs)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer; got {value!r}.")
            if value < 1:
                raise ValueError(f"{name} must be at least 1; got {value}.")
        self._cvs[cv_id] = {"n_folds": n_folds, "fold_eval_obs": fold_eval_obs}
        return self

    def remove_cv(self, cv_id):
        self._mutable()
        self._remove(self._cvs, "cv design", cv_id)
        return self

    # -- 5.7 families -------------------------------------------------------

    def add_family(self, name, *, n_trials, fixed=None, bounds=None):
        """Add a model family to screen, with its search budget and shaping."""
        self._mutable()
        if not isinstance(name, str) or not name:
            raise TypeError(f"A family name must be a non-empty string; got {name!r}.")
        if name in self._families:
            raise ValueError(f"Family {name!r} is already declared; remove it first "
                             "to replace it.")
        if not isinstance(n_trials, int) or isinstance(n_trials, bool):
            raise TypeError(f"n_trials must be an integer; got {n_trials!r}.")
        for arg, value in (("fixed", fixed), ("bounds", bounds)):
            if value is not None and not isinstance(value, dict):
                raise TypeError(f"{arg} must be a dict; got {type(value).__name__}.")
        if bounds:
            for dim, pair in bounds.items():
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise TypeError(f"bounds[{dim!r}] must be a (low, high) pair; "
                                    f"got {pair!r}.")
        self._families[name] = {"n_trials": n_trials, "fixed": dict(fixed or {}),
                                "bounds": {k: tuple(v) for k, v in (bounds or {}).items()}}
        return self

    def remove_family(self, name):
        self._mutable()
        self._remove(self._families, "family", name)
        return self

    # -- 5.8 criteria -------------------------------------------------------

    def set_criteria(self, **settings):
        """Select criteria and set thresholds: a value sets it, True selects it
        at its default, None deselects it. Unnamed criteria keep their state;
        repeated calls accumulate."""
        self._mutable()
        unknown = sorted(set(settings) - set(CRITERIA))
        if unknown:
            raise TypeError(f"Unknown criterion name(s) {unknown}; the menu is "
                            f"{list(CRITERIA)}.")
        self._criteria.update(settings)
        return self

    # -- 5.9 check origins --------------------------------------------------

    def set_check(self, n_origins=None, origin_dates=None):
        """Declare step 2's origins: a count (3 to 6) or explicit dates."""
        self._mutable()
        if n_origins is not None and origin_dates is not None:
            raise TypeError("Give n_origins or origin_dates, not both.")
        if n_origins is not None:
            if not isinstance(n_origins, int) or isinstance(n_origins, bool):
                raise TypeError(f"n_origins must be an integer; got {n_origins!r}.")
            lo, hi = N_ORIGINS_RANGE
            if not lo <= n_origins <= hi:
                raise ValueError(f"n_origins must be {lo} to {hi}; got {n_origins}.")
        if origin_dates is not None:
            if isinstance(origin_dates, str):
                raise TypeError("origin_dates must be a list of dates.")
            origin_dates = list(origin_dates)
        self._check = {"n_origins": n_origins, "origin_dates": origin_dates}
        return self

    # -- 5.10 sub-periods ---------------------------------------------------

    def add_sub_period(self, name, start, end):
        """Name a stretch of the post-event window, inclusive start and end."""
        self._mutable()
        if not isinstance(name, str) or not name:
            raise TypeError(f"A sub-period name must be a non-empty string; got {name!r}.")
        if name in self._sub_periods:
            raise ValueError(f"A sub-period {name!r} is already declared.")
        self._sub_periods[name] = {"start": start, "end": end}
        return self

    # -- explore and the figure (spec OQ-6, OQ-7) -----------------------------

    def set_exploration(self, exploration):
        """Record the boundary of an explore() run (contract 4)."""
        self._mutable()
        if not hasattr(exploration, "boundary"):
            raise TypeError("set_exploration takes the result of its2s.explore().")
        self._exploration = exploration
        return self

    def set_plot(self, all_combinations=False):
        """Plot every combination even past six (SN-3)."""
        self._mutable()
        if not isinstance(all_combinations, bool):
            raise TypeError("all_combinations must be True or False.")
        self._plot_all = all_combinations
        return self

    # -- preview ------------------------------------------------------------

    def preview(self):
        """Print exactly what declare_design would seal; write nothing.

        Returns
        -------
        list of Finding
            Every error and warning; nothing is raised for them.
        """
        resolved, findings = resolve(self)
        print(render_text(resolved))
        return findings

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _remove(store, kind, key):
        if key not in store:
            raise ValueError(f"No {kind} {key!r} is declared; declared: {list(store)}.")
        del store[key]


# ---------------------------------------------------------------------------
# Resolution and the grouped validation (spec 4.4)
# ---------------------------------------------------------------------------

def _window(w, seasonality):
    """A Window as the record holds it: bounds, inclusive last date, count,
    cycles of the longest declared cycle seen."""
    return {
        "start": _iso(w.start),
        "end": _iso(w.end),
        "last": _iso(w.date(w.n - 1)) if w.n else None,
        "n": int(w.n),
        "cycles": None if seasonality is None else cycles_seen(w, seasonality),
    }


def _resolve_criteria(settings):
    out, findings = {}, []
    for cid, c in CRITERIA.items():
        entry = {"stage": c.stage, "asks": c.asks, "statistic": c.statistic,
                 "op": c.op, "basis": c.basis, "selected": c.selected_by_default,
                 "selected_by_default": c.selected_by_default,
                 "threshold": list(c.default) if c.op == "band" else c.default,
                 "threshold_defaulted": True}
        if cid in settings:
            value = settings[cid]
            if value is None:
                entry["selected"] = False
            elif value is True:
                entry["selected"] = True
            elif c.op == "band":
                ok = (isinstance(value, (tuple, list)) and len(value) == 2
                      and all(_is_number(v) for v in value)
                      and 0 < value[0] < value[1])
                if ok:
                    entry.update(selected=True, threshold=[value[0], value[1]],
                                 threshold_defaulted=False)
                else:
                    findings.append(Finding("error", "criteria", f"{cid}={value!r}: {cid} "
                                            "is two-sided and takes a pair (lo, hi) "
                                            "with 0 < lo < hi, True or None."))
            elif _is_number(value) and math.isfinite(value) and value > 0:
                entry.update(selected=True, threshold=value, threshold_defaulted=False)
            else:
                findings.append(Finding("error", "criteria", f"{cid}={value!r}: a "
                                        "threshold must be a positive number; True "
                                        "selects the default, None deselects."))
        out[cid] = entry
    return out, findings


def _resolve_data(design, findings):
    """The grid and the event; None for whatever cannot be placed."""
    df = design.data
    cols = [design.date_col, design.outcome_col, *design.covariate_cols]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        findings.append(Finding("error", "data", f"Column(s) {missing} are not in the "
                                f"data; it has {list(df.columns)}."))
        return None, None
    for c in [design.outcome_col, *design.covariate_cols]:
        if not pd.api.types.is_numeric_dtype(df[c]):
            findings.append(Finding("error", "data", f"Column {c!r} is not numeric "
                                    f"(dtype {df[c].dtype})."))
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(df[design.date_col]))
    except (TypeError, ValueError) as e:
        findings.append(Finding("error", "data", f"Column {design.date_col!r} does not "
                                f"parse as dates: {e}"))
        return None, None
    try:
        freq = resolve_frequency(dates)
    except ValueError as e:
        findings.append(Finding("error", "data", str(e)))
        return None, None
    dates = pd.DatetimeIndex(dates, freq=freq.alias)
    e, issues = event_position(dates, design.event_date)
    findings.extend(Finding(i.level, "data", i.message) for i in issues)
    if e is None:
        return dates, None
    if len(dates) - e < design.horizon:
        findings.append(Finding("error", "data", f"The series holds {len(dates) - e} "
                                f"rows from the event, fewer than H={design.horizon}; "
                                f"it ends on {_iso(dates[-1])}."))
        return dates, None
    return dates, e


def _resolve_seasonality(design, dates, findings):
    cycles = ("annual",) if design._cycles is None else design._cycles
    try:
        seasonality = make_seasonality(cycles, SeriesFrequency.from_alias(dates.freqstr),
                                       declared=design._cycles is not None)
    except ValueError as e:
        findings.append(Finding("error", "seasonality", str(e)))
        return None
    unit, n = _sampling_interval(seasonality.freq)
    if unit == "days" and n == 1 and "annual" not in seasonality.cycles:
        findings.append(Finding("warning", "seasonality", "Daily data with no annual "
                                "cycle declared: every fit will treat the series as "
                                "free of an annual pattern, which daily health "
                                "series rarely are."))
    return seasonality


def _describe_typed(typed):
    return ", ".join(f"{k}={v!r}" for k, v in typed.items())


def _resolve_splits(design, dates, findings):
    resolved = {}
    if not design._splits:
        findings.append(Finding("error", "splits", "No split is declared; add one "
                                "with add_split."))
        return resolved
    if len(design._splits) > 1 and not any(s["primary"] for s in design._splits.values()):
        findings.append(Finding("error", "splits", f"{len(design._splits)} splits are "
                                "declared and none is primary; mark exactly one with "
                                "primary=True."))
    for sid, s in design._splits.items():
        split, issues = resolve_split(s["form"], s["typed"], dates, design.event_date,
                                      design.horizon)
        findings.extend(Finding(i.level, f"split {sid}", i.message) for i in issues)
        if split is not None:
            resolved[sid] = split

    seen = {}
    for sid, split in resolved.items():
        twin = seen.get(split.canonical)
        if twin is None:
            seen[split.canonical] = sid
            continue
        n_pre = split.train.n + split.test.n
        findings.append(Finding("error", "duplicate split", (
            f"{sid} was declared as {_describe_typed(design._splits[sid]['typed'])}. "
            f"Of {n_pre} pre-event observations that resolves to {split.train.n} "
            f"training and {split.test.n} test, which is the same canonical window "
            f"pair as {twin} (declared by {design._splits[twin]['form']}, "
            f"{_describe_typed(design._splits[twin]['typed'])}). Two names for one "
            "split would double the compute and print the same candidate twice "
            "under different ids. Remove one, or change its boundaries.")))

    boundary = getattr(design._exploration, "boundary", None)
    if boundary is not None:
        for sid, split in resolved.items():
            if split.train.end < pd.Timestamp(boundary):
                findings.append(Finding("warning", "exploration", (
                    f"{sid}'s training window ends on {_iso(split.train.date(split.train.n - 1))}, "
                    f"before the exploration boundary {_iso(boundary)}: explore read "
                    "rows of its test window, so they may have informed this "
                    "declaration.")))
    return resolved


def _check_complete(design, dates, splits, findings):
    """Outcome and covariates complete in every declared window (OQ-3)."""
    values = design.data.set_index(dates)
    for col in [design.outcome_col, *design.covariate_cols]:
        gaps = []
        for sid, split in splits.items():
            for name, w in (("training", split.train), ("test", split.test),
                            ("post-event", split.post_event)):
                block = values.loc[(dates >= w.start) & (dates < w.end), col]
                missing = block.index[block.isna()]
                if len(missing):
                    first = ", ".join(_iso(d) for d in missing[:3])
                    more = f" and {len(missing) - 3} more" if len(missing) > 3 else ""
                    gaps.append(f"{sid} {name}: {len(missing)} ({first}{more})")
        if gaps:
            findings.append(Finding("error", "data", (
                f"Column {col!r} has missing values inside declared windows: "
                + "; ".join(gaps) + ". Fill or impute them before declaring, and "
                "say how in the analysis.")))


def _resolve_combinations(design, splits, seasonality, criteria, findings):
    combos = {}
    if not design._cvs:
        findings.append(Finding("error", "cross-validation", "No cross-validation "
                                "design is declared; add one with add_cv."))
        return combos
    s4, s5, s2 = criteria["S4"], criteria["S5"], criteria["S2"]
    cycle = longest_cycle(seasonality) if seasonality is not None else None
    for sid, split in splits.items():
        for cid, cv in design._cvs.items():
            folds, issues = layout_folds(split.train, cv["n_folds"], cv["fold_eval_obs"])
            findings.extend(Finding(i.level, f"cv {cid} on {sid}", i.message)
                            for i in issues)
            if not folds:
                continue
            smallest = folds[0].train
            cyc = cycles_seen(smallest, seasonality) if seasonality is not None else None
            s5_value = cv["fold_eval_obs"] / design.horizon
            combos[f"{sid}/{cid}"] = {
                "split": sid, "cv": cid, "folds": folds,
                "smallest_cycles": cyc,
                "S4": {"value": cyc,
                       "verdict": _verdict(s4, cyc)},
                "S5": {"value": s5_value, "verdict": _verdict(s5, s5_value)},
            }
            if s4["selected"] and cyc is not None and not _passes(">=", cyc, s4["threshold"]):
                findings.append(Finding("warning", "thin folds", (
                    f"{cid} ({cv['n_folds']} folds x {cv['fold_eval_obs']} obs) on {sid} "
                    f"leaves its smallest fold trained on {smallest.n} of "
                    f"{split.train.n} observations, {_iso(smallest.start)} to "
                    f"{_iso(smallest.date(smallest.n - 1))}, which is {cyc:.2f} cycles of "
                    f"the declared {cycle} cycle. S4 is evaluated on the smallest fold "
                    f"and would FAIL for every candidate on every combination using "
                    f"{cid}. Either drop {cid}, reduce its folds or its evaluation "
                    "length, or keep it knowingly.")))
            if s5["selected"] and not _passes("band", s5_value, s5["threshold"]):
                lo, hi = s5["threshold"]
                findings.append(Finding("warning", "fold horizon", (
                    f"{cid} evaluates folds {cv['fold_eval_obs']} steps ahead against "
                    f"H={design.horizon}, a ratio of {s5_value:.2f}, outside S5's band "
                    f"{lo:g} to {hi:g}: the models would be tuned for a forecast "
                    "distance they will not be used at.")))
    if s2["selected"]:
        for cid, cv in design._cvs.items():
            if cv["n_folds"] < 3:
                findings.append(Finding("warning", "fold stability", (
                    f"S2 is selected but {cid} has {cv['n_folds']} folds; S2 is not "
                    "computable below three and will report not-applicable on every "
                    f"combination using {cid}.")))
    if len(combos) > MAX_PLOTTED_COMBINATIONS and not design._plot_all:
        findings.append(Finding("warning", "figure", (
            f"{len(combos)} combinations are more than the figure shows readably "
            f"({MAX_PLOTTED_COMBINATIONS}); it plots the primary split's "
            "combinations only. Call set_plot(all_combinations=True) to plot all.")))
    return combos


def _verdict(entry, value):
    if value is None:
        return "not applicable"
    if not entry["selected"]:
        return "not selected"
    return "PASS" if _passes(entry["op"], value, entry["threshold"]) else "FAIL"


def _resolve_families(design, seasonality, combos, findings):
    families = {}
    if not design._families:
        findings.append(Finding("error", "families", "No model family is declared; "
                                "add one with add_family."))
        return families
    for name, f in design._families.items():
        title = f"family {name}"
        spec = FAMILIES.get(name)
        if spec is None:
            findings.append(Finding("error", title, f"Unknown model family {name!r}. "
                                    f"Known families: {list(FAMILIES)}."))
            continue
        if spec.status == "withheld":
            findings.append(Finding("error", title, (
                f"{name} is withheld at this version: {spec.withheld_reason} "
                f"(GitHub issue {spec.withheld_issue}). It returns when that is fixed.")))
            continue
        if f["n_trials"] < 0 or f["n_trials"] == 1:
            findings.append(Finding("error", title, (
                f"n_trials={f['n_trials']}: give 0 to screen the default "
                "configuration with no search, or 2 or more to search. 1 is refused "
                "as almost certainly a mistake.")))
        if seasonality is None:
            continue
        try:
            space = resolve_space(spec, seasonality, f["fixed"], f["bounds"])
        except ValueError as e:
            findings.append(Finding("error", title, str(e)))
            continue
        entry = {"n_trials": f["n_trials"], "space": space, "m": None}
        if name == "arima":
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    m = arima_period(seasonality)
            except ValueError as e:
                findings.append(Finding("error", title, str(e)))
                continue
            for w in caught:
                findings.append(Finding("warning", title, str(w.message)))
            entry["m"] = m
            short = {key: c["folds"][0].train.n for key, c in combos.items()
                     if m > 1 and c["folds"][0].train.n < 2 * m}
            if short:
                where = ", ".join(f"{k} ({n} rows)" for k, n in short.items())
                findings.append(Finding("error", title, (
                    f"A seasonal arima at m={m} needs a smallest fold of at least "
                    f"2m = {2 * m} training rows (one cycle is spent by seasonal "
                    f"differencing, one more to estimate); too short on {where}. "
                    "Shorten the folds, or drop arima or those cv designs.")))
        families[name] = entry
    return families


def _resolve_origins(design, splits, seasonality, findings):
    out = {}
    for sid, split in splits.items():
        origins, issues = place_origins(split.test, design.horizon, seasonality,
                                        design.event_date,
                                        n_origins=design._check["n_origins"],
                                        origin_dates=design._check["origin_dates"])
        findings.extend(Finding(i.level, f"check origins on {sid}", i.message)
                        for i in issues)
        if origins is not None:
            out[sid] = origins
    return out


def _resolve_sub_periods(design, dates, e, findings):
    """The post-event window as full_window, then each declared sub-period."""
    step = dates.freq
    post = Window(dates[e], dates[e] + design.horizon * step, design.horizon, step)
    out = {FULL_WINDOW: post}
    for name, sp in design._sub_periods.items():
        title = f"sub-period {name}"
        if name == FULL_WINDOW:
            findings.append(Finding("error", title, f"{FULL_WINDOW!r} is reserved for "
                                    "the whole post-event window, which is always "
                                    "reported; choose another name."))
            continue
        positions = []
        for what in ("start", "end"):
            pos, issue = _grid_position(dates, sp[what], f"{name} {what}")
            if issue is not None:
                findings.append(Finding("error", title, issue.message))
            positions.append(pos)
        if None in positions:
            continue
        a, b = positions
        if a > b:
            findings.append(Finding("error", title, f"starts on {_iso(dates[a])}, after "
                                    f"it ends on {_iso(dates[b])}."))
            continue
        if a < e or b >= e + design.horizon:
            findings.append(Finding("error", title, (
                f"{_iso(dates[a])} to {_iso(dates[b])} is not inside the post-event "
                f"window {_iso(post.start)} to {_iso(post.date(post.n - 1))}.")))
            continue
        out[name] = Window(dates[a], dates[b] + step, b - a + 1, step)
    return out


def resolve(design):
    """Resolve a Design to the plain description step 0 renders and seals.

    Returns
    -------
    (dict, list of Finding)
        The dict holds only JSON values, so render_text can work from a
        written record alone. Pieces that could not be resolved are absent;
        every reason is a Finding, and the dict's "issues" lists them all.
    """
    findings = []
    criteria, crit_findings = _resolve_criteria(design._criteria)
    findings.extend(crit_findings)

    dates, e = _resolve_data(design, findings)
    seasonality = _resolve_seasonality(design, dates, findings) if dates is not None else None
    splits, combos, origins, subs = {}, {}, {}, {}
    if e is not None:
        splits = _resolve_splits(design, dates, findings)
        _check_complete(design, dates, splits, findings)
        combos = _resolve_combinations(design, splits, seasonality, criteria, findings)
        if seasonality is not None:
            origins = _resolve_origins(design, splits, seasonality, findings)
        subs = _resolve_sub_periods(design, dates, e, findings)
    families = _resolve_families(design, seasonality, combos, findings)

    primary = next((s for s, d in design._splits.items() if d["primary"]), None)
    if primary is None and len(design._splits) == 1:
        primary = next(iter(design._splits))

    exploration = None
    if design._exploration is not None:
        x = design._exploration
        exploration = {"boundary": _iso(x.boundary),
                       "rows_read": getattr(x, "rows_read", None)}

    candidates = {}
    for name, fam in families.items():
        for key, combo in combos.items():
            cand = f"{name}/{key}"
            candidates[cand] = {
                "family": name, "split": combo["split"], "cv": combo["cv"],
                "n_trials": fam["n_trials"],
                "seed": derive_seed(design.seed, cand),
                "fits": max(fam["n_trials"], 1) * len(combo["folds"]) + 1,
            }

    sw = seasonality
    resolved = {
        "label": design.series_label,
        "output_path": os.path.join(design.output_dir, design.series_label),
        "data": {
            "date_col": design.date_col, "outcome_col": design.outcome_col,
            "covariate_cols": list(design.covariate_cols),
            "n_rows": len(design.data),
            "first": _iso(dates[0]) if dates is not None else None,
            "last": _iso(dates[-1]) if dates is not None else None,
            "freq": dates.freqstr if dates is not None and dates.freq is not None else None,
        },
        "question": {"event_date": _iso(design.event_date), "H": design.horizon},
        "seed": design.seed,
        "starter_used": design._starter_used,
        "exploration": exploration,
        "seasonality": None if sw is None else {
            "cycles": list(sw.cycles), "periods": dict(sw.periods),
            "declared": sw.declared},
        "splits": {
            sid: {
                "primary": sid == primary,
                "form": design._splits[sid]["form"],
                "typed": _jsonable(design._splits[sid]["typed"]),
                "train": _window(s.train, sw),
                "test": _window(s.test, sw),
                "post_event": _window(s.post_event, None),
                "unused": None if s.unused is None else _window(s.unused, None),
            } for sid, s in splits.items()},
        "cvs": {cid: dict(cv) for cid, cv in design._cvs.items()},
        "combinations": {
            key: {
                "split": c["split"], "cv": c["cv"],
                "folds": [{"k": f.k, "train": _window(f.train, sw),
                           "evaluate": _window(f.evaluate, None)} for f in c["folds"]],
                "S4": c["S4"], "S5": c["S5"],
            } for key, c in combos.items()},
        "families": {
            name: {
                "n_trials": f["n_trials"],
                "space": {"sample": {k: list(v) for k, v in f["space"].sample.items()},
                          "fixed": dict(f["space"].fixed)},
                "flags": list(f["space"].flags),
                "m": f["m"],
            } for name, f in families.items()},
        "candidates": candidates,
        "criteria": criteria,
        "check": {
            "n_origins": design._check["n_origins"],
            "origin_dates": _jsonable(design._check["origin_dates"]),
            "defaulted": design._check == {"n_origins": None, "origin_dates": None},
            "splits": {
                sid: {"dates": [_iso(d) for d in o.dates],
                      "placed_by": o.placed_by,
                      "phase_origin": None if o.phase_origin is None else _iso(o.phase_origin),
                      "max_gap": o.max_gap,
                      "test_cycles": cycles_seen(splits[sid].test, sw)}
                for sid, o in origins.items()},
        },
        "sub_periods": {name: _window(w, None) for name, w in subs.items()},
        "seeds": {"bootstrap": derive_seed(design.seed, "bootstrap"),
                  **{c: v["seed"] for c, v in candidates.items()}},
        "sweep_fits": sum(c["fits"] for c in candidates.values()),
        "plot_all_combinations": design._plot_all,
        "issues": [f.to_dict() for f in findings],
    }
    return resolved, findings


# ---------------------------------------------------------------------------
# THE rendering (contract 5.2, 5.12)
# ---------------------------------------------------------------------------

def _wrap(text, indent=2):
    pad = " " * indent
    return textwrap.fill(text, width=78, initial_indent=pad, subsequent_indent=pad)


def _range(w):
    return f"{w['start']} to {w['last']}"


def _cycles(w, cycle):
    return "" if w["cycles"] is None else f"  {w['cycles']:.2f} {cycle} cycles"


def _count(n, word):
    return f"{n} {word}" + ("" if n == 1 else "s")


def render_text(resolved):
    """The design as the analyst reads it, from the resolved description alone.

    Used by preview and by declare_design, so the two cannot differ.
    """
    r = resolved
    sw = r["seasonality"]
    cycle = None
    if sw and sw["cycles"]:
        cycle = "annual" if "annual" in sw["cycles"] else sw["cycles"][0]
    lines = []
    add = lines.append

    d, q = r["data"], r["question"]
    add(f"its2s design: {r['label']}")
    series = f"{d['outcome_col']} by {d['date_col']}"
    if d["first"] is not None:
        series += f", {d['first']} to {d['last']}, {d['n_rows']} rows"
    if d["freq"] is not None:
        series += f", frequency {d['freq']}"
    add(f"  series       {series}")
    if d["covariate_cols"]:
        add(f"  covariates   {', '.join(d['covariate_cols'])}")
    add(f"  event        {q['event_date']}   H = {q['H']} observations")
    add(f"  seed         {r['seed']}")
    add(f"  output       {r['output_path']}")
    add(f"  starter      {'used' if r['starter_used'] else 'not used'}")
    x = r["exploration"]
    if x is None:
        add("  explore      not run")
    else:
        rows = "" if x["rows_read"] is None else f", {x['rows_read']} rows read"
        add(f"  explore      boundary {x['boundary']}{rows}")

    add("")
    if sw is None:
        add("SEASONALITY  not resolved")
    else:
        how = "declared" if sw["declared"] else "DEFAULT, not declared"
        add(f"SEASONALITY  ({how})")
        if not sw["cycles"]:
            add("  none: asserted aseasonal (S1 benchmarks against a non-seasonal "
                "naive forecast; S4 not applicable)")
        for c in sw["cycles"]:
            add(f"  {c:<8} m = {sw['periods'][c]:.2f} observations")

    add("")
    add(f"SPLITS  ({len(r['splits'])} resolved)")
    for sid, s in r["splits"].items():
        tag = "  (primary)" if s["primary"] else ""
        add(f"  {sid}{tag}   typed: {_describe_typed(s['typed'])}")
        for name, key, cyc in (("training", "train", cycle), ("test", "test", cycle),
                               ("post-event", "post_event", None)):
            w = s[key]
            add(f"    {name:<11} {_range(w)}  {w['n']:>4} obs"
                + (_cycles(w, cyc) if cyc else ""))
        if s["unused"] is not None:
            u = s["unused"]
            add(f"    {'unused':<11} {_range(u)}  {u['n']:>4} obs  (before train_start, "
                "by this split's choice)")

    add("")
    add(f"CROSS-VALIDATION  ({len(r['cvs'])} declared)")
    for cid, cv in r["cvs"].items():
        add(f"  {cid}  {cv['n_folds']} folds x {cv['fold_eval_obs']} obs")

    crit = r["criteria"]
    add("")
    add(f"COMBINATIONS  ({len(r['combinations'])})")
    for key, c in r["combinations"].items():
        s4, s5 = c["S4"], c["S5"]
        s4_text = ("S4 not applicable (no cycle declared)" if s4["value"] is None else
                   f"S4 smallest fold {s4['value']:.2f} cycles "
                   f"{_fmt_threshold('>=', crit['S4']['threshold'])}: {s4['verdict']}")
        s5_text = (f"S5 fold horizon / H = {s5['value']:.2f} "
                   f"{_fmt_threshold('band', crit['S5']['threshold'])}: {s5['verdict']}")
        add(f"  {key}")
        add(f"    {s4_text}")
        add(f"    {s5_text}")
        for f in c["folds"]:
            t, ev = f["train"], f["evaluate"]
            cyc = "" if t["cycles"] is None else f" {t['cycles']:.2f} cycles"
            add(f"    fold {f['k']}  train {_range(t)} {t['n']:>4} obs{cyc};"
                f" evaluate {_range(ev)} {ev['n']:>3} obs")

    chk = r["check"]
    add("")
    if chk["origin_dates"] is not None:
        how = f"origin_dates={chk['origin_dates']}"
    elif chk["n_origins"] is not None:
        how = f"n_origins={chk['n_origins']}"
    else:
        how = "n_origins=4, default"
    add(f"CHECK ORIGINS  ({how})")
    for sid, o in chk["splits"].items():
        add(f"  {sid}  {', '.join(o['dates'])}")
        if o["phase_origin"] is not None:
            add(f"    event-phase origin {o['phase_origin']}")
        elif cycle is not None:
            add("    no origin at the event's phase")
        if o["max_gap"] is not None:
            add(f"    maximum seasonal gap {o['max_gap']:.2f} of the {cycle} cycle; "
                f"test window {o['test_cycles']:.2f} cycles")

    fams = r["families"]
    add("")
    add(f"FAMILIES  ({len(fams)} resolved)")
    for name, f in fams.items():
        m = "" if f["m"] is None else f"   m = {f['m']}"
        add(f"  {name:<12} n_trials={f['n_trials']}   "
            f"{_count(len(f['space']['sample']), 'dimension')} searched{m}")
        for flag in f["flags"]:
            add(f"    NON-DEFAULT  {flag}")
    budgets = {name: f["n_trials"] for name, f in fams.items()}
    if len(set(budgets.values())) > 1:
        add("  UNEQUAL BUDGETS: " + ", ".join(f"{k} {v}" for k, v in budgets.items())
            + "; a bigger budget is a reason a family scores better that has "
            "nothing to do with the family.")

    add("")
    add("CRITERIA")
    for cid, c in crit.items():
        state = "selected" if c["selected"] else "not selected"
        set_by = "default" if c["threshold_defaulted"] else "SET"
        sel_by = ("" if c["selected"] == c["selected_by_default"]
                  else ", selection CHANGED from default")
        add(f"  {cid}  {c['statistic']:<48} {_fmt_threshold(c['op'], c['threshold']):<12}"
            f" {c['basis']:<10} {state} (threshold {set_by}{sel_by})")

    add("")
    add(f"SWEEP  {_count(len(r['candidates']), 'candidate')}, "
        f"{_count(r['sweep_fits'], 'fit')} (per candidate: max(n_trials, 1) x folds, "
        "plus one in-sample fit)")

    add("")
    add("SEEDS  (derived from the master seed and each id)")
    width = max((len(k) for k in r["seeds"]), default=0)
    for k, v in r["seeds"].items():
        add(f"  {k:<{width}}  {v}")

    add("")
    add("SUB-PERIODS")
    for name, w in r["sub_periods"].items():
        add(f"  {name:<12} {_range(w)}  {w['n']} obs")

    issues = r["issues"]
    n_err = sum(i["level"] == "error" for i in issues)
    n_warn = len(issues) - n_err
    if issues:
        add("")
        add(f"ISSUES  {_count(n_err, 'error')}, {_count(n_warn, 'warning')}")
        for i in sorted(issues, key=lambda i: i["level"] != "error"):
            add("")
            add(f"{i['level'].upper():<8} {i['title']}")
            add(_wrap(i["message"]))
    if n_err:
        add("")
        add(CLOSING_LINE)
    return "\n".join(lines)
