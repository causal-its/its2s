# Description: Canonical windows for step 0 (contract P11.04 sections 2.4 and
#   5.5, v4). A split typed in any of three forms (counts, dates, fractions),
#   with an optional train_start, resolves here to training, test and
#   post-event Windows, each half-open at its upper end. THE ONE CONVERSION
#   of an inclusive typed date to a half-open end is _inclusive_to_end; every
#   other module receives Windows. Conditions the analyst must fix come back
#   as Issues, never raised, so step 0 can report them as a group.
# Usage: from its2s.windows import resolve_split, split_form, Window
# Dependencies: pandas

import math
from dataclasses import dataclass, field

import pandas as pd

# The arguments of each split form (contract 5.5). train_start may join any.
FORMS = {
    "counts": ("train_obs", "test_obs"),
    "dates": ("train_end", "test_end"),
    "fractions": ("train_frac", "test_frac"),
}


@dataclass(frozen=True)
class Issue:
    """A condition step 0 reports rather than raises.

    level : "error" (nothing is written) or "warning" (recorded, printed)
    message : what is wrong and, where it helps, what to change
    """

    level: str
    message: str


@dataclass(frozen=True)
class Window:
    """Rows from start (included) to end (excluded) on the series' grid.

    step is the grid's offset, so dates inside the window can be computed;
    it does not take part in equality.
    """

    start: pd.Timestamp
    end: pd.Timestamp
    n: int
    step: pd.DateOffset = field(compare=False, repr=False)

    def date(self, k):
        """The k-th grid date from start (k=0 is start, k=n is end)."""
        return shift(self.start, k, self.step)


@dataclass(frozen=True)
class ResolvedSplit:
    """A split in canonical form, with what was typed kept verbatim.

    unused : the rows before train_start, which the split leaves out by
        choice (contract v4); None when training starts at the first row.
    """

    form: str
    typed: dict
    train: Window
    test: Window
    post_event: Window
    unused: Window | None

    @property
    def canonical(self):
        """What makes two splits the same: both windows' bounds."""
        return (self.train.start, self.train.end, self.test.start, self.test.end)


def shift(date, k, step):
    """date moved k grid steps (k may be negative)."""
    return date if k == 0 else date + k * step


def _inclusive_to_end(date, step):
    """THE ONE CONVERSION: the inclusive last date of a window, as typed by
    the analyst, becomes the half-open end, the next grid date."""
    return shift(date, 1, step)


def _round_half_up(x):
    return math.floor(x + 0.5)


def split_form(typed):
    """The form of a split's arguments; mixing forms, or none, raises.

    API misuse, so it raises at once (spec_P11.04.03 4.4) rather than
    joining the grouped issues.
    """
    given = {k for k, v in typed.items() if v is not None and k != "train_start"}
    forms = [f for f, args in FORMS.items() if given & set(args)]
    unknown = given - {a for args in FORMS.values() for a in args}
    if unknown:
        raise TypeError(f"Unknown split argument(s) {sorted(unknown)}.")
    if len(forms) != 1:
        raise TypeError(
            "A split takes exactly one form: train_obs/test_obs, "
            "train_end/test_end, or train_frac/test_frac"
            + (f"; got {sorted(given)}." if given else "; got none."))
    return forms[0]


def _grid_position(dates, value, what):
    """(position, None) for a grid date, else (None, Issue naming neighbours)."""
    ts = pd.Timestamp(value)
    if ts in dates:
        return dates.get_loc(ts), None
    i = dates.searchsorted(ts)
    if i == 0:
        where = f"before the first date of the series, {dates[0].date()}"
    elif i == len(dates):
        where = f"after the last date of the series, {dates[-1].date()}"
    else:
        where = f"between the grid dates {dates[i - 1].date()} and {dates[i].date()}"
    return None, Issue("error", f"{what} {ts.date()} is not a date of the series: "
                                f"it falls {where}.")


def event_position(dates, event_date):
    """(position, issues) of the event on the grid; off-grid is an error
    naming the grid dates either side (spec_P11.04.03 OQ-2)."""
    pos, issue = _grid_position(dates, event_date, "event_date")
    return pos, [] if issue is None else [issue]


def _n_train(form, typed, dates, start, e, issues):
    """Training rows from position start, or None when they cannot be placed."""
    n_pre = e - start
    a, b = FORMS[form]
    first, second = typed.get(a), typed.get(b)

    if form == "counts":
        for name, v in ((a, first), (b, second)):
            if v is not None and (not float(v).is_integer() or v < 1):
                issues.append(Issue("error", f"{name}={v!r} must be a whole number of "
                                             "at least 1."))
                return None
        if first is not None and second is not None and first + second != n_pre:
            verb = "exceed" if first + second > n_pre else "leave unaccounted for some of"
            issues.append(Issue("error", f"train_obs + test_obs = {first + second} {verb} "
                                         f"the {n_pre} pre-event rows from the training "
                                         "start; give one of them, or make them sum "
                                         f"to {n_pre}."))
            return None
        return int(first) if first is not None else n_pre - int(second)

    if form == "fractions":
        for name, v in ((a, first), (b, second)):
            if v is not None and not 0 < v < 1:
                issues.append(Issue("error", f"{name}={v!r} must lie strictly between "
                                             "0 and 1."))
                return None
        if first is not None and second is not None and abs(first + second - 1) > 1e-9:
            issues.append(Issue("error", f"train_frac + test_frac = {first + second:g}; "
                                         "they must sum to 1, or give only one."))
            return None
        if first is not None:
            return _round_half_up(first * n_pre)
        return n_pre - _round_half_up(second * n_pre)

    # dates: the test window always ends one step before the event, so
    # test_end alone cannot place the cut.
    if first is None:
        issues.append(Issue("error", "The dates form needs train_end; test_end alone "
                                     "cannot place the cut, since the test window "
                                     "always ends one step before the event."))
        return None
    if second is not None:
        pos, issue = _grid_position(dates, second, "test_end")
        if issue is not None:
            issues.append(issue)
            return None
        if pos != e - 1:
            issues.append(Issue("error", f"test_end {pd.Timestamp(second).date()} is not "
                                         "one step before the event; the test window "
                                         f"must end on {dates[e - 1].date()}."))
            return None
    pos, issue = _grid_position(dates, first, "train_end")
    if issue is not None:
        issues.append(issue)
        return None
    end = _inclusive_to_end(dates[pos], dates.freq)
    return dates.searchsorted(end) - start


def resolve_split(form, typed, dates, event_date, H):
    """Resolve one split to canonical Windows.

    Parameters
    ----------
    form : str
        "counts", "dates" or "fractions" (see split_form).
    typed : dict
        The arguments as the analyst typed them, train_start included; kept
        verbatim in the result.
    dates : pd.DatetimeIndex
        The series' complete regular grid, with freq set.
    event_date : date-like
        A date of the grid (checked once per design by event_position).
    H : int
        The post-event horizon, in observations.

    Returns
    -------
    (ResolvedSplit or None, list of Issue)
        None when any error prevents placing the windows.
    """
    step = dates.freq
    if step is None:
        raise ValueError("dates must carry its grid frequency (DatetimeIndex.freq).")
    e = dates.get_loc(pd.Timestamp(event_date))
    issues = []

    start = 0
    if typed.get("train_start") is not None:
        start, issue = _grid_position(dates, typed["train_start"], "train_start")
        if issue is not None:
            return None, [issue]
        if start >= e:
            return None, [Issue("error", "train_start is at or after the event, so the "
                                         "split has no pre-event rows.")]

    n_train = _n_train(form, typed, dates, start, e, issues)
    if n_train is None:
        return None, issues
    n_test = e - start - n_train

    if n_train < 1:
        if start > 0:
            issues.append(Issue("error", f"train_start {dates[start].date()} is at or "
                                         "after the start of the test window."))
        else:
            issues.append(Issue("error", "The training window is empty: the cut falls "
                                         "at or before the first row."))
    if n_test < 1:
        issues.append(Issue("error", "The test window is empty: the cut falls at or "
                                     "after the last pre-event row."))
    n_post = min(H, len(dates) - e)
    if n_post < H:
        issues.append(Issue("error", f"The post-event window holds {n_post} rows, fewer "
                                     f"than H={H}; the series ends on "
                                     f"{dates[-1].date()}."))
    if issues:
        return None, issues

    def window(i, n):
        return Window(dates[i], shift(dates[i], n, step), n, step)

    return ResolvedSplit(
        form=form,
        typed=dict(typed),
        train=window(start, n_train),
        test=window(start + n_train, n_test),
        post_event=window(e, H),
        unused=window(0, start) if start > 0 else None,
    ), []
