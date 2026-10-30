# Description: The cross-validation layout of step 0 (contract P11.04 sections
#   3.4 and 5.6, V-9) and the seasonal coverage of a window (SN-12). Folds are
#   anchored at the END of the training window with no gap, so the last
#   evaluation slice ends at the last training row and the smallest fold is
#   the first. Coverage is counted in calendar terms, never from m. Nothing
#   here fits; step 1 fits on these folds.
# Usage: from its2s.folds import layout_folds, cycles_seen
# Dependencies: pandas

from dataclasses import dataclass

from .seasonality import _CYCLE_DAYS, _sampling_interval
from .windows import Issue, Window


@dataclass(frozen=True)
class Fold:
    """Fold k (1 is the smallest): fit on train, score on evaluate."""

    k: int
    train: Window
    evaluate: Window


def layout_folds(training, n_folds, fold_eval_obs):
    """Lay n_folds folds of fold_eval_obs rows each, end-anchored, no gap.

    Fold k's evaluation slice ends fold_eval_obs * (n_folds - k) rows before
    the training window's end, and its training portion runs from the
    training window's start to the slice's start.

    Returns
    -------
    (list of Fold, list of Issue)
        No folds, and an error, when the first slice would start at or before
        the training window's start.
    """
    if n_folds < 1 or fold_eval_obs < 1:
        raise ValueError("n_folds and fold_eval_obs must each be at least 1.")
    smallest = training.n - n_folds * fold_eval_obs
    if smallest < 1:
        return [], [Issue("error", f"{n_folds} folds of {fold_eval_obs} rows need more "
                                   f"than {n_folds * fold_eval_obs} training rows; the "
                                   f"training window holds {training.n}, so the first "
                                   "evaluation slice crosses its start.")]
    folds = []
    for k in range(1, n_folds + 1):
        n_fit = smallest + (k - 1) * fold_eval_obs
        cut = training.date(n_fit)
        folds.append(Fold(
            k=k,
            train=Window(training.start, cut, n_fit, training.step),
            evaluate=Window(cut, training.date(n_fit + fold_eval_obs),
                            fold_eval_obs, training.step),
        ))
    return folds, []


def longest_cycle(seasonality):
    """The longest declared cycle, or None when none is declared."""
    if not seasonality.cycles:
        return None
    return "annual" if "annual" in seasonality.cycles else seasonality.cycles[0]


def cycles_seen(window, seasonality):
    """Turns of the longest declared cycle a window spans, in calendar terms.

    Days / cycle days on day- and week-based sampling (77 weekly rows are
    539 / 365.25 = 1.48); whole months / 12 on month-based sampling, so 24
    months is exactly 2.0. None when no cycle is declared.
    """
    cycle = longest_cycle(seasonality)
    if cycle is None:
        return None
    unit, _ = _sampling_interval(seasonality.freq)
    if unit == "days":
        return (window.end - window.start).days / _CYCLE_DAYS[cycle]
    months = ((window.end.year - window.start.year) * 12
              + window.end.month - window.start.month)
    return months / 12
