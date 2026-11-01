# Description: Step 0's figure, drawn from the resolved design description
#   (or a sealed design record, which holds the same keys). Two panels on one
#   date axis: a timeline of every plotted combination's folds over the test
#   and post-event windows with the check origins marked, and a count per
#   date of how many folds train on it and evaluate on it, which shows how
#   heavily forward chaining weights the earliest data.
# Usage: from its2s.outputs.design_plot import design_figure
# Dependencies: matplotlib, pandas

# Built with the object-oriented API (matplotlib.figure.Figure), never
# pyplot, for the reason given in plots.py: importing this module must not
# switch the host process's backend.
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

# Above this many combinations the figure is unreadable, so only the
# primary split's are drawn unless the design asks for all of them.
MAX_PLOTTED_COMBINATIONS = 6

COLORS = {
    "train": "#BBD3E8",
    "evaluate": "#2166AC",
    "test": "#FDB863",
    "post_event": "#B2182B",
    "origin": "#333333",
    "phase_origin": "#4DAF4A",
    "event": "#B2182B",
}


def plotted_combinations(resolved):
    """The combination keys the figure draws, in declared order."""
    combos = resolved["combinations"]
    if len(combos) <= MAX_PLOTTED_COMBINATIONS or resolved["report_fields"]["plot_all_combinations"]:
        return list(combos)
    primary = [sid for sid, s in resolved["splits"].items() if s["primary"]]
    return [k for k, c in combos.items() if c["split"] in primary]


def _num(date):
    return mdates.date2num(pd.Timestamp(date))


def _bar(ax, y, window, color, height=0.8):
    left = _num(window["start"])
    ax.barh(y, _num(window["end"]) - left, left=left, height=height,
            color=color, edgecolor="none")


def fold_counts(resolved, keys):
    """Per grid date, how many folds of the given combinations train on it
    and how many evaluate on it.

    Returns
    -------
    (pd.DatetimeIndex, np.ndarray, np.ndarray)
    """
    d = resolved["data"]
    grid = pd.date_range(d["first"], d["last"], freq=d["freq"])
    train = np.zeros(len(grid), dtype=int)
    evaluate = np.zeros(len(grid), dtype=int)
    for key in keys:
        for f in resolved["combinations"][key]["folds"]:
            for counts, w in ((train, f["train"]), (evaluate, f["evaluate"])):
                counts += (grid >= pd.Timestamp(w["start"])) & (grid < pd.Timestamp(w["end"]))
    return grid, train, evaluate


def design_figure(resolved):
    """Draw the design: fold timeline above, fold use per date below.

    Parameters
    ----------
    resolved : dict
        Step 0's resolved description, or a sealed design record.

    Returns
    -------
    matplotlib.figure.Figure or None
        None when no combination could be resolved, so there is nothing to
        draw.
    """
    keys = plotted_combinations(resolved)
    if not keys:
        return None
    event = _num(resolved["question"]["event_date"])
    check = resolved["check"]["splits"]

    rows = sum(len(resolved["combinations"][k]["folds"]) + 1 for k in keys)
    fig = Figure(figsize=(11, 2.0 + 0.28 * rows + 2.2))
    top, bottom = fig.subplots(2, 1, sharex=True,
                               gridspec_kw={"height_ratios": [max(rows, 3), 4]})

    labels, y = [], 0
    for key in keys:
        c = resolved["combinations"][key]
        split = resolved["splits"][c["split"]]
        for f in c["folds"]:
            _bar(top, y, f["train"], COLORS["train"])
            _bar(top, y, f["evaluate"], COLORS["evaluate"])
            labels.append(f"{key}  fold {f['k']}")
            y += 1
        _bar(top, y, split["test"], COLORS["test"])
        _bar(top, y, split["post_event"], COLORS["post_event"])
        o = check.get(c["split"])
        if o is not None:
            plain = [_num(t) for t in o["dates"] if t != o["phase_origin"]]
            top.plot(plain, [y] * len(plain), linestyle="none", marker="v",
                     color=COLORS["origin"], markersize=7)
            if o["phase_origin"] is not None:
                top.plot([_num(o["phase_origin"])], [y], linestyle="none", marker="*",
                         color=COLORS["phase_origin"], markersize=12)
        labels.append(f"{key}  test, post-event")
        y += 1
    top.set_yticks(range(len(labels)), labels, fontsize=8)
    top.invert_yaxis()
    top.axvline(event, color=COLORS["event"], linewidth=1, linestyle="--")
    shown = "" if len(keys) == len(resolved["combinations"]) else (
        f" ({len(keys)} of {len(resolved['combinations'])} combinations: primary split only)")
    top.set_title(f"its2s design: {resolved['label']}{shown}", fontsize=11, loc="left")
    top.legend(handles=[
        Patch(color=COLORS["train"], label="fold training"),
        Patch(color=COLORS["evaluate"], label="fold evaluation"),
        Patch(color=COLORS["test"], label="test"),
        Patch(color=COLORS["post_event"], label="post-event"),
        Line2D([], [], linestyle="none", marker="v", color=COLORS["origin"],
               label="check origin"),
        Line2D([], [], linestyle="none", marker="*", color=COLORS["phase_origin"],
               markersize=10, label="event-phase origin"),
    ], fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)

    grid, train, evaluate = fold_counts(resolved, keys)
    x = mdates.date2num(grid)
    bottom.step(x, train, where="post", color=COLORS["train"], linewidth=2,
                label="folds training on the date")
    bottom.step(x, evaluate, where="post", color=COLORS["evaluate"], linewidth=1.5,
                label="folds evaluating on the date")
    bottom.axvline(event, color=COLORS["event"], linewidth=1, linestyle="--")
    trimmed = len(keys) < len(resolved["combinations"])
    bottom.set_ylabel("folds of the primary split's\npairings" if trimmed else "folds")
    bottom.set_ylim(bottom=0)
    bottom.yaxis.set_major_locator(MaxNLocator(integer=True))
    bottom.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    bottom.xaxis_date()
    fig.tight_layout()
    return fig
