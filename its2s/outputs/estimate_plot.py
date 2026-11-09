# Description: Step 4's figure, 04-counterfactual.png, drawn from the
#   counterfactual table (the content of 04-counterfactual.csv) and the
#   estimate record. The observed series across every window; the training
#   and test windows shaded and named; the event line; the counterfactual
#   forecast and its interval ribbon over the post-event window only. No
#   in-sample fit line is drawn, since its closeness to the data would read
#   as evidence for the forecast, which it is not.
# Usage: from its2s.outputs.estimate_plot import counterfactual_figure, resolve_style
# Dependencies: matplotlib, pandas

# Built with the object-oriented API (matplotlib.figure.Figure), never
# pyplot, for the reason given in plots.py. Such a figure is never registered
# with pyplot, so nothing stays open once it is written and dropped.
import copy

import matplotlib.dates as mdates
import pandas as pd
from matplotlib.figure import Figure

from ..estimate_tables import bound_columns, interval_words

PLOT_FILE = "04-counterfactual.png"

STYLE_COLORS = ("observed", "counterfactual", "event", "training", "test", "post_event")
STYLE_FONTS = ("title", "axis_label", "tick", "legend")
STYLE_KEYS = ("colors", "fonts", "figsize", "dpi")

# The old path's defaults where it had one (its "intervention" colour is the
# event line here); training and test shading are new, so stated here.
DEFAULT_STYLE = {
    "colors": {"observed": "#333333", "counterfactual": "#B2182B", "event": "#4DAF4A",
               "training": "#D1E5F0", "test": "#E0E0E0", "post_event": "#FEE08B"},
    "fonts": {"title": 22, "axis_label": 20, "tick": 18, "legend": 18},
    "figsize": [14, 6],
    "dpi": 150,
}


def resolve_style(style):
    """The defaults with the analyst's style laid over them, key by key.

    style is assumed already checked (estimate._check_style). Returns a new
    dict, JSON-ready, with figsize as a list.
    """
    resolved = copy.deepcopy(DEFAULT_STYLE)
    for key, value in (style or {}).items():
        if key in ("colors", "fonts"):
            resolved[key].update(value)
        elif key == "figsize":
            resolved[key] = list(value)
        else:
            resolved[key] = value
    return resolved


def _span(ax, dates, table, window, color, label):
    """Shade a window from its first row up to the next window's first row,
    so adjacent windows meet without a gap."""
    where = (table["window"] == window).to_numpy().nonzero()[0]
    if len(where) == 0:
        return
    last = where[-1] + 1 if where[-1] + 1 < len(dates) else where[-1]
    ax.axvspan(dates.iloc[where[0]], dates.iloc[last], color=color, alpha=0.35,
               linewidth=0, label=label)


def counterfactual_figure(record, table, resolved, outcome):
    """Draw the counterfactual figure.

    Parameters
    ----------
    record : dict
        The estimate record: label, candidate, interval and stamps.
    table : pd.DataFrame
        The counterfactual table.
    resolved : dict
        The resolved style.
    outcome : str
        The outcome column's name, for the y axis.

    Returns
    -------
    matplotlib.figure.Figure
    """
    colors, fonts = resolved["colors"], resolved["fonts"]
    interval = record["interval"]
    lo, hi = bound_columns("counterfactual", interval)
    dates = pd.to_datetime(table["date"])
    post = (table["window"] == "post_event").to_numpy()

    fig = Figure(figsize=tuple(resolved["figsize"]), dpi=resolved["dpi"])
    ax = fig.subplots()
    _span(ax, dates, table, "training", colors["training"], "training")
    _span(ax, dates, table, "test", colors["test"], "test (used in the final fit)")
    _span(ax, dates, table, "post_event", colors["post_event"], "post-event")
    ax.plot(dates, table["observed"], color=colors["observed"], linewidth=0.8,
            label="observed")
    ax.fill_between(dates[post], table.loc[post, lo], table.loc[post, hi],
                    color=colors["counterfactual"], alpha=0.2,
                    label=f"{interval_words(interval)}, not validated")
    ax.plot(dates[post], table.loc[post, "forecast"], color=colors["counterfactual"],
            linewidth=1.4, label="counterfactual forecast")
    event = dates[post].iloc[0]
    ax.axvline(event, color=colors["event"], linestyle="--", linewidth=1.2, label="event")

    ax.set_xlabel("Date", fontsize=fonts["axis_label"])
    ax.set_ylabel(outcome, fontsize=fonts["axis_label"])
    ax.tick_params(axis="both", labelsize=fonts["tick"])
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    # Below the axes, so it never covers the series.
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
              fontsize=fonts["legend"], frameon=False)
    title = (f"its2s estimate: {record['label']}, {record['candidate_id']}, "
             f"{interval_words(interval)}")
    if record["stamps"]["split_chosen_after_evidence"]:
        title += "\nSPLIT CHOSEN AFTER EVIDENCE: not the primary split"
    ax.set_title(title, fontsize=fonts["title"], loc="left")
    fig.tight_layout()
    return fig
