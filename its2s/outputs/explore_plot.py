# Description: The figure of explore(), the optional look before a design is
#   declared. Four panels: the series as given; the periodogram of the
#   detrended, variance-stabilised series with the candidate cycles marked;
#   the autocorrelation at lags up to the longest candidate, with the
#   candidate lags marked; and the seasonal decomposition at the longest
#   candidate period, or a note where there is too little data for one.
# Usage: from its2s.outputs.explore_plot import explore_figure
# Dependencies: matplotlib, numpy

# Built with the object-oriented API (matplotlib.figure.Figure), never
# pyplot, so importing this module does not switch the host's backend.
import numpy as np
from matplotlib.figure import Figure

SERIES = "#2166AC"
MARK = "#B2182B"
GREY = "#777777"


def explore_figure(dates, y, outcome_col, transform, numbers, freqs, power, rho,
                   band, stl):
    """Draw explore()'s four panels and return the Figure."""
    fig = Figure(figsize=(10, 9), layout="constrained")
    ax_series, ax_spec, ax_acf, ax_stl = fig.subplots(4, 1)
    tname = transform or "untransformed"

    ax_series.plot(dates, y, color=SERIES, lw=1)
    ax_series.set_title(f"{outcome_col}, the {numbers['n']} rows before the boundary")

    periods = 1.0 / freqs
    ax_spec.plot(periods, power, color=SERIES, lw=1, marker=".", ms=3)
    ax_spec.set_xscale("log")
    for c, v in numbers["cycles"].items():
        ax_spec.axvline(v["m"], color=MARK, ls="--", lw=1)
        ax_spec.annotate(f"{c} (m = {v['m']:.2f})", (v["m"], 1), xycoords=("data", "axes fraction"),
                         xytext=(3, -12), textcoords="offset points", color=MARK, fontsize=8)
    ax_spec.set_xlabel("period (observations)")
    ax_spec.set_ylabel("power")
    ax_spec.set_title(f"Periodogram ({tname}, detrended); one point per frequency bin")

    lags = np.arange(len(rho))
    ax_acf.vlines(lags, 0, rho, color=SERIES, lw=1)
    ax_acf.axhline(0, color=GREY, lw=0.5)
    ax_acf.axhspan(-band, band, color=GREY, alpha=0.15, lw=0)
    for c, v in numbers["cycles"].items():
        if v["acf"] is not None:
            ax_acf.plot([v["acf_lag"]], [v["acf"]], "o", color=MARK, ms=5)
            ax_acf.annotate(f"{c}, lag {v['acf_lag']}", (v["acf_lag"], v["acf"]),
                            xytext=(4, 4), textcoords="offset points", color=MARK, fontsize=8)
    ax_acf.set_xlabel("lag (observations)")
    ax_acf.set_title(f"Autocorrelation ({tname}, detrended); shaded: +/- 1.96 / sqrt(n)")

    s = numbers["stl"]
    if stl is None:
        ax_stl.axis("off")
        note = ("No cycle examined: no decomposition." if s is None
                else f"No decomposition: {s['skipped']}.")
        ax_stl.text(0.5, 0.5, note, ha="center", va="center", transform=ax_stl.transAxes)
    else:
        ax_stl.plot(dates, np.asarray(stl.trend), color=GREY, lw=1, label="trend")
        ax_stl.plot(dates, np.asarray(stl.trend) + np.asarray(stl.seasonal), color=SERIES,
                    lw=1, label="trend + seasonal")
        ax_stl.legend(loc="upper left", fontsize=8, frameon=False)
        ax_stl.set_title(f"STL at the {s['cycle']} period {s['period']} ({tname}); "
                         f"seasonal strength {s['seasonal_strength']:.2f}, shuffled copy "
                         f"{s['shuffled_strength']:.2f}")
    return fig
