# Description: Step 4's two tables, the content of 04-counterfactual.csv and
#   04-excess_table.csv. The excess table is rendered from the estimate record
#   alone; the counterfactual table adds the observed series before the
#   event, read from the data, with each row's window. Every interval column
#   names its type and level in full (excess_prediction95_lo), so the two
#   interval types never differ by one letter and each file stays a plain
#   table whose header line says what its bounds are.
# Usage: from its2s.estimate_tables import excess_table, counterfactual_table
# Dependencies: numpy, pandas

from decimal import Decimal

import numpy as np
import pandas as pd

from .design_record import _iso

COUNTERFACTUAL_FILE = "04-counterfactual.csv"
EXCESS_TABLE_FILE = "04-excess_table.csv"
WINDOWS = ("before_training", "training", "test", "post_event")


def level_percent(level):
    """The level in percent as written, from its decimal string: 0.975 ->
    '97.5', 0.57 -> '57'. Never level * 100 in floating point, which turns
    0.57 into 56.99999..."""
    d = Decimal(repr(float(level))) * 100
    return format(d.normalize(), "f")


def interval_words(interval):
    """'95% prediction interval', the interval as the report and figure name it."""
    return f"{level_percent(interval['level'])}% {interval['type']} interval"


def bound_columns(quantity, interval):
    """The lower and upper bound column names of one quantity, for example
    ('excess_prediction95_lo', 'excess_prediction95_hi')."""
    tag = f"{interval['type']}{level_percent(interval['level']).replace('.', '_')}"
    return f"{quantity}_{tag}_lo", f"{quantity}_{tag}_hi"


def _with_bounds(quantity, interval, point=True):
    lo, hi = bound_columns(quantity, interval)
    return ([quantity] if point else []) + [lo, hi]


def _rename_bounds(row, interval):
    """A record row's quantity_lo / quantity_hi keys under their full names."""
    out = {}
    for key, value in row.items():
        for q in ("counterfactual", "excess_pct", "excess"):
            if key in (f"{q}_lo", f"{q}_hi"):
                key = bound_columns(q, interval)[key.endswith("_hi")]
                break
        out[key] = value
    return out


def excess_columns(interval):
    return (["period", "first", "last", "n", "observed", "forecast"]
            + _with_bounds("excess", interval) + _with_bounds("excess_pct", interval)
            + ["point_outside_interval", "median_path_minus_point"])


def counterfactual_columns(interval):
    return (["date", "window", "observed", "forecast"]
            + _with_bounds("counterfactual", interval, point=False)
            + _with_bounds("excess", interval) + _with_bounds("excess_pct", interval))


def excess_table(record):
    """One row per period, from the record's periods, as 04-excess_table.csv.

    Parameters
    ----------
    record : dict
        The estimate record.

    Returns
    -------
    pd.DataFrame
    """
    interval = record["interval"]
    rows = [_rename_bounds({"period": p["name"], **p}, interval) for p in record["periods"]]
    return pd.DataFrame(rows, columns=excess_columns(interval))


def _windows(dates, candidate):
    """Each date's window, by the candidate's half-open windows; a date in
    none of them is an error, not a guess."""
    bounds = {w: (pd.Timestamp(candidate[w]["start"]), pd.Timestamp(candidate[w]["end"]))
              for w in ("train", "test", "post_event")}
    conditions = [dates < bounds["train"][0]]
    conditions += [(dates >= s) & (dates < e) for s, e in bounds.values()]
    labels = np.select(conditions, list(WINDOWS), default="")
    if (labels == "").any():
        first = dates[labels == ""].iloc[0]
        raise ValueError(f"The date {_iso(first)} falls in no window of the fit.")
    return labels


def counterfactual_table(record, handoff, data):
    """One row per data row from the first date in the data to the end of the
    post-event window, as 04-counterfactual.csv.

    Before the event only the date, window and observed value are filled:
    step 4 forecasts the post-event window alone, so the forecast, interval,
    excess and percent columns there are empty.

    Parameters
    ----------
    record : dict
        The estimate record; supplies every post-event column.
    handoff : dict
        The fit record's handoff; supplies the column names and windows.
    data : pd.DataFrame
        The series, already verified against the design's fingerprint.

    Returns
    -------
    pd.DataFrame
    """
    interval = record["interval"]
    d = handoff["data"]
    candidate = handoff["candidate"]
    dates = pd.to_datetime(data[d["date_col"]]).reset_index(drop=True)
    keep = (dates < pd.Timestamp(candidate["post_event"]["end"])).to_numpy()
    dates = dates[keep].reset_index(drop=True)
    table = pd.DataFrame({
        "date": [_iso(t) for t in dates],
        "window": _windows(dates, candidate),
        "observed": data[d["outcome_col"]].to_numpy(dtype=float)[keep],
    })
    steps = pd.DataFrame([_rename_bounds(s, interval) for s in record["per_step"]])
    steps = steps.drop(columns=["observed", "excess_pct_null"])
    table = table.merge(steps, on="date", how="left", validate="one_to_one")
    if table.loc[table["window"] == "post_event", "forecast"].isna().any():
        raise ValueError("A post-event row of the data has no estimate in the record.")
    return table[counterfactual_columns(interval)]


def to_csv_text(table):
    """The CSV text as written: no index, empty cells for missing values."""
    return table.to_csv(index=False, lineterminator="\n")
