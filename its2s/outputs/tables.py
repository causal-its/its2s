# Description: Save excess tables, metrics tables, and ATE summaries to CSV/Excel.
# Usage: from its2s.outputs.tables import save_excess_table, save_metrics_table
# Dependencies: pandas

from dataclasses import asdict
from pathlib import Path

import pandas as pd


def save_excess_table(excess_result, path, fmt="csv"):
    """Save daily and period excess tables.

    Parameters
    ----------
    excess_result : ExcessResult
    path : str or Path
        Output file path. Period excess is saved with '_period' suffix.
    fmt : str
        'csv' or 'xlsx'.
    """
    path = Path(path)

    if fmt == "csv":
        if not excess_result.daily_excess.empty:
            excess_result.daily_excess.to_csv(path, index=False)
        if not excess_result.period_excess.empty:
            period_path = path.with_name(path.stem + "_period" + path.suffix)
            excess_result.period_excess.to_csv(period_path, index=False)
    elif fmt == "xlsx":
        with pd.ExcelWriter(path) as writer:
            if not excess_result.daily_excess.empty:
                excess_result.daily_excess.to_excel(writer, sheet_name="Daily", index=False)
            if not excess_result.period_excess.empty:
                excess_result.period_excess.to_excel(writer, sheet_name="Period", index=False)


def save_metrics_table(metrics_dict, path):
    """Save metrics for multiple evaluation windows.

    Parameters
    ----------
    metrics_dict : dict[str, MetricsResult]
        e.g. {"train": MetricsResult(...), "test": MetricsResult(...)}.
    path : str or Path
    """
    rows = []
    for window, mr in metrics_dict.items():
        row = asdict(mr)
        row["window"] = window
        rows.append(row)
    df = pd.DataFrame(rows)
    cols = ["window"] + [c for c in df.columns if c != "window"]
    df[cols].to_csv(path, index=False)


def save_ate_summary(ate_df, path):
    """Save ATE summary table.

    Parameters
    ----------
    ate_df : pd.DataFrame
    path : str or Path
    """
    ate_df.to_csv(path, index=False)
