# Description: Save excess tables, metrics tables, ATE summaries, and residual
#   diagnostics to CSV/Excel.
# Usage: from its2s.outputs.tables import save_excess_table, save_metrics_table
# Dependencies: numpy, pandas

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd


def save_excess_table(excess_result, path, fmt="csv"):
    """Save per-observation and period excess tables.

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
        if not excess_result.obs_excess.empty:
            excess_result.obs_excess.to_csv(path, index=False)
        if not excess_result.period_excess.empty:
            period_path = path.with_name(path.stem + "_period" + path.suffix)
            excess_result.period_excess.to_csv(period_path, index=False)
    elif fmt == "xlsx":
        with pd.ExcelWriter(path) as writer:
            if not excess_result.obs_excess.empty:
                excess_result.obs_excess.to_excel(writer, sheet_name="Obs", index=False)
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


_DIAGNOSTICS_COLUMNS = [
    "model_name", "section", "statistic", "lag", "lag_units",
    "value", "status", "note", "freq_alias", "m", "n",
]


def _diag_rows(diag):
    """Flatten a DiagnosticsResult into tidy long-format rows (GH #64)."""
    params = diag.params
    max_lag = params.get("max_lag")
    n = params.get("n")
    rows = []

    def add(section, statistic, value, lag=None, lag_units=None,
            status=None, note=None):
        if status is None:
            if value is not None and np.isfinite(value):
                status = "ok"
            else:
                status = "nan"
        rows.append({
            "model_name": diag.model_metadata.get("model_name"),
            "section": section,
            "statistic": statistic,
            "lag": lag,
            "lag_units": lag_units,
            "value": value if status == "ok" else np.nan,
            "status": status,
            "note": note if status != "ok" else None,
            "freq_alias": params.get("freq_alias"),
            "m": params.get("m"),
            "n": n,
        })

    add("summary", "residual_mean", diag.residual_mean)
    add("summary", "residual_std", diag.residual_std)

    for lag in sorted(diag.acf):
        if max_lag is not None and lag > max_lag:
            add("acf", "acf", None, lag=lag, lag_units="observations",
                status="not_computed",
                note=(f"key lag {lag} exceeds max_lag={max_lag} "
                      f"(n={n} residuals)"))
        else:
            add("acf", "acf", diag.acf[lag], lag=lag,
                lag_units="observations")

    if diag.ljung_box_lags is None:
        lb_note = "n <= 15: Ljung-Box skipped"
        add("ljung_box", "ljung_box_stat", None,
            status="not_computed", note=lb_note)
        add("ljung_box", "ljung_box_pvalue", None,
            status="not_computed", note=lb_note)
        add("ljung_box", "ljung_box_lags", None, lag_units="observations",
            status="not_computed", note=lb_note)
    else:
        lb_note = "Ljung-Box computation failed"
        add("ljung_box", "ljung_box_stat", diag.ljung_box_stat, note=lb_note)
        add("ljung_box", "ljung_box_pvalue", diag.ljung_box_pvalue,
            note=lb_note)
        add("ljung_box", "ljung_box_lags", diag.ljung_box_lags,
            lag_units="observations")

    if diag.shapiro_stat is None:
        # 3..5000 mirrors compute_diagnostics' default max_shapiro_n; a
        # non-default bound is not recoverable from the result object.
        if n is not None and 3 <= n <= 5000:
            sh_note = "Shapiro-Wilk computation failed"
        else:
            sh_note = "n outside 3..5000"
        add("shapiro", "shapiro_stat", None,
            status="not_computed", note=sh_note)
        add("shapiro", "shapiro_pvalue", None,
            status="not_computed", note=sh_note)
    else:
        add("shapiro", "shapiro_stat", diag.shapiro_stat)
        add("shapiro", "shapiro_pvalue", diag.shapiro_pvalue)

    add("params", "max_lag", max_lag, lag_units="observations")
    add("params", "min_acf_pairs", params.get("min_acf_pairs"))

    return rows


def save_diagnostics_table(diag, path):
    """Save residual diagnostics as a tidy long-format CSV (GH #64).

    One row per statistic; the persisted ACF vector gets one row per lag.
    freq_alias, m, and n repeat on every row so values stay self-describing;
    lag units are observations of the resolved frequency. status
    distinguishes ok / nan (computed, result NaN) / not_computed
    (precondition failed), with the reason in note. Until the final model
    is refit on train plus test (GH #63), the values describe the
    train-only fit.

    Parameters
    ----------
    diag : DiagnosticsResult
    path : str or Path
    """
    df = pd.DataFrame(_diag_rows(diag), columns=_DIAGNOSTICS_COLUMNS)
    for col in ("lag", "m", "n"):
        df[col] = df[col].astype("Int64")
    df.to_csv(path, index=False)
