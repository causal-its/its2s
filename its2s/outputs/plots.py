# Description: Counterfactual plot with train fit, test/holdout predictions, and CI ribbon.
# Usage: from its2s.outputs.plots import plot_counterfactual
# Dependencies: matplotlib, numpy, pandas

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_counterfactual(pipeline_result, splits, save_path=None, config=None):
    """Generate a counterfactual ITS plot.

    Parameters
    ----------
    pipeline_result : PipelineResult
        Output of run_single_its.
    splits : TimeSeriesSplits
        Data splits from prepare_splits.
    save_path : str or Path, optional
        If provided, save figure to this path.
    config : dict, optional
        Config dict for plot settings.
    """
    config = config or {}
    plot_cfg = config.get("output", {})
    figsize = tuple(plot_cfg.get("plot_figsize", [14, 6]))
    dpi = plot_cfg.get("plot_dpi", 150)

    br = pipeline_result.bootstrap_result
    fr = pipeline_result.fit_result
    intervention = splits.intervention_date

    fig, ax = plt.subplots(figsize=figsize)

    # Observed series (full data across all splits)
    all_dates = pd.concat([
        splits.train_df[["ds"]], splits.test_df[["ds"]], splits.holdout_df[["ds"]]
    ]).drop_duplicates()
    all_dates = all_dates.sort_values("ds")

    train_dates = pd.to_datetime(splits.train_df["ds"])
    train_y = splits.train_df[config.get("data", {}).get("target_col", "y")].values

    # Full observed
    target_col = config.get("data", {}).get("target_col", "y")
    for split_df, label, color in [
        (splits.train_df, None, "#333333"),
        (splits.test_df, None, "#333333"),
        (splits.holdout_df, None, "#333333"),
    ]:
        if not split_df.empty and target_col in split_df.columns:
            ax.plot(pd.to_datetime(split_df["ds"]), split_df[target_col],
                    color=color, linewidth=0.6, alpha=0.7)

    # Observed label (single entry)
    ax.plot([], [], color="#333333", linewidth=0.6, alpha=0.7, label="Observed")

    # Fitted line on training period
    ax.plot(train_dates, fr.fitted_values, color="#2166AC", linewidth=1.0,
            alpha=0.8, label="Model fit (train)")

    # Predicted line on test + holdout
    pred_dates = pd.to_datetime(br.dates)
    ax.plot(pred_dates, br.predicted, color="#B2182B", linewidth=1.2,
            label="Counterfactual prediction")

    # CI ribbon
    ax.fill_between(pred_dates, br.conf_lo, br.conf_hi,
                     color="#B2182B", alpha=0.15,
                     label=f"{pipeline_result.bootstrap_result.ci_level*100:.0f}% CI")

    # Intervention line
    ax.axvline(intervention, color="#4DAF4A", linestyle="--", linewidth=1.2,
               label="Intervention")

    # Holdout shading
    if not splits.holdout_df.empty:
        h_start = pd.to_datetime(splits.holdout_df["ds"]).min()
        h_end = pd.to_datetime(splits.holdout_df["ds"]).max()
        ax.axvspan(h_start, h_end, color="#FEE08B", alpha=0.2, label="Holdout period")

    ax.set_xlabel("Date")
    ax.set_ylabel(target_col)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)

    # Subtitle with test metrics
    mt = pipeline_result.metrics_test
    subtitle = f"Model: {pipeline_result.model_name}  |  Test RMSE: {mt.rmse:.2f}  |  Test MAPE: {mt.mape:.1f}%"
    ax.set_title(subtitle, fontsize=10)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig
