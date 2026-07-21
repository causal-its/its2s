# Description: Counterfactual plot with train fit, test/holdout predictions, and CI ribbon.
# Usage: from its2s.outputs.plots import plot_counterfactual
# Dependencies: matplotlib, pandas

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_DEFAULT_PLOT_COLORS = {
    "observed": "#333333",
    "fit": "#2166AC",
    "counterfactual": "#B2182B",
    "intervention": "#4DAF4A",
    "holdout": "#FEE08B",
}

_PLOT_COLOR_SEQUENCE_KEYS = ("fit", "counterfactual", "intervention", "holdout")


def _resolve_plot_colors(plot_cfg):
    """Resolve plot colors from defaults plus optional config overrides."""
    colors = _DEFAULT_PLOT_COLORS.copy()
    overrides = plot_cfg.get("plot_colors")
    if overrides is None:
        return colors
    if isinstance(overrides, dict):
        colors.update(overrides)
        return colors
    if isinstance(overrides, str):
        raise TypeError("output.plot_colors must be a dict or sequence of color strings.")
    for key, color in zip(_PLOT_COLOR_SEQUENCE_KEYS, overrides):
        colors[key] = color
    return colors


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
    colors = _resolve_plot_colors(plot_cfg)
    font_sizes = plot_cfg.get("plot_font_sizes", {})
    title_fontsize = font_sizes.get("title", 22)
    axis_label_fontsize = font_sizes.get("axis_label", 20)
    tick_fontsize = font_sizes.get("tick", 18)
    legend_fontsize = font_sizes.get("legend", 18)
    date_col = config.get("data", {}).get("date_col", "ds")

    br = pipeline_result.bootstrap_result
    fr = pipeline_result.fit_result
    intervention = splits.intervention_date

    fig, ax = plt.subplots(figsize=figsize)

    train_dates = pd.to_datetime(splits.train_df[date_col])

    # Full observed
    target_col = config.get("data", {}).get("target_col", "y")
    for split_df, label, color in [
        (splits.train_df, None, colors["observed"]),
        (splits.test_df, None, colors["observed"]),
        (splits.holdout_df, None, colors["observed"]),
    ]:
        if not split_df.empty and target_col in split_df.columns:
            ax.plot(pd.to_datetime(split_df[date_col]), split_df[target_col],
                    color=color, linewidth=0.6, alpha=0.7)

    # Observed label (single entry)
    ax.plot([], [], color=colors["observed"], linewidth=0.6, alpha=0.7, label="Observed")

    # Fitted line on training period
    ax.plot(train_dates, fr.fitted_values, color=colors["fit"], linewidth=1.0,
            alpha=0.8, label="Model fit (train)")

    # Predicted line on test + holdout
    pred_dates = pd.to_datetime(br.dates)
    ax.plot(pred_dates, br.predicted, color=colors["counterfactual"], linewidth=1.2,
            label="Counterfactual prediction")

    # CI ribbon
    ax.fill_between(pred_dates, br.conf_lo, br.conf_hi,
                     color=colors["counterfactual"], alpha=0.15,
                     label=f"{pipeline_result.bootstrap_result.ci_level*100:.0f}% CI")

    # Intervention line
    ax.axvline(intervention, color=colors["intervention"], linestyle="--", linewidth=1.2,
               label="Intervention")

    # Holdout shading
    if not splits.holdout_df.empty:
        h_start = pd.to_datetime(splits.holdout_df[date_col]).min()
        h_end = pd.to_datetime(splits.holdout_df[date_col]).max()
        ax.axvspan(h_start, h_end, color=colors["holdout"], alpha=0.2, label="Holdout period")

    ax.set_xlabel("Date", fontsize=axis_label_fontsize)
    ax.set_ylabel(target_col, fontsize=axis_label_fontsize)
    ax.tick_params(axis="both", labelsize=tick_fontsize)
    ax.legend(loc="upper left", fontsize=legend_fontsize, framealpha=0.9)

    # Subtitle with test metrics
    mt = pipeline_result.metrics_test
    subtitle = f"Model: {pipeline_result.model_name}  |  Test RMSE: {mt.rmse:.2f}  |  Test MAPE: {mt.mape:.1f}%"
    ax.set_title(subtitle, fontsize=title_fontsize)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return fig
