# Description: Residual diagnostic plots (GH #65): ACF correlogram, PACF,
#   residuals over time, and normal QQ, written into the run output directory.
#   The ACF correlogram renders the persisted DiagnosticsResult.acf vector and
#   never recomputes (the three-layer contract in docs/diagnostics.md); PACF is
#   computed at plot time. All figures describe the train-only fit until the
#   final refit lands (GH #63).
# Usage: from its2s.outputs.diagnostic_plots import plot_residual_diagnostics
# Dependencies: matplotlib, numpy, pandas (scipy and statsmodels imported
#   lazily inside functions)

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .plots import _resolve_plot_colors

_BAND_COLOR = "#BBBBBB"


def _plot_style(config):
    """Resolve shared style settings from config['output']."""
    config = config or {}
    plot_cfg = config.get("output", {})
    return {
        "dpi": plot_cfg.get("plot_dpi", 150),
        "colors": _resolve_plot_colors(plot_cfg),
        "title_fontsize": plot_cfg.get("plot_font_sizes", {}).get("title", 22),
        "axis_label_fontsize": plot_cfg.get("plot_font_sizes", {}).get("axis_label", 20),
        "tick_fontsize": plot_cfg.get("plot_font_sizes", {}).get("tick", 18),
        "legend_fontsize": plot_cfg.get("plot_font_sizes", {}).get("legend", 18),
    }


def _finish(fig, save_path, dpi):
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    return fig


def _annotate(ax, message, style):
    ax.text(
        0.98, 0.95, message, transform=ax.transAxes,
        ha="right", va="top", fontsize=style["tick_fontsize"] - 4,
        wrap=True,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )


def _placeholder(ax, message, style):
    ax.text(0.5, 0.5, message, transform=ax.transAxes,
            ha="center", va="center", fontsize=style["tick_fontsize"])
    ax.set_xticks([])
    ax.set_yticks([])


def _seasonal_lag_note(m, freq_alias, max_reachable, n):
    """Annotation when the dominant seasonal lag cannot be shown."""
    if m is None:
        return (f"No dominant seasonal cycle mapped for freq "
                f"'{freq_alias}'; key lag 1 only.")
    if m > max_reachable:
        return (f"Seasonal lag m={m} (freq {freq_alias}) exceeds max "
                f"estimable lag {max_reachable} (n={n}); the dominant-cycle "
                f"check is unavailable.")
    return None


def _correlogram(ax, lags, values, n, key_lags, style, ylabel):
    """Shared ACF/PACF rendering: vlines, white-noise band, key-lag marks."""
    band = 1.96 / np.sqrt(n) if n > 0 else np.nan
    if np.isfinite(band):
        ax.axhspan(-band, band, color=_BAND_COLOR, alpha=0.3,
                   label="95% white-noise band")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.vlines(lags, 0, values, color=style["colors"]["fit"], linewidth=1.5)

    lag_to_value = dict(zip(lags, values))
    marked = [lag for lag in key_lags if lag in lag_to_value]
    for lag in marked:
        val = lag_to_value[lag]
        ax.vlines([lag], 0, [val], color=style["colors"]["counterfactual"],
                  linewidth=2.5)
        ax.plot([lag], [val], "o", color=style["colors"]["counterfactual"],
                markersize=6)
        ax.annotate(f"lag {lag}: {val:.2f}", xy=(lag, val),
                    xytext=(0, 8 if val >= 0 else -14),
                    textcoords="offset points", ha="center",
                    fontsize=style["tick_fontsize"] - 4,
                    color=style["colors"]["counterfactual"])
    if marked:
        ax.plot([], [], "o", color=style["colors"]["counterfactual"],
                label="key lags")

    ax.set_ylabel(ylabel, fontsize=style["axis_label_fontsize"])
    ax.tick_params(axis="both", labelsize=style["tick_fontsize"])
    ax.legend(loc="lower left", fontsize=style["legend_fontsize"],
              framealpha=0.9)


def plot_residual_acf(diag, save_path=None, config=None):
    """Residual ACF correlogram from the persisted diagnostics vector.

    Renders DiagnosticsResult.acf as computed (never recomputes), with
    95% white-noise bands (+/-1.96/sqrt(n)) and the key lags {1, m}
    marked. Describes the train-only fit.

    Parameters
    ----------
    diag : DiagnosticsResult
    save_path : str or Path, optional
    config : dict, optional
    """
    style = _plot_style(config)
    params = diag.params
    n = params.get("n", 0)
    max_lag = params.get("max_lag", 0)
    m = params.get("m")
    freq_alias = params.get("freq_alias")
    model_name = diag.model_metadata.get("model_name", "model")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(f"Residual ACF (train-only fit) - {model_name}",
                 fontsize=style["title_fontsize"])

    in_range = sorted(lag for lag in diag.acf if lag <= max_lag)
    if not in_range:
        message = f"Series too short for autocorrelation estimates (n={n})"
        warnings.warn(f"Diagnostic plot [{model_name}]: {message}.",
                      UserWarning, stacklevel=2)
        _placeholder(ax, message, style)
        return _finish(fig, save_path, style["dpi"])

    values = [diag.acf[lag] for lag in in_range]
    _correlogram(ax, in_range, values, n, diag.key_lags, style, ylabel="ACF")
    ax.set_xlabel(f"Lag (observations, freq {freq_alias})",
                  fontsize=style["axis_label_fontsize"])

    note = _seasonal_lag_note(m, freq_alias, max_lag, n)
    if note:
        _annotate(ax, note, style)

    return _finish(fig, save_path, style["dpi"])


def plot_residual_pacf(diag, fit_result, save_path=None, config=None):
    """Residual PACF correlogram, computed at plot time.

    Uses statsmodels pacf (method 'ywm': the yule-walker variant that
    cannot exceed |1| on short series, unlike 'ywadjusted'), capped at
    min(max_lag, n // 2 - 1) per the statsmodels bound. Same bands and
    annotations as the ACF plot so the two read in parallel. Describes
    the train-only fit.

    Parameters
    ----------
    diag : DiagnosticsResult
    fit_result : FitResult
        Source of the raw residuals (NaNs are dropped).
    save_path : str or Path, optional
    config : dict, optional
    """
    style = _plot_style(config)
    params = diag.params
    m = params.get("m")
    freq_alias = params.get("freq_alias")
    model_name = diag.model_metadata.get("model_name", "model")

    residuals = np.asarray(fit_result.residuals, dtype=float)
    residuals = residuals[~np.isnan(residuals)]
    n = len(residuals)
    nlags = min(params.get("max_lag", 0), n // 2 - 1)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(f"Residual PACF (train-only fit) - {model_name}",
                 fontsize=style["title_fontsize"])

    if nlags < 1:
        message = ("Series too short for partial autocorrelation "
                   f"estimates (n={n})")
        warnings.warn(f"Diagnostic plot [{model_name}]: {message}.",
                      UserWarning, stacklevel=2)
        _placeholder(ax, message, style)
        return _finish(fig, save_path, style["dpi"])

    from statsmodels.tsa.stattools import pacf

    values = pacf(residuals, nlags=nlags, method="ywm")[1:]
    lags = list(range(1, nlags + 1))
    _correlogram(ax, lags, values, n, diag.key_lags, style, ylabel="PACF")
    ax.set_xlabel(f"Lag (observations, freq {freq_alias})",
                  fontsize=style["axis_label_fontsize"])

    note = _seasonal_lag_note(m, freq_alias, nlags, n)
    if note:
        _annotate(ax, note, style)

    return _finish(fig, save_path, style["dpi"])


def plot_residuals_over_time(fit_result, splits, save_path=None, config=None):
    """Raw training residuals against the training dates.

    NaN residuals (e.g. NeuralProphet AR warmup) appear as gaps. Falls
    back to an observation-index x-axis with a warning if the residual
    and training lengths disagree. Describes the train-only fit.

    Parameters
    ----------
    fit_result : FitResult
    splits : TimeSeriesSplits
    save_path : str or Path, optional
    config : dict, optional
    """
    style = _plot_style(config)
    config = config or {}
    date_col = config.get("data", {}).get("date_col", "ds")

    residuals = np.asarray(fit_result.residuals, dtype=float)
    if len(residuals) == len(splits.train_df):
        x = pd.to_datetime(splits.train_df[date_col])
        xlabel = "Date"
    else:
        warnings.warn(
            f"Residuals over time: residual count ({len(residuals)}) does "
            f"not match the training rows ({len(splits.train_df)}); "
            "plotting against observation index.",
            UserWarning, stacklevel=2,
        )
        x = np.arange(len(residuals))
        xlabel = "Observation index"

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.plot(x, residuals, color=style["colors"]["fit"], linewidth=0.9,
            marker=".", markersize=3)
    ax.set_title("Residuals over time (train-only fit)",
                 fontsize=style["title_fontsize"])
    ax.set_xlabel(xlabel, fontsize=style["axis_label_fontsize"])
    ax.set_ylabel("Residual", fontsize=style["axis_label_fontsize"])
    ax.tick_params(axis="both", labelsize=style["tick_fontsize"])

    return _finish(fig, save_path, style["dpi"])


def plot_residual_qq(fit_result, save_path=None, config=None):
    """Normal QQ plot of the training residuals (NaNs dropped).

    Describes the train-only fit.

    Parameters
    ----------
    fit_result : FitResult
    save_path : str or Path, optional
    config : dict, optional
    """
    style = _plot_style(config)
    residuals = np.asarray(fit_result.residuals, dtype=float)
    residuals = residuals[~np.isnan(residuals)]
    n = len(residuals)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Residual normal QQ (train-only fit)",
                 fontsize=style["title_fontsize"])

    if n < 3:
        message = f"Series too short for a QQ plot (n={n})"
        warnings.warn(f"Diagnostic plot: {message}.", UserWarning,
                      stacklevel=2)
        _placeholder(ax, message, style)
        return _finish(fig, save_path, style["dpi"])

    from scipy.stats import probplot

    (osm, osr), (slope, intercept, _r) = probplot(residuals, dist="norm")
    ax.plot(osm, osr, "o", color=style["colors"]["fit"], markersize=4,
            alpha=0.8)
    ax.plot(osm, slope * np.asarray(osm) + intercept,
            color=style["colors"]["counterfactual"], linewidth=1.2)
    ax.set_xlabel("Theoretical quantiles",
                  fontsize=style["axis_label_fontsize"])
    ax.set_ylabel("Sample quantiles", fontsize=style["axis_label_fontsize"])
    ax.tick_params(axis="both", labelsize=style["tick_fontsize"])

    return _finish(fig, save_path, style["dpi"])


def plot_residual_diagnostics(diag, fit_result, splits, output_dir,
                              model_name, config=None):
    """Write all four residual diagnostic plots to the output directory.

    Files: {model_name}_residual_acf.png, {model_name}_residual_pacf.png,
    {model_name}_residuals_over_time.png, {model_name}_residual_qq.png.

    Parameters
    ----------
    diag : DiagnosticsResult
    fit_result : FitResult
    splits : TimeSeriesSplits
    output_dir : str or Path
    model_name : str
    config : dict, optional

    Returns
    -------
    list of Path
        The four written file paths.
    """
    out = Path(output_dir)
    paths = []

    path = out / f"{model_name}_residual_acf.png"
    plot_residual_acf(diag, save_path=path, config=config)
    paths.append(path)

    path = out / f"{model_name}_residual_pacf.png"
    plot_residual_pacf(diag, fit_result, save_path=path, config=config)
    paths.append(path)

    path = out / f"{model_name}_residuals_over_time.png"
    plot_residuals_over_time(fit_result, splits, save_path=path,
                             config=config)
    paths.append(path)

    path = out / f"{model_name}_residual_qq.png"
    plot_residual_qq(fit_result, save_path=path, config=config)
    paths.append(path)

    return paths
