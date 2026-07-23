# Description: Main orchestrator for single-run ITS counterfactual analysis.
# Usage: from its2s.pipeline import run_single_its
# Dependencies: all its2s submodules

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .bootstrap.mbb import MovingBlockBootstrap
from .diagnostics import compute_diagnostics, DiagnosticsResult
from .settings import get_model_config, load_config
from .data_prep import prepare_splits, resolve_split_config
from .frequency import resolve_frequency
from .validation import validate_inputs
from .metrics.error_metrics import compute_metrics, MetricsResult
from .metrics.excess import ExcessResult, calc_ate_summary, calculate_excess
from .outputs.plots import plot_counterfactual
from .outputs.tables import save_ate_summary, save_excess_table, save_metrics_table

logger = logging.getLogger(__name__)

# Lazy model import map: model name -> (module path, class name)
_MODEL_IMPORT_MAP = {
    "prophet_xgb": (".models.prophet_xgb", "ProphetXGBHybridModel"),
    "prophet_then_xgb": (".models.prophet_then_xgb", "ProphetThenXGBModel"),
    "neuralprophet": (".models.neuralprophet", "NeuralProphetModel"),
    "arima": (".models.arima", "ARIMAModel"),
}

_MODEL_CACHE = {}


def _get_available_model_names():
    """Return list of model names whose dependencies can be imported."""
    available = []
    for name in _MODEL_IMPORT_MAP:
        try:
            _get_model_class(name)
            available.append(name)
        except ImportError:
            pass
    return available


def _get_model_class(model_name):
    """Import and cache a single model class by name."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    if model_name not in _MODEL_IMPORT_MAP:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Known models: {list(_MODEL_IMPORT_MAP.keys())}"
        )

    module_path, class_name = _MODEL_IMPORT_MAP[model_name]
    try:
        import importlib
        mod = importlib.import_module(module_path, package=__package__)
        cls = getattr(mod, class_name)
    except ImportError as e:
        raise ImportError(
            f"Model '{model_name}' is not available because a required "
            f"dependency could not be imported: {e}. "
            f"Check that all dependencies for this model are installed."
        ) from e

    _MODEL_CACHE[model_name] = cls
    return cls


def _get_model(model_name, params):
    cls = _get_model_class(model_name)
    return cls(params=params)


@dataclass
class PipelineResult:
    """Output of a single ITS pipeline run.

    Attributes
    ----------
    model_name : str
        Name of the model used (e.g. "prophet_xgb", "arima").
    fit_result : FitResult
        Raw fit output from the model, including fitted_values and residuals
        on the training period.
    bootstrap_result : BootstrapCIResult
        MBB output for the prediction period (test + holdout). Exposes
        dates, actual, predicted, conf_lo, conf_hi, pred_matrix, and
        n_successful.
    metrics_train : MetricsResult
        RMSE, MAE, MAPE, and R2 computed on the training period.
    metrics_test : MetricsResult
        RMSE, MAE, MAPE, and R2 computed on the test period.
    excess_table : ExcessResult
        Per-observation excess estimates with CIs for the holdout period.
        Pass to calc_ate_summary() to get total and per-observation ATE
        with CIs.
    config : dict
        Full resolved config dict used for this run.
    diagnostics : DiagnosticsResult or None
        Residual diagnostics (Ljung-Box, Shapiro-Wilk, ACF lags). None if
        diagnostics could not be computed.
    series_frequency : SeriesFrequency or None
        Frequency resolved from the data (its2s.frequency), the single
        source for all window-unit interpretation.
    """

    model_name: str
    fit_result: object
    bootstrap_result: object
    metrics_train: MetricsResult
    metrics_test: MetricsResult
    excess_table: ExcessResult
    config: dict
    diagnostics: DiagnosticsResult | None = None
    series_frequency: object | None = None

    def summary(self):
        """Return a human-readable summary string."""
        lines = []
        lines.append(f"Model: {self.model_name}")
        lines.append(f"Bootstrap: {self.config['bootstrap']['n_sim']} simulations, "
                      f"{self.bootstrap_result.n_successful} successful")
        lines.append("")
        lines.append("Train metrics:")
        mt = self.metrics_train
        lines.append(f"  RMSE={mt.rmse:.4f}  MAE={mt.mae:.4f}  "
                      f"MAPE={mt.mape:.2f}%  R2={mt.r2:.4f}")
        lines.append("Test metrics:")
        mt = self.metrics_test
        lines.append(f"  RMSE={mt.rmse:.4f}  MAE={mt.mae:.4f}  "
                      f"MAPE={mt.mape:.2f}%  R2={mt.r2:.4f}")
        if not self.excess_table.obs_excess.empty:
            ate = calc_ate_summary(self.excess_table)
            total = ate[ate["metric"] == "Total ATE"].iloc[0]
            per_obs = ate[ate["metric"] == "Mean ATE per obs"].iloc[0]
            lines.append("")
            lines.append(f"Total ATE: {total['estimate']:.2f} "
                          f"[{total['ci_lo']:.2f}, {total['ci_hi']:.2f}]")
            lines.append(f"Mean ATE per obs: {per_obs['estimate']:.4f} "
                          f"[{per_obs['ci_lo']:.4f}, {per_obs['ci_hi']:.4f}]")
            lines.append(f"Holdout obs: {int(total['n_obs'])}")
        if self.diagnostics:
            d = self.diagnostics
            lines.append("")
            lines.append("Residual diagnostics:")
            lines.append(f"  Mean={d.residual_mean:.4f}  "
                          f"Std={d.residual_std:.4f}")
            lines.append(f"  ACF(1)={d.acf_lag1:.3f}  "
                          f"ACF(7)={d.acf_lag7:.3f}  "
                          f"ACF(14)={d.acf_lag14:.3f}")
            if not pd.isna(d.ljung_box_pvalue):
                lines.append(f"  Ljung-Box(10) p={d.ljung_box_pvalue:.4f}")
            if d.shapiro_pvalue is not None:
                lines.append(f"  Shapiro-Wilk p={d.shapiro_pvalue:.4f}")
        return "\n".join(lines)


def run_single_its(
    df,
    intervention_date,
    target_col=None,
    date_col=None,
    covariate_cols=None,
    model_name="prophet_xgb",
    config_path=None,
    config_overrides=None,
    output_dir=None,
    seed=42,
    split_method=None,
):
    """Run a single ITS counterfactual analysis pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Full time series dataset.
    intervention_date : str or pd.Timestamp
        Date of the intervention.
    target_col : str, optional
        Target column name. Defaults to config value.
    date_col : str, optional
        Date column name. Defaults to config value.
    covariate_cols : list[str], optional
        Covariate column names. Defaults to config value.
    model_name : str
        Model to use. One of: prophet_xgb, prophet_then_xgb, neuralprophet, arima.
    config_path : str or Path, optional
        Path to custom YAML config.
    config_overrides : dict, optional
        Runtime config overrides.
    output_dir : str or Path, optional
        Directory for saving outputs. If None, no files are saved.
    seed : int
        Random seed for bootstrap reproducibility.

    Returns
    -------
    PipelineResult
    """
    # 1. Load config
    config = load_config(config_path, config_overrides)
    date_col = date_col or config["data"]["date_col"]
    target_col = target_col or config["data"]["target_col"]
    covariate_cols = covariate_cols if covariate_cols is not None else config["data"]["covariate_cols"]
    config["data"]["date_col"] = date_col
    config["data"]["target_col"] = target_col

    # Resolve split-method config (function kwarg overrides config). Only the
    # arguments belonging to the resolved method are read and passed on:
    # prepare_splits raises on cross-method arguments (#28, #54).
    periods_cfg = config["periods"]
    split_method_resolved, split_kwargs = resolve_split_config(
        periods_cfg, split_method)

    # 1b. Validate inputs
    validate_inputs(df, intervention_date, date_col, target_col,
                    covariate_cols, model_name,
                    split_method=split_method_resolved,
                    **split_kwargs)

    # M2-6: Check date-sort and warn if DataFrame is unsorted
    dates_check = pd.to_datetime(df[date_col])
    if not dates_check.is_monotonic_increasing:
        warnings.warn(
            f"DataFrame is not sorted by '{date_col}'. Rows will be reordered "
            "before model fitting. If covariate columns are present, ensure "
            "their values are aligned with the date column, not with the "
            "original row positions. Pre-sort your DataFrame to suppress this warning.",
            UserWarning,
            stacklevel=2,
        )

    # 1c. Handle missing data in target column
    missing_strategy = config["data"].get("missing_data", "error")
    n_missing = df[target_col].isna().sum()
    if n_missing > 0:
        if missing_strategy == "error":
            raise ValueError(
                f"Target column '{target_col}' contains {n_missing} missing "
                f"values. Set data.missing_data to 'drop' or 'interpolate' "
                f"in config to handle them automatically."
            )
        elif missing_strategy == "drop":
            warnings.warn(
                f"Dropping {n_missing} row(s) with missing values in '{target_col}'.",
                UserWarning,
                stacklevel=2,
            )
            df = df.dropna(subset=[target_col]).copy()
        elif missing_strategy == "interpolate":
            warnings.warn(
                f"Interpolating {n_missing} missing value(s) in '{target_col}' "
                "using linear method.",
                UserWarning,
                stacklevel=2,
            )
            df = df.copy()
            df[target_col] = df[target_col].interpolate(method="linear")
            df[target_col] = df[target_col].bfill().ffill()
        else:
            raise ValueError(
                f"Unknown missing_data strategy: '{missing_strategy}'. "
                f"Expected 'error', 'drop', or 'interpolate'."
            )

    # 1d. Resolve series frequency once, from the data, after missing-data
    # handling so that rows dropped above surface here as gaps (#48, #52).
    series_freq = resolve_frequency(pd.to_datetime(df[date_col]).sort_values())
    logger.info("Resolved series frequency: %s", series_freq.alias)

    # 2. Prepare splits (prepare_splits logs the resulting window sizes)
    splits = prepare_splits(
        df,
        intervention_date,
        date_col=date_col,
        split_method=split_method_resolved,
        min_test_obs=periods_cfg.get("min_test_obs", 30),
        **split_kwargs,
    )

    # 3. Instantiate model
    model_params = get_model_config(config, model_name)
    if model_name == "neuralprophet":
        # freq comes from the resolved series frequency, never from user
        # config: a declared value cannot disagree with the data (#52).
        model_params = dict(model_params)
        model_params["freq"] = series_freq.alias
    model = _get_model(model_name, model_params)

    # 3b. Warn about long-horizon ARIMA forecasts (B5)
    holdout_days = len(splits.holdout_df)
    if model_name == "arima" and holdout_days > 90:
        warnings.warn(
            f"ARIMA with holdout_days={holdout_days}: ARIMA point forecasts "
            "converge to the unconditional mean over long horizons, which can "
            "bias the counterfactual estimate. Consider prophet_xgb or "
            "prophet_then_xgb for holdout windows beyond 90 days.",
            UserWarning,
            stacklevel=2,
        )

    # 4. Fit model
    logger.info("Fitting %s model...", model_name)
    fit_result = model.fit(
        splits.train_df, target_col=target_col,
        date_col=date_col, covariate_cols=covariate_cols or None,
    )

    # 4b. Compute residual diagnostics
    diag = compute_diagnostics(fit_result, model_name)

    # 5. Bootstrap CIs
    boot_config = config["bootstrap"]
    mbb = MovingBlockBootstrap(
        n_sim=boot_config["n_sim"],
        block_length=boot_config["block_length"],
        ci_method=boot_config["ci_method"],
        ci_level=boot_config["ci_level"],
        n_jobs=boot_config.get("n_jobs", 1),
    )

    logger.info("Running MBB with %d simulations...", boot_config["n_sim"])
    bootstrap_result = mbb.generate_cis(
        model, splits.train_df, splits.full_predict_df,
        target_col=target_col, date_col=date_col,
        covariate_cols=covariate_cols or None, seed=seed,
    )

    # 6. Compute metrics
    train_pred = model.predict(
        splits.train_df, target_col=target_col,
        date_col=date_col, covariate_cols=covariate_cols or None,
    )
    metrics_train = compute_metrics(
        splits.train_df[target_col].values,
        train_pred.predicted,
        seasonality=config["metrics"]["seasonality"],
    )

    # Test metrics from bootstrap point predictions
    test_mask = pd.to_datetime(bootstrap_result.dates) < splits.intervention_date
    if test_mask.any():
        metrics_test = compute_metrics(
            bootstrap_result.actual[test_mask],
            bootstrap_result.predicted[test_mask],
            training_actual=splits.train_df[target_col].values,
            seasonality=config["metrics"]["seasonality"],
        )
    else:
        metrics_test = metrics_train

    # 7. Excess calculation
    excess_table = calculate_excess(
        bootstrap_result,
        intervention_date=splits.intervention_date,
        periods_config=config.get("excess_periods") or None,
        ci_level=boot_config["ci_level"],
    )

    # 8. Save outputs
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        result_for_plot = PipelineResult(
            model_name=model_name,
            fit_result=fit_result,
            bootstrap_result=bootstrap_result,
            metrics_train=metrics_train,
            metrics_test=metrics_test,
            excess_table=excess_table,
            config=config,
            diagnostics=diag,
            series_frequency=series_freq,
        )

        plot_counterfactual(
            result_for_plot, splits,
            save_path=out / f"{model_name}_counterfactual.png",
            config=config,
        )
        save_excess_table(excess_table, out / f"{model_name}_excess.csv")
        save_metrics_table(
            {"train": metrics_train, "test": metrics_test},
            out / f"{model_name}_metrics.csv",
        )
        if not excess_table.obs_excess.empty:
            ate = calc_ate_summary(excess_table)
            save_ate_summary(ate, out / f"{model_name}_ate_summary.csv")

        logger.info("Outputs saved to %s", out)

    return PipelineResult(
        model_name=model_name,
        fit_result=fit_result,
        bootstrap_result=bootstrap_result,
        metrics_train=metrics_train,
        metrics_test=metrics_test,
        excess_table=excess_table,
        config=config,
        diagnostics=diag,
        series_frequency=series_freq,
    )
