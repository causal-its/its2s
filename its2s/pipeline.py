# Description: Main orchestrator for single-run ITS counterfactual analysis.
# Usage: from its2s.pipeline import run_single_its
# Dependencies: all its2s submodules

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .bootstrap.block_length import grid_search_block_length, resolve_block_length
from .bootstrap.mbb import MovingBlockBootstrap
from .diagnostics import compute_diagnostics, DiagnosticsResult
from .settings import get_model_config, load_config
from .data_prep import prepare_splits
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
class _PreparedRun:
    """Shared output of the load/validate/split/fit setup (pipeline steps 1-4).

    Produced by :func:`_prepare_fitted_model` and consumed by both
    :func:`run_single_its` and :func:`calibrate_block_length` so that block-length
    calibration fits the model on exactly the same train data the run uses.
    """

    config: dict
    target_col: str
    date_col: str
    covariate_cols: list | None
    splits: object
    model: object
    fit_result: object


def _prepare_fitted_model(df, intervention_date, target_col=None, date_col=None,
                          covariate_cols=None, model_name="prophet_xgb",
                          config_path=None, config_overrides=None,
                          split_method=None):
    """Run pipeline steps 1-4: config, validation, missing data, splits, fit.

    Centralizes the setup shared by the single-run pipeline and block-length
    calibration. Returns a :class:`_PreparedRun`. Side effects (validation
    errors, sort/missing-data/ARIMA-horizon warnings) match the single-run
    pipeline exactly, because both paths call this function.
    """
    # 1. Load config
    config = load_config(config_path, config_overrides)
    date_col = date_col or config["data"]["date_col"]
    target_col = target_col or config["data"]["target_col"]
    covariate_cols = covariate_cols if covariate_cols is not None else config["data"]["covariate_cols"]
    config["data"]["date_col"] = date_col
    config["data"]["target_col"] = target_col

    # Resolve split-method config (function kwarg overrides config)
    periods_cfg = config["periods"]
    if split_method is not None:
        periods_cfg["split_method"] = split_method
    split_method_resolved = periods_cfg.get("split_method", "percent")
    test_pct_resolved = periods_cfg.get("test_pct", 0.20)
    holdout_pct_resolved = periods_cfg.get("holdout_pct", 1.0)
    test_days_resolved = periods_cfg.get("test_days", 365)
    holdout_days_resolved = periods_cfg.get("holdout_days", 365)

    # 1b. Validate inputs
    validate_inputs(df, intervention_date, date_col, target_col,
                    covariate_cols, model_name,
                    split_method=split_method_resolved,
                    test_pct=test_pct_resolved,
                    holdout_pct=holdout_pct_resolved,
                    test_days=test_days_resolved,
                    holdout_days=holdout_days_resolved)

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

    # 2. Prepare splits
    splits = prepare_splits(
        df,
        intervention_date,
        date_col=date_col,
        split_method=split_method_resolved,
        test_pct=test_pct_resolved,
        holdout_pct=holdout_pct_resolved,
        test_days=test_days_resolved,
        holdout_days=holdout_days_resolved,
    )

    logger.info(
        "Splits: train=%d, test=%d, holdout=%d",
        len(splits.train_df), len(splits.test_df), len(splits.holdout_df),
    )

    # 3. Instantiate model
    model_params = get_model_config(config, model_name)
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

    return _PreparedRun(
        config=config,
        target_col=target_col,
        date_col=date_col,
        covariate_cols=covariate_cols,
        splits=splits,
        model=model,
        fit_result=fit_result,
    )


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
        Day-level excess estimates with CIs for the holdout period. Pass to
        calc_ate_summary() to get total and mean-daily ATE with CIs.
    config : dict
        Full resolved config dict used for this run.
    diagnostics : DiagnosticsResult or None
        Residual diagnostics (Ljung-Box, Shapiro-Wilk, ACF lags). None if
        diagnostics could not be computed.
    """

    model_name: str
    fit_result: object
    bootstrap_result: object
    metrics_train: MetricsResult
    metrics_test: MetricsResult
    excess_table: ExcessResult
    config: dict
    diagnostics: DiagnosticsResult | None = None

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
        if not self.excess_table.daily_excess.empty:
            ate = calc_ate_summary(self.excess_table)
            total = ate[ate["metric"] == "Total ATE"].iloc[0]
            daily = ate[ate["metric"] == "Mean Daily ATE"].iloc[0]
            lines.append("")
            lines.append(f"Total ATE: {total['estimate']:.2f} "
                          f"[{total['ci_lo']:.2f}, {total['ci_hi']:.2f}]")
            lines.append(f"Mean Daily ATE: {daily['estimate']:.4f} "
                          f"[{daily['ci_lo']:.4f}, {daily['ci_hi']:.4f}]")
            lines.append(f"Holdout days: {int(total['n_days'])}")
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
    # 1-4. Load config, validate, handle missing data, split, and fit. Shared
    # with calibrate_block_length so calibration fits on identical train data.
    prep = _prepare_fitted_model(
        df,
        intervention_date,
        target_col=target_col,
        date_col=date_col,
        covariate_cols=covariate_cols,
        model_name=model_name,
        config_path=config_path,
        config_overrides=config_overrides,
        split_method=split_method,
    )
    config = prep.config
    date_col = prep.date_col
    target_col = prep.target_col
    covariate_cols = prep.covariate_cols
    splits = prep.splits
    model = prep.model
    fit_result = prep.fit_result

    # 4b. Compute residual diagnostics
    diag = compute_diagnostics(fit_result, model_name)

    # 5. Bootstrap CIs. Resolve block_length (int / "auto"; "grid" is rejected
    # here -- it is a one-off calibration mode, see calibrate_block_length).
    boot_config = config["bootstrap"]
    warmup = int(getattr(model, "warmup_rows", 0))
    resid_finite = fit_result.residuals[warmup:]
    resolved_block_length = resolve_block_length(
        boot_config["block_length"], residuals=resid_finite
    )
    if resolved_block_length != boot_config["block_length"]:
        logger.info(
            "Resolved block_length %r -> %d observations.",
            boot_config["block_length"], resolved_block_length,
        )
    # Persist the resolved int back onto the config so it appears in
    # PipelineResult.config and any saved config (partial step toward #32).
    boot_config["block_length"] = resolved_block_length

    mbb = MovingBlockBootstrap(
        n_sim=boot_config["n_sim"],
        block_length=resolved_block_length,
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
        if not excess_table.daily_excess.empty:
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
    )


def calibrate_block_length(
    df,
    intervention_date,
    target_col=None,
    date_col=None,
    covariate_cols=None,
    model_name="prophet_xgb",
    config_path=None,
    config_overrides=None,
    seed=42,
    split_method=None,
    L_range=None,
    n_sim=None,
    tol=0.05,
    window=5,
):
    """Calibrate the MBB block length once by CI-width-stability grid search.

    This is the one-off calibration step behind ``block_length: "grid"``. It
    fits the model on the SAME train data as :func:`run_single_its` (both call
    :func:`_prepare_fitted_model`), then runs the grid search over the
    post-intervention event window and returns the selected L together with the
    width-vs-L diagnostic curve. Set ``bootstrap.block_length`` to the returned
    int for subsequent runs (grid search is too expensive to run every pipeline
    call -- ``len(L_range) * n_sim`` model refits).

    Parameters
    ----------
    df, intervention_date, target_col, date_col, covariate_cols, model_name,
    config_path, config_overrides, seed, split_method
        Same meaning as in :func:`run_single_its`; the setup is shared.
    L_range : iterable[int], optional
        Candidate block lengths. Defaults (in
        :func:`~its2s.bootstrap.block_length.grid_search_block_length`) to 1..50.
    n_sim : int, optional
        MBB simulations per candidate L. Defaults to the config's
        ``bootstrap.n_sim``. This is the dominant cost.
    tol : float
        Plateau threshold on the relative change in CI width (default 0.05).
    window : int
        Consecutive sub-threshold steps required for a plateau (default 5).

    Returns
    -------
    tuple[int, pandas.DataFrame]
        ``(L, diagnostics)`` where ``diagnostics`` has columns
        ``L, ci_lo, ci_hi, ci_width, rel_change`` (the Figure S2 evidence).
    """
    prep = _prepare_fitted_model(
        df,
        intervention_date,
        target_col=target_col,
        date_col=date_col,
        covariate_cols=covariate_cols,
        model_name=model_name,
        config_path=config_path,
        config_overrides=config_overrides,
        split_method=split_method,
    )
    boot_config = prep.config["bootstrap"]
    if n_sim is None:
        n_sim = boot_config.get("n_sim", 500)

    if prep.splits.holdout_df.empty:
        raise ValueError(
            "Cannot calibrate block length: the post-intervention event window "
            "(holdout) is empty. Check intervention_date and the split settings."
        )

    logger.info(
        "Calibrating block length: grid search over %s at n_sim=%d "
        "(event window = %d post-intervention rows).",
        "the default L range 1..50" if L_range is None else "the given L range",
        n_sim, len(prep.splits.holdout_df),
    )

    L, diagnostics = grid_search_block_length(
        prep.model,
        prep.splits.train_df,
        prep.splits.holdout_df,
        L_range=L_range,
        n_sim=n_sim,
        target_col=prep.target_col,
        date_col=prep.date_col,
        covariate_cols=prep.covariate_cols or None,
        ci_level=boot_config.get("ci_level", 0.95),
        tol=tol,
        window=window,
        seed=seed,
        n_jobs=boot_config.get("n_jobs", 1),
        return_diagnostics=True,
    )

    logger.info(
        "Calibrated block_length L=%d. Set bootstrap.block_length to this int "
        "for subsequent runs.", L,
    )
    return L, diagnostics
