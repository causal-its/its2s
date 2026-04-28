# Description: Main orchestrator for single-run ITS counterfactual analysis.
# Usage: from its2s.pipeline import run_single_its
# Dependencies: all its2s submodules

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .bootstrap.mbb import MovingBlockBootstrap
from .settings import get_model_config, load_config
from .data_prep import prepare_splits
from .metrics.error_metrics import compute_metrics, MetricsResult
from .metrics.excess import ExcessResult, calc_ate_summary, calculate_excess
from .outputs.plots import plot_counterfactual
from .outputs.tables import save_ate_summary, save_excess_table, save_metrics_table

logger = logging.getLogger(__name__)

# Lazily import only the requested model so e.g. `arima` does not load xgboost.
_MODEL_CLASS_CACHE = {}


def _is_xgboost_native_load_failure(err):
    """True when xgboost's shared library failed (e.g. missing libomp on macOS)."""
    name = type(err).__name__
    if "XGBoost" in name:
        return True
    msg = str(err).lower()
    return "libxgboost" in msg or "libomp" in msg or "openmp" in msg


def _get_model_class(model_name):
    if model_name in _MODEL_CLASS_CACHE:
        return _MODEL_CLASS_CACHE[model_name]

    if model_name == "arima":
        try:
            from .models.arima import ARIMAModel

            cls = ARIMAModel
        except ImportError as e:
            raise ImportError(
                "ARIMA backend could not be imported. Install project dependencies "
                "(e.g. pip install -e . from the its2s repo, or pip install pmdarima statsmodels)."
            ) from e
    elif model_name == "neuralprophet":
        try:
            from .models.neuralprophet import NeuralProphetModel

            cls = NeuralProphetModel
        except ImportError as e:
            logger.warning("NeuralProphet not available (missing dependency).")
            raise ImportError(
                "NeuralProphet could not be imported. Install with: pip install neuralprophet"
            ) from e
    elif model_name == "prophet_xgb":
        try:
            from .models.prophet_xgb import ProphetXGBHybridModel

            cls = ProphetXGBHybridModel
        except ImportError as e:
            raise ImportError(
                "prophet_xgb could not be imported. Install Python deps with: "
                "pip install prophet xgboost (or reinstall its2s from pyproject.toml)."
            ) from e
        except Exception as e:
            if _is_xgboost_native_load_failure(e):
                raise RuntimeError(
                    "XGBoost could not load its native library (common on macOS without OpenMP). "
                    "Install OpenMP, e.g. brew install libomp  or  "
                    "conda install -c conda-forge llvm-openmp"
                ) from e
            raise
    elif model_name == "prophet_then_xgb":
        try:
            from .models.prophet_then_xgb import ProphetThenXGBModel

            cls = ProphetThenXGBModel
        except ImportError as e:
            raise ImportError(
                "prophet_then_xgb could not be imported. Install Python deps with: "
                "pip install prophet xgboost (or reinstall its2s from pyproject.toml)."
            ) from e
        except Exception as e:
            if _is_xgboost_native_load_failure(e):
                raise RuntimeError(
                    "XGBoost could not load its native library (common on macOS without OpenMP). "
                    "Install OpenMP, e.g. brew install libomp  or  "
                    "conda install -c conda-forge llvm-openmp"
                ) from e
            raise
    else:
        raise ValueError(
            "Unknown model "
            f"'{model_name}'. Available: arima, neuralprophet, prophet_xgb, prophet_then_xgb"
        )

    _MODEL_CLASS_CACHE[model_name] = cls
    return cls


def _get_model(model_name, params):
    model_cls = _get_model_class(model_name)
    return model_cls(params=params)


@dataclass
class PipelineResult:
    """Output of a single ITS pipeline run."""

    model_name: str
    fit_result: object
    bootstrap_result: object
    metrics_train: MetricsResult
    metrics_test: MetricsResult
    excess_table: ExcessResult
    config: dict


def run_single_its(
    df,
    intervention_date,
    target_col=None,
    date_col=None,
    covariate_cols=None,
    model_name="arima",
    config_path=None,
    config_overrides=None,
    output_dir=None,
    seed=42,
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
        Model to use. One of: arima, neuralprophet, prophet_xgb, prophet_then_xgb.
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

    # 2. Prepare splits
    splits = prepare_splits(
        df,
        intervention_date,
        date_col=date_col,
        test_days=config["periods"]["test_days"],
        holdout_days=config["periods"]["holdout_days"],
    )

    logger.info(
        "Splits: train=%d, test=%d, holdout=%d",
        len(splits.train_df), len(splits.test_df), len(splits.holdout_df),
    )

    # 3. Instantiate model
    model_params = get_model_config(config, model_name)
    model = _get_model(model_name, model_params)

    # 4. Fit model
    logger.info("Fitting %s model...", model_name)
    fit_result = model.fit(
        splits.train_df, target_col=target_col,
        date_col=date_col, covariate_cols=covariate_cols or None,
    )

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
            ate = calc_ate_summary(excess_table.daily_excess)
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
    )
