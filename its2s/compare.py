# Description: Compare multiple models on the same dataset.
# Usage: from its2s.compare import compare_models
# Dependencies: pandas

import logging
import warnings

import pandas as pd

from .metrics.excess import calc_ate_summary

logger = logging.getLogger(__name__)


def compare_models(df, intervention_date, model_names=None,
                   target_col=None, date_col=None, covariate_cols=None,
                   config_path=None, config_overrides=None,
                   output_dir=None, seed=42):
    """Run the ITS pipeline with multiple models and return a comparison table.

    Parameters
    ----------
    df : pd.DataFrame
        Full time series dataset.
    intervention_date : str or pd.Timestamp
        Date of the intervention.
    model_names : list[str], optional
        Models to compare. Defaults to all available models.
    target_col : str, optional
    date_col : str, optional
    covariate_cols : list[str], optional
    config_path : str or Path, optional
    config_overrides : dict, optional
    output_dir : str or Path, optional
    seed : int

    Returns
    -------
    tuple[pd.DataFrame, dict[str, PipelineResult]]
        (comparison_table, results_dict)
    """
    from .pipeline import run_single_its, _get_available_model_names

    if model_names is None:
        model_names = _get_available_model_names()

    results = {}
    rows = []

    for model_name in model_names:
        logger.info("Running model: %s", model_name)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = run_single_its(
                    df, intervention_date,
                    target_col=target_col,
                    date_col=date_col,
                    covariate_cols=covariate_cols,
                    model_name=model_name,
                    config_path=config_path,
                    config_overrides=config_overrides,
                    output_dir=output_dir,
                    seed=seed,
                )
            results[model_name] = result

            row = {
                "model": model_name,
                "train_rmse": result.metrics_train.rmse,
                "train_r2": result.metrics_train.r2,
                "test_rmse": result.metrics_test.rmse,
                "test_mae": result.metrics_test.mae,
                "test_mape": result.metrics_test.mape,
                "test_r2": result.metrics_test.r2,
                "bootstrap_n_successful": result.bootstrap_result.n_successful,
            }

            if not result.excess_table.obs_excess.empty:
                ate = calc_ate_summary(result.excess_table)
                total_row = ate[ate["metric"] == "Total ATE"].iloc[0]
                per_obs_row = ate[ate["metric"] == "Mean ATE per obs"].iloc[0]
                row["total_ate"] = total_row["estimate"]
                row["total_ate_ci_lo"] = total_row["ci_lo"]
                row["total_ate_ci_hi"] = total_row["ci_hi"]
                row["mean_ate_per_obs"] = per_obs_row["estimate"]

            rows.append(row)

        except Exception as e:
            logger.error("Model '%s' failed: %s", model_name, e)
            rows.append({"model": model_name, "error": str(e)})

    comparison = pd.DataFrame(rows)
    return comparison, results
