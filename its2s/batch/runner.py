# Description: Parallel batch execution over multiple series.
# Usage: from its2s.batch.runner import run_batch
# Dependencies: joblib

import logging
from datetime import datetime
from pathlib import Path

from joblib import Parallel, delayed

from ..pipeline import PipelineResult, run_single_its
from .seed_manager import derive_seed

logger = logging.getLogger(__name__)


def _make_run_dir(output_dir, n_sim):
    """Create a versioned output directory."""
    base = Path(output_dir)
    date_str = datetime.now().strftime("%Y-%m-%d")
    version = 1
    while True:
        run_dir = base / f"run_{date_str}.v{version:03d}_sim{n_sim}"
        if not run_dir.exists():
            break
        version += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _run_one_series(series_spec, config_path, series_output_dir, seed):
    """Run a single series through the pipeline."""
    series_seed = derive_seed(seed, series_spec["series_id"])

    return run_single_its(
        df=series_spec["df"],
        intervention_date=series_spec["intervention_date"],
        target_col=series_spec.get("target_col"),
        date_col=series_spec.get("date_col"),
        covariate_cols=series_spec.get("covariate_cols"),
        model_name=series_spec.get("model_name", "arima"),
        config_path=config_path,
        config_overrides=series_spec.get("config_overrides"),
        output_dir=series_output_dir,
        seed=series_seed,
    )


def run_batch(series_list, config_path=None, output_dir="output",
              n_jobs=1, seed=42):
    """Run ITS pipeline on multiple series.

    Parameters
    ----------
    series_list : list[dict]
        Each dict has: series_id, df, intervention_date, target_col,
        date_col, covariate_cols, model_name (optional), config_overrides (optional).
    config_path : str or Path, optional
        Path to shared YAML config.
    output_dir : str or Path
        Base output directory.
    n_jobs : int
        Number of parallel jobs. 1 = sequential.
    seed : int
        Global seed for reproducibility.

    Returns
    -------
    list[PipelineResult]
    """
    from ..settings import load_config
    config = load_config(config_path)
    n_sim = config["bootstrap"]["n_sim"]

    run_dir = _make_run_dir(output_dir, n_sim)
    logger.info("Batch output directory: %s", run_dir)

    def _process(spec):
        series_dir = run_dir / spec["series_id"]
        series_dir.mkdir(parents=True, exist_ok=True)
        return _run_one_series(spec, config_path, series_dir, seed)

    if n_jobs == 1:
        results = []
        for spec in series_list:
            logger.info("Processing series: %s", spec["series_id"])
            results.append(_process(spec))
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process)(spec) for spec in series_list
        )

    logger.info("Batch complete: %d series processed.", len(results))
    return results
