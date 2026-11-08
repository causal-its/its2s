# Description: Moving Block Bootstrap for counterfactual CIs.
# Usage: from its2s.bootstrap.mbb import MovingBlockBootstrap
# Dependencies: numpy, joblib

import logging
import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .base import BaseBootstrap, BootstrapCIResult

logger = logging.getLogger(__name__)


def _resample_blocks(residuals, block_length, rng):
    """Resample residuals using moving blocks.

    Parameters
    ----------
    residuals : np.ndarray
        Original residuals from the fitted model.
    block_length : int
        Length of each block, in observations (rows), never calendar days.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Resampled residuals of the same length as input.
    """
    n = len(residuals)
    n_blocks = math.ceil(n / block_length)
    max_start = n - block_length

    if max_start < 1:
        return rng.choice(residuals, size=n, replace=True)

    starts = rng.integers(0, max_start + 1, size=n_blocks)
    blocks = [residuals[s : s + block_length] for s in starts]
    resampled = np.concatenate(blocks)[:n]
    return resampled


@dataclass(frozen=True)
class ReplicateColumns:
    """The column names one replication fits and forecasts with."""

    target: str = "y"
    date: str = "ds"
    covariates: list | None = None


@dataclass(frozen=True)
class Replication:
    """One replication's outcome: its paths, or the kind of failure.

    On success failure is None and mean_path is the refitted model's
    forecast of the target rows; path is the replication's path, the mean
    path itself while no noise is drawn. On failure both paths are None and
    failure names the kind: "raised" (the refit or forecast raised; detail
    holds the exception) or "wrong_length" (no forecast, or one whose length
    differs from the target rows).
    """

    index: int
    mean_path: np.ndarray | None = None
    path: np.ndarray | None = None
    failure: str | None = None
    detail: str | None = None

    @property
    def ok(self):
        return self.failure is None


def replicate(r, model, fit_df, target_df, fitted, residuals, block_length,
              seed_r, cols):
    """Run bootstrap replication r and return its Replication outcome.

    Builds the perturbed series fitted + block-resampled residuals, refits
    a fresh copy of the model on it and forecasts target_df. Every random
    draw comes from seed_r alone, so replication r's outcome does not
    depend on which other replications run or in what order. Never raises
    for a failed refit or forecast: the failure is returned, named.
    """
    rng = np.random.default_rng(seed_r)
    perturbed = fit_df.copy()
    perturbed[cols.target] = fitted + _resample_blocks(residuals, block_length, rng)
    try:
        fresh_model = model.clone_fresh()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fresh_model.fit(perturbed, target_col=cols.target,
                            date_col=cols.date, covariate_cols=cols.covariates)
            mean_path = fresh_model.predict(
                target_df, target_col=cols.target, date_col=cols.date,
                covariate_cols=cols.covariates).predicted
    except Exception as exc:
        return Replication(r, failure="raised", detail=repr(exc))
    if mean_path is None or len(mean_path) != len(target_df):
        return Replication(r, failure="wrong_length")
    return Replication(r, mean_path=mean_path, path=mean_path)


class MovingBlockBootstrap(BaseBootstrap):
    """Moving Block Bootstrap for generating counterfactual prediction CIs.

    block_length is measured in observations (rows of the regular series),
    never calendar days: the default 14, calibrated on daily data, spans 14
    weeks on a weekly series.
    """

    def __init__(self, n_sim=1000, block_length=14, ci_method="quantile",
                 ci_level=0.95, n_jobs=1):
        self.n_sim = n_sim
        self.block_length = block_length
        self.ci_method = ci_method
        self.ci_level = ci_level
        self.n_jobs = n_jobs

    def generate_cis(self, model, train_df, target_df, target_col="y",
                     date_col="ds", covariate_cols=None, seed=None):
        if model._fit_result is None:
            raise ValueError("Model must be fitted before generating bootstrap CIs.")

        fitted_values = model._fit_result.fitted_values
        residuals = model._fit_result.residuals

        # Point prediction from original model
        point_pred = model.predict(target_df, target_col=target_col,
                                   date_col=date_col,
                                   covariate_cols=covariate_cols)

        base_rng = np.random.default_rng(seed)
        sim_seeds = base_rng.integers(0, 2**31, size=self.n_sim)
        cols = ReplicateColumns(target_col, date_col, covariate_cols)

        def run(i):
            return replicate(i, model, train_df, target_df, fitted_values,
                             residuals, self.block_length, int(sim_seeds[i]), cols)

        try:
            outcomes = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(run)(i) for i in range(self.n_sim)
            )
        except Exception:
            warnings.warn(
                "Parallel MBB execution failed; falling back to sequential.",
                UserWarning,
                stacklevel=2,
            )
            outcomes = [run(i) for i in range(self.n_sim)]

        # A failed replication leaves its column NaN.
        pred_matrix = np.full((len(target_df), self.n_sim), np.nan)
        n_successful = 0
        log_interval = max(1, self.n_sim // 10)
        for i, outcome in enumerate(outcomes):
            if outcome.ok:
                pred_matrix[:, i] = outcome.mean_path
                n_successful += 1
            if (i + 1) % log_interval == 0:
                logger.info("MBB progress: %d / %d simulations processed.",
                            i + 1, self.n_sim)

        logger.info("MBB: %d / %d simulations successful.", n_successful, self.n_sim)

        if n_successful < self.n_sim // 2:
            warnings.warn(
                f"Only {n_successful} / {self.n_sim} bootstrap simulations succeeded. "
                "CIs may be unreliable.",
                UserWarning,
                stacklevel=2,
            )

        conf_lo, conf_hi = self.calculate_ci(
            pred_matrix, point_pred.predicted, self.ci_method, self.ci_level
        )

        actual = target_df[target_col].values if target_col in target_df.columns else None

        return BootstrapCIResult(
            dates=target_df[date_col].values,
            actual=actual,
            predicted=point_pred.predicted,
            conf_lo=conf_lo,
            conf_hi=conf_hi,
            pred_matrix=pred_matrix,
            n_successful=n_successful,
            ci_method=self.ci_method,
            ci_level=self.ci_level,
        )
