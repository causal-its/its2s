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


def check_block_length(n_pool, block_length):
    """Refuse a block length that leaves no room to move a block.

    A block must be shorter than the residual pool it is drawn from, or
    every draw would be the same single block. There is no fallback to
    resampling single residuals: that would silently change the method.
    """
    if block_length >= n_pool:
        raise ValueError(
            f"block_length {block_length} must be shorter than the residual "
            f"pool ({n_pool} observations).")


def _resample_blocks(residuals, block_length, rng, size=None):
    """Resample residuals using moving blocks.

    Parameters
    ----------
    residuals : np.ndarray
        Original residuals from the fitted model.
    block_length : int
        Length of each block, in observations (rows), never calendar days.
    rng : np.random.Generator
        Random number generator.
    size : int or None
        Length of the result. None (the default) gives a result as long as
        residuals. Whole blocks are drawn and the last one is truncated.

    Returns
    -------
    np.ndarray
        Resampled residuals of length size.

    Raises
    ------
    ValueError
        If block_length is not shorter than residuals (check_block_length).
    """
    n = len(residuals)
    check_block_length(n, block_length)
    if size is None:
        size = n
    n_blocks = math.ceil(size / block_length)
    max_start = n - block_length

    starts = rng.integers(0, max_start + 1, size=n_blocks)
    blocks = [residuals[s : s + block_length] for s in starts]
    resampled = np.concatenate(blocks)[:size]
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
    forecast of the target rows; path is the replication's path: the mean
    path itself when no noise is drawn, the mean path plus the noise draw
    when it is. On failure both paths are None and
    failure names the kind: "raised" (the refit or forecast raised; detail
    holds the exception), "wrong_length" (no forecast, or one whose length
    differs from the target rows) or "non_finite" (a forecast holding NaN or
    an infinite value). A failed replication contributes no path.
    """

    index: int
    mean_path: np.ndarray | None = None
    path: np.ndarray | None = None
    failure: str | None = None
    detail: str | None = None

    @property
    def ok(self):
        return self.failure is None


def residual_pool(residuals, startup_rows):
    """The residuals a replication resamples: those after the start-up rows,
    centred on their mean, so resampling them does not shift every perturbed
    series by that mean."""
    pool = np.asarray(residuals, dtype=float)[startup_rows:]
    return pool - pool.mean()


def replicate(r, model, fit_df, target_df, fitted, pool, block_length,
              seed_r, cols, startup_rows=0, noise=False):
    """Run bootstrap replication r and return its Replication outcome.

    Builds the perturbed series: the first startup_rows rows keep their
    observed values, every later row is fitted + block-resampled residuals
    drawn from pool (residual_pool). Then refits a fresh copy of the model
    on it and forecasts target_df: the mean path.

    With noise=True the replication also draws one residual per target row,
    in blocks of block_length from the same pool and the same random stream,
    after the perturbation's draws, and adds them to the mean path. Because
    the noise is drawn last, a run with and a run without noise under the
    same seed share every mean path. The noise is added to the forecast as
    it stands; it is not passed through the model's dynamics. This makes an
    interim prediction interval; it has not been shown to reach its stated
    coverage.

    Every random draw comes from seed_r alone, so replication r's outcome
    does not depend on which other replications run or in what order. Never
    raises for a failed refit or forecast: the failure is returned, named.
    """
    rng = np.random.default_rng(seed_r)
    perturbed = fit_df.copy()
    k = startup_rows
    observed = fit_df[cols.target].to_numpy(dtype=float)
    perturbed[cols.target] = np.concatenate([
        observed[:k],
        np.asarray(fitted, dtype=float)[k:] + _resample_blocks(pool, block_length, rng),
    ])
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
    mean_path = np.asarray(mean_path, dtype=float)
    if not np.all(np.isfinite(mean_path)):
        return Replication(r, failure="non_finite")
    if not noise:
        return Replication(r, mean_path=mean_path, path=mean_path)
    noise_r = _resample_blocks(pool, block_length, rng, size=len(mean_path))
    return Replication(r, mean_path=mean_path, path=mean_path + noise_r)


def _failure_counts(outcomes):
    """'n kind' for each failure kind present, e.g. '50 non_finite'."""
    counts = {}
    for o in outcomes:
        if not o.ok:
            counts[o.failure] = counts.get(o.failure, 0) + 1
    return ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items()))


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
        startup = model.startup_rows()
        pool = residual_pool(model._fit_result.residuals, startup)
        # Checked once here, so a bad length is one clear error rather than a
        # failure inside every replication.
        check_block_length(len(pool), self.block_length)

        # Point prediction from original model
        point_pred = model.predict(target_df, target_col=target_col,
                                   date_col=date_col,
                                   covariate_cols=covariate_cols)

        base_rng = np.random.default_rng(seed)
        sim_seeds = base_rng.integers(0, 2**31, size=self.n_sim)
        cols = ReplicateColumns(target_col, date_col, covariate_cols)

        def run(i):
            return replicate(i, model, train_df, target_df, fitted_values,
                             pool, self.block_length, int(sim_seeds[i]), cols,
                             startup_rows=startup)

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

        # A failed replication contributes no column: the matrix holds the
        # successful paths only, in replication order.
        paths = [o.mean_path for o in outcomes if o.ok]
        n_successful = len(paths)
        if n_successful == 0:
            raise ValueError(
                f"All {self.n_sim} bootstrap replications failed "
                f"({_failure_counts(outcomes)}); no interval can be formed.")
        pred_matrix = np.column_stack(paths)
        log_interval = max(1, self.n_sim // 10)
        for i in range(self.n_sim):
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
