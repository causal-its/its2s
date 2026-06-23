# Description: Moving Block Bootstrap for counterfactual CIs.
# Usage: from its2s.bootstrap.mbb import MovingBlockBootstrap
# Dependencies: numpy, joblib

import logging
import math
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .base import BaseBootstrap, BootstrapCIResult

logger = logging.getLogger(__name__)


def _resample_blocks(residuals, block_length, rng, n_out=None):
    """Resample residuals using moving blocks.

    Parameters
    ----------
    residuals : np.ndarray
        Residuals to draw blocks from (must be free of NaN).
    block_length : int
        Length of each block.
    rng : np.random.Generator
        Random number generator.
    n_out : int, optional
        Length of the resampled series to return. Defaults to ``len(residuals)``.
        Used to produce a series matching the non-warmup portion of the training
        set when blocks are drawn from a shorter (warmup-excluded) residual vector.

    Returns
    -------
    np.ndarray
        Resampled residuals of length ``n_out``.
    """
    n = len(residuals)
    if n_out is None:
        n_out = n
    n_blocks = math.ceil(n_out / block_length)
    max_start = n - block_length

    if max_start < 1:
        return rng.choice(residuals, size=n_out, replace=True)

    starts = rng.integers(0, max_start + 1, size=n_blocks)
    blocks = [residuals[s : s + block_length] for s in starts]
    resampled = np.concatenate(blocks)[:n_out]
    return resampled


def _single_mbb_sim(sim_idx, model, train_df, target_df, fitted_values,
                     resid_finite, warmup, block_length, target_col, date_col,
                     covariate_cols, seed):
    """Run a single MBB simulation.

    ``resid_finite`` holds the warmup-excluded, NaN-free residuals to resample
    from; ``warmup`` is the number of leading rows whose fitted values are
    undefined (e.g. AR warmup). Those rows are held at the observed target so the
    refit never receives NaN; the rest are perturbed by resampled residuals.
    """
    rng = np.random.default_rng(seed)

    n = len(fitted_values)

    # Resample residuals for the non-warmup rows and create perturbed series
    resampled_resid = _resample_blocks(resid_finite, block_length, rng,
                                       n_out=n - warmup)
    perturbed_y = np.asarray(fitted_values, dtype=float).copy()
    if warmup > 0:
        # Warmup rows have NaN fitted values; hold them at the observed target.
        obs_y = train_df[target_col].to_numpy(dtype=float)
        perturbed_y[:warmup] = obs_y[:warmup]
    perturbed_y[warmup:] = np.asarray(fitted_values[warmup:], dtype=float) + resampled_resid

    # Build perturbed training DataFrame
    perturbed_train = train_df.copy()
    perturbed_train[target_col] = perturbed_y

    # Fit fresh model on perturbed data
    fresh_model = model.clone_fresh()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh_model.fit(perturbed_train, target_col=target_col,
                        date_col=date_col, covariate_cols=covariate_cols)
        pred_result = fresh_model.predict(target_df, target_col=target_col,
                                          date_col=date_col,
                                          covariate_cols=covariate_cols)
    return pred_result.predicted


class MovingBlockBootstrap(BaseBootstrap):
    """Moving Block Bootstrap for generating counterfactual prediction CIs."""

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

        # Exclude the warmup segment (NaN fitted values, e.g. AR warmup) from the
        # residuals we resample. Drop any remaining NaN defensively. The bootstrap
        # then perturbs only the non-warmup rows; warmup rows are held at observed y.
        warmup = int(getattr(model, "warmup_rows", 0))
        resid_finite = np.asarray(residuals[warmup:], dtype=float)
        resid_finite = resid_finite[np.isfinite(resid_finite)]
        if resid_finite.size == 0:
            raise ValueError(
                "No finite residuals available for bootstrap resampling after "
                "excluding the warmup segment."
            )

        # Point prediction from original model
        point_pred = model.predict(target_df, target_col=target_col,
                                   date_col=date_col,
                                   covariate_cols=covariate_cols)

        base_rng = np.random.default_rng(seed)
        sim_seeds = base_rng.integers(0, 2**31, size=self.n_sim)

        n_dates = len(target_df)
        pred_matrix = np.full((n_dates, self.n_sim), np.nan)
        n_successful = 0

        log_interval = max(1, self.n_sim // 10)

        try:
            results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(_single_mbb_sim)(
                    i, model, train_df, target_df, fitted_values, resid_finite,
                    warmup, self.block_length, target_col, date_col, covariate_cols,
                    int(sim_seeds[i])
                )
                for i in range(self.n_sim)
            )
            for i, preds in enumerate(results):
                if preds is not None and len(preds) == n_dates:
                    pred_matrix[:, i] = preds
                    n_successful += 1
                if (i + 1) % log_interval == 0:
                    logger.info(
                        "MBB progress: %d / %d simulations processed.",
                        i + 1, self.n_sim,
                    )
        except Exception:
            warnings.warn(
                "Parallel MBB execution failed; falling back to sequential.",
                UserWarning,
                stacklevel=2,
            )
            for i in range(self.n_sim):
                try:
                    preds = _single_mbb_sim(
                        i, model, train_df, target_df, fitted_values,
                        resid_finite, warmup, self.block_length, target_col,
                        date_col, covariate_cols, int(sim_seeds[i])
                    )
                    if preds is not None and len(preds) == n_dates:
                        pred_matrix[:, i] = preds
                        n_successful += 1
                except Exception:
                    continue
                if (i + 1) % log_interval == 0:
                    logger.info(
                        "MBB progress: %d / %d simulations processed "
                        "(%d successful).",
                        i + 1, self.n_sim, n_successful,
                    )

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
