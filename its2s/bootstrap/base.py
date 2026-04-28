# Description: Abstract base class for bootstrap CI generation.
# Usage: Subclass BaseBootstrap to implement a bootstrap method.
# Dependencies: numpy

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class BootstrapCIResult:
    """Output of bootstrap confidence interval generation."""

    dates: np.ndarray
    actual: np.ndarray | None
    predicted: np.ndarray
    conf_lo: np.ndarray
    conf_hi: np.ndarray
    pred_matrix: np.ndarray
    n_successful: int
    ci_method: str
    ci_level: float


class BaseBootstrap(ABC):
    """Abstract base for bootstrap CI methods."""

    @abstractmethod
    def generate_cis(self, model, train_df, target_df, target_col="y",
                     date_col="ds", covariate_cols=None, seed=None) -> BootstrapCIResult:
        """Generate bootstrap confidence intervals for predictions."""

    @staticmethod
    def calculate_ci(pred_matrix, point_est, method="quantile", level=0.95):
        """Calculate confidence intervals from a prediction matrix.

        Parameters
        ----------
        pred_matrix : np.ndarray
            Shape (n_dates, n_sims).
        point_est : np.ndarray
            Point predictions from the original model.
        method : str
            "quantile" or "symmetric_sd".
        level : float
            Confidence level (e.g. 0.95).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            (conf_lo, conf_hi) arrays.
        """
        alpha = 1 - level
        if method == "quantile":
            conf_lo = np.nanpercentile(pred_matrix, 100 * alpha / 2, axis=1)
            conf_hi = np.nanpercentile(pred_matrix, 100 * (1 - alpha / 2), axis=1)
        elif method == "symmetric_sd":
            sd = np.nanstd(pred_matrix, axis=1)
            from scipy.stats import norm
            z = norm.ppf(1 - alpha / 2)
            conf_lo = point_est - z * sd
            conf_hi = point_est + z * sd
        else:
            raise ValueError(f"Unknown CI method: {method}")
        return conf_lo, conf_hi
