# Description: Abstract base class for ITS models.
# Usage: Subclass BaseModel to implement a new forecasting model.
# Dependencies: none (stdlib only)

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FitResult:
    """Output of model fitting."""

    fitted_values: np.ndarray
    residuals: np.ndarray
    model_object: Any = None
    metadata: dict = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Output of model prediction."""

    dates: np.ndarray
    predicted: np.ndarray
    actual: np.ndarray | None = None


class BaseModel(ABC):
    """Abstract base for all ITS forecasting models."""

    def __init__(self, params: dict | None = None):
        self.params = params or {}
        self._fit_result: FitResult | None = None

    @property
    def warmup_rows(self) -> int:
        """Number of leading training rows with undefined fitted values.

        Autoregressive models (e.g. NeuralProphet with ``n_lags``) cannot produce
        fitted values for the first few rows, leaving NaN in ``fitted_values`` and
        ``residuals``. Consumers (e.g. the moving block bootstrap) use this to
        exclude the warmup segment from residual resampling. Defaults to 0 for
        models that fit every row.
        """
        return 0

    @abstractmethod
    def fit(self, train_df: pd.DataFrame, target_col: str = "y",
            date_col: str = "ds", covariate_cols: list[str] | None = None) -> FitResult:
        """Fit model on training data."""

    @abstractmethod
    def predict(self, target_df: pd.DataFrame, target_col: str = "y",
                date_col: str = "ds", covariate_cols: list[str] | None = None) -> PredictionResult:
        """Generate predictions for target dates."""

    @abstractmethod
    def clone_fresh(self) -> "BaseModel":
        """Return an unfitted copy of this model with the same params."""
