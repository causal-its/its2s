# Description: Abstract base class for ITS models.
# Usage: Subclass BaseModel to implement a new forecasting model.
# Dependencies: none (stdlib only)

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Both prophet 1.3.0 and neuralprophet 0.8.0 enable yearly seasonality under
# "auto" only when the training history spans at least this many days.
_YEARLY_AUTO_MIN_DAYS = 730


def warn_if_auto_yearly_disabled(df, yearly_arg):
    """Warn visibly when the library's auto rule will disable yearly seasonality.

    Prophet and NeuralProphet apply the identical rule: under "auto", the
    yearly component is enabled only when the non-null training history spans
    at least 730 days. The libraries announce the disable with a quiet log
    line; this surfaces it as a warning so the change to the user's model is
    never silent (GH #60, D-057). df must carry ds and y columns.
    """
    if yearly_arg != "auto":
        return
    dates = pd.to_datetime(df.loc[df["y"].notna(), "ds"])
    if dates.empty:
        return
    span_days = (dates.max() - dates.min()).days
    if span_days < _YEARLY_AUTO_MIN_DAYS:
        warnings.warn(
            f"yearly_seasonality='auto': the training history spans "
            f"{span_days} days, under the {_YEARLY_AUTO_MIN_DAYS}-day "
            f"(two annual cycles) minimum, so the yearly component is "
            f"disabled. Pass yearly_seasonality=True to force it."
        )


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
