# Description: Abstract base class for ITS models.
# Usage: Subclass BaseModel to implement a new forecasting model.
# Dependencies: none (stdlib only)

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from ..seasonality import Seasonality
from ..settings import _deep_merge
from .families import FamilySpec

_FAMILY_FILE = "family.json"


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


def _versions(packages):
    out = {}
    for name in ("its2s", *packages):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


class BaseModel(ABC):
    """Abstract base for all ITS forecasting models.

    A family is one subclass. Its spec (name, defaults, search space) lives in
    its2s.models.families. Every fit carries the seasonality it was built
    with: there is no default and no "auto" (contract 5.4, GH #74).
    """

    spec: ClassVar[FamilySpec]
    # Library distributions whose versions are recorded beside a saved model.
    _libraries: ClassVar[tuple] = ()

    def __init__(self, params: dict | None, seasonality: Seasonality):
        if not isinstance(seasonality, Seasonality):
            raise TypeError(
                f"{type(self).__name__} needs a Seasonality, got "
                f"{type(seasonality).__name__}; build one with "
                "its2s.seasonality.make_seasonality.")
        self.params = _deep_merge(self.spec.defaults, params or {})
        self.seasonality = seasonality
        self._fit_result: FitResult | None = None
        self._covariate_cols = None

    @abstractmethod
    def fit(self, train_df, target_col="y", date_col="ds",
            covariate_cols=None) -> FitResult:
        """Fit model on training data."""

    @abstractmethod
    def predict(self, target_df, target_col="y", date_col="ds",
                covariate_cols=None) -> PredictionResult:
        """Generate predictions for target dates."""

    @abstractmethod
    def clone_fresh(self) -> "BaseModel":
        """Return an unfitted copy of this model with the same params."""

    @abstractmethod
    def seasonal_components(self) -> dict:
        """The seasonal structure this fit actually carries (P11.04.V-8)."""

    @abstractmethod
    def _save_fitted(self, directory: Path) -> None:
        """Write the fitted library objects into directory."""

    @abstractmethod
    def _load_fitted(self, directory: Path) -> None:
        """Read back what _save_fitted wrote."""

    def save(self, directory) -> None:
        """Save the fitted model and a family.json describing it."""
        if self._fit_result is None:
            raise RuntimeError(f"{self.spec.name}: fit the model before saving it.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self._save_fitted(directory)
        record = {"family": self.spec.name, "params": self.params,
                  "seasonality": self.seasonality.to_dict(),
                  "covariate_cols": self._covariate_cols,
                  "versions": _versions(self._libraries)}
        (directory / _FAMILY_FILE).write_text(json.dumps(record, indent=2))

    @classmethod
    def load(cls, directory) -> "BaseModel":
        """Load a model saved by save(); a different family's save raises."""
        directory = Path(directory)
        record = json.loads((directory / _FAMILY_FILE).read_text())
        if record["family"] != cls.spec.name:
            raise ValueError(
                f"{directory} holds a {record['family']!r} model, not "
                f"{cls.spec.name!r}.")
        model = cls(params=record["params"],
                    seasonality=Seasonality.from_dict(record["seasonality"]))
        model._covariate_cols = record["covariate_cols"]
        model._load_fitted(directory)
        return model
