# Description: NeuralProphet model with autoregression and lagged regressors.
# Usage: from its2s.models.neuralprophet import NeuralProphetModel
# Dependencies: neuralprophet, numpy, pandas

import copy

import pandas as pd
from neuralprophet import NeuralProphet, set_log_level

from .base import BaseModel, FitResult, PredictionResult
from .families import get_family
from ..frequency import resolve_frequency


class NeuralProphetModel(BaseModel):
    """NeuralProphet model with AR terms and optional lagged regressors.

    n_lags counts OBSERVATIONS at the resolved series frequency, never
    calendar days: the default 14 is a two-week AR window on daily data but a
    14-week window on a weekly series. The first n_lags training rows are AR
    warmup and produce no fitted value.

    Withheld at launch (GH #53, contract 5.7): the class serves the old path
    until P11.04.07, and a fitted model cannot be saved.
    """

    spec = get_family("neuralprophet")
    _libraries = ("neuralprophet",)

    def __init__(self, params, seasonality):
        super().__init__(params, seasonality)
        self._model = None
        self._date_col = None
        self._target_col = None

    def _build_model(self):
        p = self.params
        cycles = self.seasonality.cycles
        set_log_level("ERROR")
        model = NeuralProphet(
            n_lags=p["n_lags"],
            yearly_seasonality="annual" in cycles,
            weekly_seasonality="weekly" in cycles,
            daily_seasonality=False,
            learning_rate=p["learning_rate"],
            epochs=p["epochs"],
            batch_size=p["batch_size"],
        )
        return model

    def _prep_df(self, df, date_col, target_col, covariate_cols=None):
        """Prepare DataFrame in NeuralProphet's expected format (ds, y, ...)."""
        out = df[[date_col, target_col]].copy()
        out.columns = ["ds", "y"]
        out["ds"] = pd.to_datetime(out["ds"])
        if covariate_cols:
            for col in covariate_cols:
                out[col] = df[col].values
        return out

    def fit(self, train_df, target_col="y", date_col="ds", covariate_cols=None):
        self._covariate_cols = covariate_cols
        self._date_col = date_col
        self._target_col = target_col

        self._model = self._build_model()
        np_df = self._prep_df(train_df, date_col, target_col, covariate_cols)

        if covariate_cols:
            for col in covariate_cols:
                self._model = self._model.add_lagged_regressor(col)

        # freq is injected by the pipeline from the resolved series frequency
        # (#48, #52); standalone use resolves it from the training dates. It
        # is never read from user configuration.
        freq = self.params.get("freq")
        if freq is None:
            freq = resolve_frequency(np_df["ds"]).alias
        metrics_df = self._model.fit(np_df, freq=freq)

        fitted_df = self._model.predict(np_df)
        fitted_vals = fitted_df["yhat1"].values
        actual = np_df["y"].values
        residuals = actual - fitted_vals

        self._fit_result = FitResult(
            fitted_values=fitted_vals,
            residuals=residuals,
            model_object=self._model,
            metadata={"final_metrics": metrics_df},
        )
        return self._fit_result

    def predict(self, target_df, target_col="y", date_col="ds", covariate_cols=None):
        covariate_cols = covariate_cols or self._covariate_cols
        date_col = date_col or self._date_col
        target_col = target_col or self._target_col

        np_df = self._prep_df(target_df, date_col, target_col, covariate_cols)
        forecast = self._model.predict(np_df)

        actual = target_df[target_col].values if target_col in target_df.columns else None

        return PredictionResult(
            dates=target_df[date_col].values,
            predicted=forecast["yhat1"].values,
            actual=actual,
        )

    def clone_fresh(self):
        return NeuralProphetModel(params=copy.deepcopy(self.params),
                                  seasonality=self.seasonality)

    def seasonal_components(self):
        """The seasonalities the fitted model was configured with."""
        periods = self._model.config_seasonality.periods
        return {"neuralprophet": sorted(name for name, period in periods.items()
                                        if period.resolution > 0)}

    def save(self, directory):
        raise NotImplementedError(
            f"neuralprophet is withheld (GitHub issue {self.spec.withheld_issue}): "
            f"{self.spec.withheld_reason}. It cannot be saved.")

    @classmethod
    def load(cls, directory):
        raise NotImplementedError(
            f"neuralprophet is withheld (GitHub issue {cls.spec.withheld_issue}); "
            "there is no saved form to load.")

    # save and load are refused above, so these are never reached.
    def _save_fitted(self, directory):
        raise NotImplementedError

    def _load_fitted(self, directory):
        raise NotImplementedError
