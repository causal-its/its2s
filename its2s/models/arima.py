# Description: ARIMA model via pmdarima auto_arima.
# Usage: from its2s.models.arima import ARIMAModel
# Dependencies: pmdarima, joblib, numpy

import copy

import joblib
import numpy as np
import pmdarima as pm

from .base import BaseModel, FitResult, PredictionResult
from .families import arima_period, get_family

# auto_arima's own starting orders. It raises when a bound is below its start
# (and when max_d is 0), which tuning would score as inf and silently lose
# (spec_P11.04.02 OQ-12, F-012), so each search starts at min(start, bound).
_START = {"p": 2, "q": 2, "P": 1, "Q": 1}


class ARIMAModel(BaseModel):
    """ARIMA model using pmdarima's auto_arima for automatic order selection.

    On first fit, auto_arima selects the best (p,d,q) and seasonal order
    within the declared seasonal period m (arima_period). clone_fresh()
    preserves the discovered order and intercept setting so MBB refits use
    the same model structure without repeating the expensive stepwise
    search.
    """

    spec = get_family("arima")
    _libraries = ("pmdarima",)

    def __init__(self, params, seasonality, _fixed_order=None,
                 _fixed_seasonal_order=None, _fixed_with_intercept=None):
        super().__init__(params, seasonality)
        self._m = arima_period(seasonality)
        self._model = None
        self._fixed_order = _fixed_order
        self._fixed_seasonal_order = _fixed_seasonal_order
        self._fixed_with_intercept = _fixed_with_intercept

    def fit(self, train_df, target_col="y", date_col="ds", covariate_cols=None):
        p = self.params
        y = train_df[target_col].values.astype(float)
        exog = train_df[covariate_cols].values if covariate_cols else None
        self._covariate_cols = covariate_cols

        if self._fixed_order is not None:
            # Refit with known order (fast path for MBB). The intercept
            # setting is carried too: pm.ARIMA defaults to an intercept,
            # which auto_arima may have chosen against.
            intercept = {} if self._fixed_with_intercept is None else {
                "with_intercept": self._fixed_with_intercept}
            self._model = pm.ARIMA(
                order=self._fixed_order,
                seasonal_order=self._fixed_seasonal_order,
                suppress_warnings=p["suppress_warnings"],
                **intercept,
            )
            self._model.fit(y, X=exog)
        else:
            # Initial fit with order selection
            self._model = pm.auto_arima(
                y,
                X=exog,
                start_p=min(_START["p"], p["max_p"]),
                max_p=p["max_p"],
                d=0 if p["max_d"] == 0 else None,
                max_d=p["max_d"],
                start_q=min(_START["q"], p["max_q"]),
                max_q=p["max_q"],
                start_P=min(_START["P"], p["max_P"]),
                max_P=p["max_P"],
                D=0 if p["max_D"] == 0 else None,
                max_D=p["max_D"],
                start_Q=min(_START["Q"], p["max_Q"]),
                max_Q=p["max_Q"],
                seasonal=self._m > 1,
                m=self._m,
                stepwise=p["stepwise"],
                suppress_warnings=p["suppress_warnings"],
                error_action="ignore",
            )
            self._fixed_order = self._model.order
            self._fixed_seasonal_order = self._model.seasonal_order
            self._fixed_with_intercept = bool(self._model.with_intercept)

        fitted = self._model.predict_in_sample(X=exog)
        residuals = y - fitted

        self._fit_result = FitResult(
            fitted_values=fitted,
            residuals=residuals,
            model_object=self._model,
            metadata={"order": self._model.order, "seasonal_order": self._model.seasonal_order},
        )
        return self._fit_result

    def predict(self, target_df, target_col="y", date_col="ds", covariate_cols=None):
        covariate_cols = covariate_cols or self._covariate_cols
        n_periods = len(target_df)
        exog = target_df[covariate_cols].values if covariate_cols else None

        preds = self._model.predict(n_periods=n_periods, X=exog)

        actual = target_df[target_col].values if target_col in target_df.columns else None

        return PredictionResult(
            dates=target_df[date_col].values,
            predicted=np.array(preds),
            actual=actual,
        )

    def startup_rows(self):
        """d + D*m: a differenced model has no in-sample prediction for its
        first d + D*m rows (pmdarima returns about 0 there), so their
        residuals are about the observations themselves."""
        if self._fixed_order is None:
            raise ValueError("startup_rows needs a fitted or loaded model.")
        d = self._fixed_order[1]
        _, D, _, m = self._fixed_seasonal_order
        return int(d + D * m)

    def clone_fresh(self):
        return ARIMAModel(
            params=copy.deepcopy(self.params),
            seasonality=self.seasonality,
            _fixed_order=self._fixed_order,
            _fixed_seasonal_order=self._fixed_seasonal_order,
            _fixed_with_intercept=self._fixed_with_intercept,
        )

    def seasonal_components(self):
        """The declared m and whether the seasonal search ran (OQ-5); the
        fitted seasonal order is an estimate, reported but not held equal."""
        return {"m": self._m, "seasonal_search": self._m > 1,
                "fitted_seasonal_order": (tuple(self._model.seasonal_order)
                                          if self._model is not None else None)}

    def _save_fitted(self, directory):
        # pmdarima has no native format; joblib is its documented route.
        joblib.dump(self._model, directory / "arima.joblib")

    def _load_fitted(self, directory):
        self._model = joblib.load(directory / "arima.joblib")
        self._fixed_order = self._model.order
        self._fixed_seasonal_order = self._model.seasonal_order
        self._fixed_with_intercept = bool(self._model.with_intercept)
        self._fit_result = None
