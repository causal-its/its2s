# Description: ARIMA model via pmdarima auto_arima.
# Usage: from its2s.models.arima import ARIMAModel
# Dependencies: pmdarima, joblib, numpy

import copy
import warnings

import joblib
import numpy as np
import pmdarima as pm

from .base import BaseModel, FitResult, PredictionResult
from .families import get_family

# Relative rounding of a declared period beyond which m is reported as far
# from an integer (spec_P11.04.02 OQ-3): 52.18 -> 52 moves m by 0.35%.
_M_ROUNDING_WARN = 0.01

# The (cycle, m) pairs ARIMA can honour (OQ-3): weekly on daily data; annual
# on weekly, monthly or quarterly data.
_HONOURED = {("weekly", 7), ("annual", 52), ("annual", 12), ("annual", 4)}

# auto_arima's own starting orders. It raises when a bound is below its start
# (and when max_d is 0), which tuning would score as inf and silently lose
# (spec_P11.04.02 OQ-12, F-012), so each search starts at min(start, bound).
_START = {"p": 2, "q": 2}


def declared_m(seasonality):
    """The integer seasonal period ARIMA carries for a declaration (OQ-3).

    ARIMA holds at most one seasonal period, and auto_arima cannot fit a
    period in the hundreds, so it accepts exactly: no cycle (m=1); weekly on
    daily data (7); annual on weekly (52), monthly (12) or quarterly (4)
    data. Anything else raises, naming prophet_xgb.
    """
    cycles = seasonality.cycles
    if not cycles:
        return 1
    if len(cycles) > 1:
        raise ValueError(
            f"arima carries one seasonal period; {list(cycles)} are declared. "
            "Declare one cycle, or use prophet_xgb, which carries both.")
    period = seasonality.periods[cycles[0]]
    m = int(round(period))
    if (cycles[0], m) not in _HONOURED:
        raise ValueError(
            f"arima cannot carry the {cycles[0]} cycle on "
            f"{seasonality.freq.alias!r} data (m={m}). Use prophet_xgb, or "
            "aggregate the series to a coarser interval.")
    if abs(m - period) / period > _M_ROUNDING_WARN:
        warnings.warn(
            f"The {cycles[0]} cycle spans {period:.2f} observations on "
            f"{seasonality.freq.alias!r} data; arima rounds it to m={m}.",
            UserWarning, stacklevel=3)
    return m


class ARIMAModel(BaseModel):
    """ARIMA model using pmdarima's auto_arima for automatic order selection.

    On first fit, auto_arima selects the best (p,d,q) and seasonal order
    within the declared seasonal period m (declared_m). clone_fresh()
    preserves the discovered order so MBB refits use the same model
    structure without repeating the expensive stepwise search.
    """

    spec = get_family("arima")
    _libraries = ("pmdarima",)

    def __init__(self, params, seasonality, _fixed_order=None,
                 _fixed_seasonal_order=None):
        super().__init__(params, seasonality)
        self._m = declared_m(seasonality)
        self._model = None
        self._fixed_order = _fixed_order
        self._fixed_seasonal_order = _fixed_seasonal_order

    def fit(self, train_df, target_col="y", date_col="ds", covariate_cols=None):
        p = self.params
        y = train_df[target_col].values.astype(float)
        exog = train_df[covariate_cols].values if covariate_cols else None
        self._covariate_cols = covariate_cols

        if self._fixed_order is not None:
            # Refit with known order (fast path for MBB)
            self._model = pm.ARIMA(
                order=self._fixed_order,
                seasonal_order=self._fixed_seasonal_order,
                suppress_warnings=p["suppress_warnings"],
            )
            self._model.fit(y, exogenous=exog)
        else:
            # Initial fit with order selection
            self._model = pm.auto_arima(
                y,
                exogenous=exog,
                start_p=min(_START["p"], p["max_p"]),
                max_p=p["max_p"],
                d=0 if p["max_d"] == 0 else None,
                max_d=p["max_d"],
                start_q=min(_START["q"], p["max_q"]),
                max_q=p["max_q"],
                seasonal=self._m > 1,
                m=self._m,
                stepwise=p["stepwise"],
                suppress_warnings=p["suppress_warnings"],
                error_action="ignore",
            )
            self._fixed_order = self._model.order
            self._fixed_seasonal_order = self._model.seasonal_order

        fitted = self._model.predict_in_sample(exogenous=exog)
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

        preds = self._model.predict(n_periods=n_periods, exogenous=exog)

        actual = target_df[target_col].values if target_col in target_df.columns else None

        return PredictionResult(
            dates=target_df[date_col].values,
            predicted=np.array(preds),
            actual=actual,
        )

    def clone_fresh(self):
        return ARIMAModel(
            params=copy.deepcopy(self.params),
            seasonality=self.seasonality,
            _fixed_order=self._fixed_order,
            _fixed_seasonal_order=self._fixed_seasonal_order,
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
        self._fit_result = None
