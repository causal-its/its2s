# Description: ARIMA model via pmdarima auto_arima.
# Usage: from its2s.models.arima import ARIMAModel
# Dependencies: pmdarima, numpy, pandas

import copy
import warnings

import numpy as np
import pandas as pd
import pmdarima as pm

from .base import BaseModel, FitResult, PredictionResult


class ARIMAModel(BaseModel):
    """ARIMA model using pmdarima's auto_arima for automatic order selection.

    On first fit, auto_arima selects the best (p,d,q) and seasonal order.
    clone_fresh() preserves the discovered order so MBB refits use the same
    model structure without repeating the expensive stepwise search.
    """

    def __init__(self, params=None, _fixed_order=None, _fixed_seasonal_order=None):
        super().__init__(params)
        self._model = None
        self._covariate_cols = None
        self._fixed_order = _fixed_order
        self._fixed_seasonal_order = _fixed_seasonal_order

    def fit(self, train_df, target_col="y", date_col="ds", covariate_cols=None):
        p = self.params
        y = train_df[target_col].values.astype(float)
        exog = train_df[covariate_cols].values if covariate_cols else None
        self._covariate_cols = covariate_cols

        # M2-8: Warn when m=7 (default) so users on non-daily series can override
        m_val = p.get("m", 7)
        if m_val == 7 and self._fixed_order is None:
            warnings.warn(
                "ARIMAModel is using m=7 (weekly seasonality period), which "
                "assumes daily data with weekly cycles. If your series is "
                "weekly, monthly, or has a different periodicity, set "
                "m to the appropriate value via config_overrides: "
                '{"models": {"arima": {"m": <period>}}}.',
                UserWarning,
                stacklevel=3,
            )

        if self._fixed_order is not None:
            # Refit with known order (fast path for MBB)
            self._model = pm.ARIMA(
                order=self._fixed_order,
                seasonal_order=self._fixed_seasonal_order,
                suppress_warnings=p.get("suppress_warnings", True),
            )
            self._model.fit(y, exogenous=exog)
        else:
            # Initial fit with order selection
            self._model = pm.auto_arima(
                y,
                exogenous=exog,
                max_p=p.get("max_p", 5),
                max_d=p.get("max_d", 2),
                max_q=p.get("max_q", 5),
                seasonal=p.get("seasonal", True),
                m=p.get("m", 7),
                stepwise=p.get("stepwise", True),
                suppress_warnings=p.get("suppress_warnings", True),
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
            _fixed_order=self._fixed_order,
            _fixed_seasonal_order=self._fixed_seasonal_order,
        )
