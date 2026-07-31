# Description: Sequential Prophet-then-XGBoost -- Prophet forecasts, XGB corrects errors.
# Usage: from its2s.models.prophet_then_xgb import ProphetThenXGBModel
# Dependencies: prophet, xgboost, numpy, pandas

import numpy as np
import pandas as pd
from prophet import Prophet
from xgboost import XGBRegressor

from .base import BaseModel, FitResult, PredictionResult
from .utils import make_time_features as _make_time_features


class ProphetThenXGBModel(BaseModel):
    """Sequential: Prophet makes a standalone forecast, XGB corrects its errors.

    Distinct from the hybrid approach: Prophet generates its own complete
    forecast first, then XGB is trained on Prophet's forecast errors
    (actual - prophet_forecast) using covariates and time features.
    Final prediction = prophet_forecast + xgb_correction.
    """

    def __init__(self, params=None):
        super().__init__(params)
        self._prophet = None
        self._xgb = None
        self._covariate_cols = None

    def fit(self, train_df, target_col="y", date_col="ds", covariate_cols=None):
        self._covariate_cols = covariate_cols
        p_params = self.params.get("prophet", {})
        x_params = self.params.get("xgb", {})

        # Step 1: Fit Prophet standalone
        prophet_df = train_df[[date_col, target_col]].copy()
        prophet_df.columns = ["ds", "y"]
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])

        self._prophet = Prophet(
            yearly_seasonality=p_params.get("yearly_seasonality", True),
            weekly_seasonality=p_params.get("weekly_seasonality", "auto"),
            daily_seasonality=p_params.get("daily_seasonality", False),
            changepoint_prior_scale=p_params.get("changepoint_prior_scale", 0.05),
        )
        self._prophet.fit(prophet_df)

        # Step 2: Prophet's in-sample forecast
        prophet_forecast = self._prophet.predict(prophet_df)["yhat"].values

        # Step 3: Prophet's forecast errors
        y = train_df[target_col].values.astype(float)
        prophet_errors = y - prophet_forecast

        # Step 4: Fit XGB on Prophet's errors
        time_feats = _make_time_features(train_df, date_col)
        xgb_features = time_feats.copy()
        if covariate_cols:
            for col in covariate_cols:
                xgb_features[col] = train_df[col].values

        # Include Prophet's forecast as a feature for the correction model
        xgb_features["prophet_forecast"] = prophet_forecast

        self._xgb = XGBRegressor(**x_params)
        self._xgb.fit(xgb_features, prophet_errors)

        # Combined fitted values
        xgb_correction = self._xgb.predict(xgb_features)
        fitted_values = prophet_forecast + xgb_correction
        final_residuals = y - fitted_values

        self._fit_result = FitResult(
            fitted_values=fitted_values,
            residuals=final_residuals,
            model_object={"prophet": self._prophet, "xgb": self._xgb},
        )
        return self._fit_result

    def predict(self, target_df, target_col="y", date_col="ds", covariate_cols=None):
        covariate_cols = covariate_cols or self._covariate_cols

        # Prophet forecast on target dates
        prophet_df = target_df[[date_col]].copy()
        prophet_df.columns = ["ds"]
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        prophet_forecast = self._prophet.predict(prophet_df)["yhat"].values

        # XGB correction
        time_feats = _make_time_features(target_df, date_col)
        xgb_features = time_feats.copy()
        if covariate_cols:
            for col in covariate_cols:
                xgb_features[col] = target_df[col].values
        xgb_features["prophet_forecast"] = prophet_forecast

        xgb_correction = self._xgb.predict(xgb_features)
        combined = prophet_forecast + xgb_correction

        actual = target_df[target_col].values if target_col in target_df.columns else None

        return PredictionResult(
            dates=target_df[date_col].values,
            predicted=combined,
            actual=actual,
        )

    def clone_fresh(self):
        return ProphetThenXGBModel(params=self.params.copy())
