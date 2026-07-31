# Description: Hybrid Prophet+XGBoost model -- Prophet decomposes, XGB fits residuals.
# Usage: from its2s.models.prophet_xgb import ProphetXGBHybridModel
# Dependencies: prophet, xgboost, numpy, pandas

import numpy as np
import pandas as pd
from prophet import Prophet
from xgboost import XGBRegressor

from .base import BaseModel, FitResult, PredictionResult
from .utils import make_time_features as _make_time_features


class ProphetXGBHybridModel(BaseModel):
    """Hybrid: Prophet captures trend+seasonality, XGBoost models the residuals.

    Simultaneous approach -- Prophet is fit first, then XGB is trained on
    (target - prophet_components) using covariates and time features.
    Final prediction = prophet_components + xgb_residual_prediction.
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

        # Fit Prophet
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

        # Get Prophet in-sample components (trend + seasonality)
        prophet_pred = self._prophet.predict(prophet_df)
        prophet_components = prophet_pred["yhat"].values

        # Residuals for XGBoost
        y = train_df[target_col].values.astype(float)
        residuals_for_xgb = y - prophet_components

        # Build XGB features
        time_feats = _make_time_features(train_df, date_col)
        xgb_features = time_feats.copy()
        if covariate_cols:
            for col in covariate_cols:
                xgb_features[col] = train_df[col].values

        self._xgb = XGBRegressor(**x_params)
        self._xgb.fit(xgb_features, residuals_for_xgb)

        # Combined fitted values
        xgb_fitted = self._xgb.predict(xgb_features)
        fitted_values = prophet_components + xgb_fitted
        final_residuals = y - fitted_values

        self._fit_result = FitResult(
            fitted_values=fitted_values,
            residuals=final_residuals,
            model_object={"prophet": self._prophet, "xgb": self._xgb},
        )
        return self._fit_result

    def predict(self, target_df, target_col="y", date_col="ds", covariate_cols=None):
        covariate_cols = covariate_cols or self._covariate_cols

        # Prophet forecast
        prophet_df = target_df[[date_col]].copy()
        prophet_df.columns = ["ds"]
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        prophet_pred = self._prophet.predict(prophet_df)
        prophet_components = prophet_pred["yhat"].values

        # XGB residual prediction
        time_feats = _make_time_features(target_df, date_col)
        xgb_features = time_feats.copy()
        if covariate_cols:
            for col in covariate_cols:
                xgb_features[col] = target_df[col].values

        xgb_preds = self._xgb.predict(xgb_features)
        combined = prophet_components + xgb_preds

        actual = target_df[target_col].values if target_col in target_df.columns else None

        return PredictionResult(
            dates=target_df[date_col].values,
            predicted=combined,
            actual=actual,
        )

    def clone_fresh(self):
        return ProphetXGBHybridModel(params=self.params.copy())
