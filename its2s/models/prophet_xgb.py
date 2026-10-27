# Description: Hybrid Prophet+XGBoost model -- Prophet decomposes, XGB fits residuals.
# Usage: from its2s.models.prophet_xgb import ProphetXGBHybridModel
# Dependencies: prophet, xgboost, numpy, pandas

import copy

import pandas as pd
from prophet import Prophet
from prophet.serialize import model_from_json, model_to_json
from xgboost import XGBRegressor

from .base import BaseModel, FitResult, PredictionResult
from .families import get_family
from .utils import make_time_features as _make_time_features


class ProphetXGBHybridModel(BaseModel):
    """Hybrid: Prophet captures trend+seasonality, XGBoost models the residuals.

    Simultaneous approach -- Prophet is fit first, then XGB is trained on
    (target - prophet_components) using covariates and time features.
    Final prediction = prophet_components + xgb_residual_prediction.

    Prophet carries exactly the declared cycles (yearly iff "annual", weekly
    iff "weekly", never daily); nothing is left on "auto".
    """

    spec = get_family("prophet_xgb")
    _libraries = ("prophet", "xgboost")

    def __init__(self, params, seasonality):
        super().__init__(params, seasonality)
        self._prophet = None
        self._xgb = None

    def _build_prophet(self):
        p_params = self.params["prophet"]
        cycles = self.seasonality.cycles
        return Prophet(
            yearly_seasonality="annual" in cycles,
            weekly_seasonality="weekly" in cycles,
            daily_seasonality=False,
            changepoint_prior_scale=p_params["changepoint_prior_scale"],
        )

    def _xgb_features(self, df, date_col, covariate_cols):
        features = _make_time_features(df, date_col)
        if covariate_cols:
            for col in covariate_cols:
                features[col] = df[col].values
        return features

    def fit(self, train_df, target_col="y", date_col="ds", covariate_cols=None):
        self._covariate_cols = covariate_cols

        # Fit Prophet
        prophet_df = train_df[[date_col, target_col]].copy()
        prophet_df.columns = ["ds", "y"]
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        self._prophet = self._build_prophet()
        self._prophet.fit(prophet_df)

        # Get Prophet in-sample components (trend + seasonality)
        prophet_components = self._prophet.predict(prophet_df)["yhat"].values

        # Residuals for XGBoost
        y = train_df[target_col].values.astype(float)
        residuals_for_xgb = y - prophet_components

        xgb_features = self._xgb_features(train_df, date_col, covariate_cols)
        self._xgb = XGBRegressor(**self.params["xgb"])
        self._xgb.fit(xgb_features, residuals_for_xgb)

        # Combined fitted values
        fitted_values = prophet_components + self._xgb.predict(xgb_features)

        self._fit_result = FitResult(
            fitted_values=fitted_values,
            residuals=y - fitted_values,
            model_object={"prophet": self._prophet, "xgb": self._xgb},
        )
        return self._fit_result

    def predict(self, target_df, target_col="y", date_col="ds", covariate_cols=None):
        covariate_cols = covariate_cols or self._covariate_cols

        # Prophet forecast
        prophet_df = target_df[[date_col]].copy()
        prophet_df.columns = ["ds"]
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        prophet_components = self._prophet.predict(prophet_df)["yhat"].values

        # XGB residual prediction
        xgb_features = self._xgb_features(target_df, date_col, covariate_cols)
        combined = prophet_components + self._xgb.predict(xgb_features)

        actual = target_df[target_col].values if target_col in target_df.columns else None

        return PredictionResult(
            dates=target_df[date_col].values,
            predicted=combined,
            actual=actual,
        )

    def clone_fresh(self):
        return ProphetXGBHybridModel(params=copy.deepcopy(self.params),
                                     seasonality=self.seasonality)

    def seasonal_components(self):
        """Fitted Prophet seasonalities and the XGB time-feature columns."""
        return {"prophet": sorted(self._prophet.seasonalities),
                "xgb_time_features": list(_make_time_features(
                    pd.DataFrame({"ds": pd.to_datetime([])}), "ds").columns)}

    def _save_fitted(self, directory):
        (directory / "prophet.json").write_text(model_to_json(self._prophet))
        self._xgb.save_model(directory / "xgb.json")

    def _load_fitted(self, directory):
        self._prophet = model_from_json((directory / "prophet.json").read_text())
        self._xgb = XGBRegressor()
        self._xgb.load_model(directory / "xgb.json")
        # A loaded model can predict but carries no in-sample fit.
        self._fit_result = None
