# Description: ARIMA model via pmdarima auto_arima.
# Usage: from its2s.models.arima import ARIMAModel
# Dependencies: pmdarima, numpy, pandas

import copy
import logging
import warnings

import numpy as np
import pandas as pd
import pmdarima as pm

from ..frequency import dominant_seasonal_period, resolve_frequency
from .base import BaseModel, FitResult, PredictionResult

logger = logging.getLogger(__name__)


def resolve_arima_m(m_cfg, n_train, series_freq=None):
    """Resolve the seasonal period m for the auto_arima search (GH #59).

    "auto" derives the dominant cycle from the resolved series frequency
    (daily 7, weekly 52, monthly 12) and falls back LOUDLY to m=1
    (non-seasonal) when the frequency is unmapped or the training window
    fails the length guard n_train >= 2m (one cycle is consumed by seasonal
    differencing; at least another is needed to estimate the seasonal terms).

    An explicit integer is always honored, never substituted: m < 1 raises,
    and an explicit value failing the length guard fits as asked with an
    advisory warning. This deliberately differs from
    resolve_metrics_seasonality, which raises on an explicit guard failure:
    there m defines the benchmark a reported metric is measured against;
    here it is a model specification whose quality is visible in the fit's
    own diagnostics.
    """
    if m_cfg == "auto":
        m = dominant_seasonal_period(series_freq)
        if m is None:
            alias = series_freq.alias if series_freq is not None else "unknown"
            warnings.warn(
                f"models.arima.m='auto': no dominant seasonal period is "
                f"mapped for series frequency '{alias}'. Falling back to "
                "m=1 (non-seasonal ARIMA). Set models.arima.m to an integer "
                "to name the period explicitly: "
                'config_overrides={"models": {"arima": {"m": <period>}}}.',
                UserWarning,
                stacklevel=2,
            )
            return 1
        if n_train < 2 * m:
            warnings.warn(
                f"models.arima.m='auto': a seasonal fit at m={m} needs "
                f"n_train >= 2*m ({2 * m}) but only {n_train} training "
                "observations are available. Falling back to m=1 "
                "(non-seasonal ARIMA). Set models.arima.m to an integer to "
                "override.",
                UserWarning,
                stacklevel=2,
            )
            return 1
        logger.info(
            "ARIMA seasonal period resolved from series frequency %s: m=%d",
            series_freq.alias, m,
        )
        return m

    m = int(m_cfg)
    if m < 1:
        raise ValueError(f"models.arima.m must be >= 1, got {m_cfg!r}.")
    if n_train < 2 * m:
        warnings.warn(
            f"models.arima.m={m}: a seasonal fit at m={m} needs n_train >= "
            f"2*m ({2 * m}) but only {n_train} training observations are "
            "available. The explicit value is honored; expect an unstable "
            "or degenerate seasonal fit. Set a smaller m, or "
            "models.arima.seasonal=false, to silence this warning.",
            UserWarning,
            stacklevel=2,
        )
    return m


class ARIMAModel(BaseModel):
    """ARIMA model using pmdarima's auto_arima for automatic order selection.

    On first fit, auto_arima selects the best (p,d,q) and seasonal order.
    clone_fresh() preserves the discovered order so MBB refits use the same
    model structure without repeating the expensive stepwise search.

    The seasonal period m defaults to "auto": resolved from the training
    dates via the series-frequency dominant-cycle mapping (daily 7, weekly
    52, monthly 12), with a loud fallback to m=1 when the frequency is
    unmapped, unresolvable, or the training window is too short. An explicit
    integer m is always honored (see resolve_arima_m). m is only consulted
    on the initial auto_arima search: refit clones reuse the discovered
    seasonal_order, which already carries it.
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
                m=self._resolve_m(train_df, date_col),
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

    def _resolve_m(self, train_df, date_col):
        """Resolve the seasonal period for the initial auto_arima search."""
        p = self.params
        if not p.get("seasonal", True):
            # m cannot matter when the seasonal search is off; skip
            # resolution so no frequency warning fires for an inert knob.
            return 1
        m_cfg = p.get("m", "auto")
        if m_cfg != "auto":
            return resolve_arima_m(m_cfg, n_train=len(train_df))
        try:
            series_freq = resolve_frequency(
                pd.to_datetime(train_df[date_col]).sort_values()
            )
        except ValueError as exc:
            # Standalone use on an irregular grid must stay usable: warn and
            # fall back rather than turning a previously working fit into a
            # hard error. Pipeline callers never reach this (the pipeline
            # resolves and raises at ingest).
            warnings.warn(
                f"models.arima.m='auto': could not resolve the series "
                f"frequency from the training dates "
                f"({str(exc).splitlines()[0]}) Falling back to m=1 "
                "(non-seasonal ARIMA). Set models.arima.m explicitly if the "
                "series is seasonal.",
                UserWarning,
                stacklevel=3,
            )
            return 1
        return resolve_arima_m(
            "auto", n_train=len(train_df), series_freq=series_freq
        )

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
