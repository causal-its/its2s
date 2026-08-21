# Description: Model registry for ITS forecasting models.
# Lazy imports to avoid pulling heavy dependencies at package load time.


def __getattr__(name):
    if name == "ARIMAModel":
        from .arima import ARIMAModel
        return ARIMAModel
    if name == "NeuralProphetModel":
        from .neuralprophet import NeuralProphetModel
        return NeuralProphetModel
    if name == "ProphetXGBHybridModel":
        from .prophet_xgb import ProphetXGBHybridModel
        return ProphetXGBHybridModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ProphetXGBHybridModel",
    "NeuralProphetModel",
    "ARIMAModel",
]
