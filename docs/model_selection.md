# Model Selection

## Selection criterion

Choose the model with the lowest RMSE on the **pre-event test window** — not on the
training data. A model that fits training data well but generalizes poorly to a held-out
pre-event window will produce an unreliable counterfactual. Training fit and test fit
can diverge substantially, particularly for flexible nonlinear models.

Use `compare_models()` to evaluate all three models on the same series before committing:

```python
from its2s import compare_models

comparison_df, results = compare_models(
    df,
    intervention_date="2022-06-01",
    model_names=["prophet_xgb", "neuralprophet", "arima"],
)
print(comparison_df)
```

The returned `comparison_df` shows train and test metrics side by side. Select the model
with the lowest test RMSE, then pass it to `run_single_its` for the final counterfactual.
If you tune hyperparameters first, `tune_model` keeps its CV folds out of this test
window by default (GH #40), so the test metrics you select on stay clean out-of-sample.

---

## Prophet+XGBoost (recommended default)

A hybrid model that combines Prophet and XGBoost in a single optimization. Prophet
handles the structured temporal components — trend, seasonality, and holiday effects —
via additive decomposition. XGBoost then models the nonlinear residuals from Prophet as
an ensemble of decision trees.

This combination handles:

- Complex seasonal patterns (multiple periodicities, non-smooth cycles)
- Saturating or nonlinear trends
- Nonlinear covariate relationships
- Irregular holiday or event effects

**When to use**: daily health or count outcome series with rich seasonal structure and
moderate to complex covariate relationships, and at least two to three years of training
data to calibrate Prophet's changepoint detection reliably.

**Computational cost**: moderate. Grows with the number of tuning trials (`n_trials`).
Fitting a single model is fast; 100-trial tuning adds several minutes on a daily series
of several years.

---

## NeuralProphet

A neural network autoregressive model. Captures complex nonlinear temporal patterns
that purely additive decomposition models may miss, including high-order lag dependencies
and nonlinear interaction effects across covariates and time.

**When to use**: series where the temporal structure is highly nonlinear and standard
decomposition approaches underfit. Requires substantially more training data than ARIMA
or Prophet+XGB to generalize reliably.

**Caution**: NeuralProphet is more susceptible to overfitting than the other models.
Strong training performance does not imply strong test generalization. Always check test
metrics explicitly before selecting this model.

**Optional dependency**: NeuralProphet requires PyTorch (~1 GB). Install via:

```bash
pip install its2s[neural]
```

---

## ARIMA

A traditional autoregressive integrated moving average model, fitted via
`pmdarima.auto_arima`. The model assumes that the temporal structure of the outcome
is a linear combination of past values, past forecast errors, and seasonal components.

**When to use**:

- Short series where more flexible models overfit
- Series with a simple, stable linear trend and standard autocorrelation structure
- Situations where computational resources are limited (ARIMA is by far the fastest)
- As a baseline to assess how much nonlinear models improve over a linear fit

**Limitations**:

- The seasonal period `m` defaults to `"auto"` (resolved from the series frequency:
  daily 7, weekly 52, monthly 12, with a loud non-seasonal fallback when the frequency
  is unmapped or the training window is under `2m`). On weekly data the resolved
  `m=52` can make the stepwise search substantially slower; set an explicit `m` or
  `seasonal: false` to trade seasonality for speed.
- ARIMA cannot capture nonlinear trend or covariate effects. If the pre-event series
  contains structural nonlinearities, ARIMA's counterfactual will be miscalibrated.

---

## Recommendation

1. Run `compare_models()` first and examine test RMSE across all models.
2. Select the model with the lowest test RMSE.
3. If test metrics are similar across models, prefer Prophet+XGB for its balance of
   nonlinearity and structural interpretability.
4. Use ARIMA for short series or when compute is limited.
5. Use NeuralProphet only when other models show clear underfitting on the test window,
   and verify test performance carefully before using it for the counterfactual.
