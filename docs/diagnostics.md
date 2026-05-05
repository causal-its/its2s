# Diagnostics

## Pre-event fit is the primary check

Before interpreting any excess estimates, inspect `{model}_counterfactual.png`. The
observed outcome and the model's expected (counterfactual) line should track closely
throughout the pre-event period. If they diverge substantially before the event,
the model has not captured the baseline trend, and the counterfactual projection into
the post-event period is unreliable — regardless of what formal residual tests show.

Visual inspection of the pre-event fit takes priority over all other diagnostics.

---

## Residual diagnostics

Residual diagnostics are available on the `PipelineResult` object after calling
`run_single_its`:

```python
result = run_single_its(df, intervention_date="2022-06-01")
diag = result.diagnostics  # DiagnosticsResult | None
```

`diagnostics` is `None` if the model did not produce residuals (e.g., fit failed).

### Available fields

| Field | Type | Description |
|-------|------|-------------|
| `residual_mean` | `float` | Mean of model residuals on the training data |
| `residual_std` | `float` | Standard deviation of residuals |
| `acf_lag1` | `float` | Autocorrelation of residuals at lag 1 |
| `acf_lag7` | `float` | Autocorrelation of residuals at lag 7 (weekly) |
| `acf_lag14` | `float` | Autocorrelation of residuals at lag 14 (biweekly) |
| `ljung_box_stat` | `float` | Ljung-Box test statistic (at 10 lags) |
| `ljung_box_pvalue` | `float` | Ljung-Box p-value |
| `shapiro_stat` | `float \| None` | Shapiro-Wilk test statistic (`None` if n > 5000) |
| `shapiro_pvalue` | `float \| None` | Shapiro-Wilk p-value (`None` if n > 5000) |
| `model_metadata` | `dict` | Model-specific metadata (varies by model) |

---

### Ljung-Box test

Tests whether the model residuals are serially independent (white noise) at up to
10 lags. Computed via `statsmodels.stats.diagnostic.acorr_ljungbox`.

- **Null hypothesis**: residuals are independent (no remaining autocorrelation).
- **Significant p-value** (< 0.05): the model has not captured all temporal dependence
  in the pre-event series. Remaining autocorrelation invalidates the standard MBB
  assumption that blocks of residuals are approximately exchangeable at the chosen
  block length.

**Action**: try a different model (`compare_models()`), add covariates that explain
the remaining structure, or verify that the seasonal period (`m` for ARIMA) is
correctly specified.

---

### ACF at lags 1, 7, 14

Three scalar autocorrelation values for the model residuals. The package does not
produce a full ACF plot; these three lags are the ones exposed in `DiagnosticsResult`.

| Lag | What it signals |
|-----|-----------------|
| 1 | Day-to-day carryover; large value = short-range dependence not captured |
| 7 | Weekly cycle; large value = unmodeled day-of-week effect |
| 14 | Biweekly cycle; large value = unmodeled fortnightly pattern |

A value above ~0.2 in absolute terms at any of these lags warrants investigation.

---

### Shapiro-Wilk test

Tests whether the model residuals are normally distributed.

- Skipped automatically when the training series length exceeds 5000 observations
  (both fields are `None` in that case).
- **Non-normality does not invalidate the MBB confidence interval**, which is
  nonparametric. However, strongly skewed or heavy-tailed residuals may indicate a
  mis-specified outcome distribution (for example, sparse count outcomes that produce
  many zero residuals and occasional large outliers).

**Action**: inspect residual distribution visually; consider log-transforming the
outcome or switching to a model better suited to count data.

---

### Residual mean

A `residual_mean` far from zero indicates systematic bias in the model's training
predictions. Mild bias is common and does not disqualify the counterfactual, but large
systematic over- or under-prediction suggests the model is missing a structural feature
of the series.

---

## Interpreting confidence interval width

The width of the 95% empirical CI reflects prediction uncertainty from both sources
captured by the MBB:

- **Series variability**: noisier outcomes naturally produce wider CIs.
- **Model uncertainty**: poor pre-event fit amplifies the CI width.

A total excess CI that spans zero means the data are statistically consistent with
no effect at the 95% level. This is distinct from the effect being absent — it means
the uncertainty is too large to distinguish an effect from noise at this sample size.

---

## What to do when diagnostics flag a problem

| Problem | Likely cause | Action |
|---------|-------------|--------|
| Ljung-Box p < 0.05 | Remaining autocorrelation | Try a different model; check `m`; add covariates |
| High ACF at lag 7 | Unmodeled weekly seasonality | Verify `m=7` for ARIMA; check Prophet seasonality config |
| Non-normal residuals | Mis-specified distribution or outliers | Inspect for outliers; consider log-transforming outcome |
| Large `residual_mean` | Systematic bias | Add missing predictors; check for trend mis-specification |
| Poor test R² | Model does not generalize | Run `compare_models()`; add covariates; extend training window |
