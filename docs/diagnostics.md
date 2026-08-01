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
| `acf` | `dict[int, float]` | Full residual ACF vector at lags 1..`max_lag` (the persisted descriptive record) |
| `key_lags` | `list[int]` | The pre-specified inferential lags `{1, m}`; values are looked up in `acf` |
| `params` | `dict` | `n`, `max_lag`, `m`, `freq_alias`, and any fallback notes |
| `ljung_box_stat` | `float` | Ljung-Box test statistic |
| `ljung_box_pvalue` | `float` | Ljung-Box p-value |
| `ljung_box_lags` | `int \| None` | Pooled depth actually used, `min(2m, n // 5)` |
| `shapiro_stat` | `float \| None` | Shapiro-Wilk test statistic (`None` if n > 5000) |
| `shapiro_pvalue` | `float \| None` | Shapiro-Wilk p-value (`None` if n > 5000) |
| `model_metadata` | `dict` | Model-specific metadata (varies by model) |

---

### Frequency-conditional lag semantics

Lags are row counts, so their meaning changes with the series frequency: lag 7 is
weekly structure on daily data and nothing in particular on weekly data. The
diagnostics therefore derive their lag choices from the resolved series frequency
rather than assuming daily data. The seasonal period `m` is the dominant cycle of
the frequency — daily 7, weekly 52, monthly 12 — the same mapping the metrics use
for the MASE benchmark. Frequencies outside that mapping fall back loudly (a
warning is emitted, `params["notes"]` records the substitution).

The diagnostic splits into three layers computed once:

- **The persisted record** — `acf`, the full ACF vector at lags
  1..`max_lag = min(n // 2, n - 30)` (beyond `n // 2` an estimate uses fewer than
  half the data; 30 pairs is the floor below which an ACF estimate stops being an
  estimate). Nothing is invisible: on a weekly series the annual lag 52 is in the
  vector whenever the series can support it.
- **The inferential claim** — `key_lags = {1, m}`, deliberately minimal: lag 1 is
  frequency-invariant short-range dependence; `m` is the dominant seasonal cycle.
  Every key lag is a pre-specified check. A key lag the series is too short to
  estimate is reported as `NaN` with a warning naming the reason — "cannot
  estimate the annual lag" is itself diagnostic information.
- **The picture** — the correlogram plot (`{model}_residual_acf.png`, written to
  the run output directory) renders the persisted vector with the key lags
  marked; it never recomputes.

A value above ~0.2 in absolute terms at a key lag warrants investigation. The
rest of the vector is descriptive context, not a menu of hypothesis tests: with
~70 lags, some will exceed the naive threshold by chance.

---

### Residual plots

When an `output_dir` is supplied, four residual diagnostic plots are written
alongside the counterfactual figure. All four describe the train-only fit — the
model fit on the training window — until the final refit lands (GH #63), after
which they follow the final fit automatically. The same values are persisted in
`{model}_diagnostics.csv` (see the [output guide](output_guide.md)).

| File | Shows |
|------|-------|
| `{model}_residual_acf.png` | The persisted ACF vector (never recomputed) over lags 1..`max_lag`, with the key lags `{1, m}` marked and labeled. The bands are 95% white-noise nulls (`+/-1.96/sqrt(n)`), the reference for "is this lag distinguishable from noise" |
| `{model}_residual_pacf.png` | Partial ACF over the same lag range (capped at `n // 2 - 1`), computed at plot time via statsmodels (`method="ywm"`). Same bands, so the two correlograms read in parallel |
| `{model}_residuals_over_time.png` | Raw training residuals against the training dates; NaN residuals (e.g. NeuralProphet AR warmup) appear as gaps. Look for drift, variance changes, or clusters |
| `{model}_residual_qq.png` | Normal QQ plot of the residuals; the visual companion of the Shapiro-Wilk test |

When the series is too short to reach the dominant seasonal lag
(`max_lag < m`), the correlograms say so on the figure instead of silently
truncating; when it is too short for any lag, an annotated placeholder is
written (with a warning) so the run's file inventory stays stable.

The correlogram is what makes structure at unexpected lags visible: on a weekly
series, unmodeled annual seasonality shows up at lag 52 — a lag no fixed
daily-shaped report would surface.

---

### Ljung-Box test

Tests whether the model residuals are serially independent (white noise), pooled
over `min(2m, n // 5)` lags — the seasonal prescription, power-capped. Computed
via `statsmodels.stats.diagnostic.acorr_ljungbox`; the depth actually used is
always reported in `ljung_box_lags`.

- **Null hypothesis**: residuals are independent (no remaining autocorrelation).
- **Significant p-value** (< 0.05): the model has not captured all temporal dependence
  in the pre-event series. Remaining autocorrelation invalidates the standard MBB
  assumption that blocks of residuals are approximately exchangeable at the chosen
  block length.
- **When the cap binds** (`n // 5 < m`, common for weekly series): the pooled
  window cannot reach the seasonal lag, and `params["notes"]` says so. In that
  regime the key-lag ACF at `m` carries the seasonal check directly; pooling to
  `2m` anyway would destroy the test's power.

**Action**: try a different model (`compare_models()`), add covariates that explain
the remaining structure, or verify that the seasonal period (`m` for ARIMA,
`"auto"` by default: resolved from the series frequency with a loud non-seasonal
fallback) is correctly specified.

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
| High ACF at lag 1 | Short-range dependence not captured | Try a different model; add covariates |
| High ACF at the seasonal key lag `m` | Unmodeled seasonality at the dominant cycle | Verify the ARIMA `m`; check Prophet seasonality config |
| Key lag reported as `NaN` | Series too short to estimate that lag | Treat the seasonal check as unavailable; extend the training window if possible |
| Non-normal residuals | Mis-specified distribution or outliers | Inspect for outliers; consider log-transforming outcome |
| Large `residual_mean` | Systematic bias | Add missing predictors; check for trend mis-specification |
| Large test RMSE vs the outcome scale | Model does not generalize | Run `compare_models()`; add covariates; extend training window |
