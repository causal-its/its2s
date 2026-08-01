# Interpreting Outputs

## How excess is computed

For each time point in the post-event period:

- **Expected** (counterfactual): what the model estimates the outcome would have been
  absent the event, based on the pre-event trend.
- **Excess** = observed − expected.
- **95% empirical CI**: derived from the 2.5th and 97.5th percentiles of 1000 Moving
  Block Bootstrap (MBB) iterations. The CI is empirical and asymmetric — the point
  estimate does not necessarily fall at the midpoint of the interval.
- **`excess_pct`** = (excess / expected) × 100. This is the attributable fraction
  expressed as a percentage.

A negative excess indicates fewer observed events than expected — the event was
associated with a reduction in the outcome.

---

## Output files

All output files are written to the `output_dir` you specify. File names include the
`model_name` (e.g., `prophet_xgb`).

### `{model}_excess.csv` — per-observation excess estimates

One row per post-event time point.

| Column | Description |
|--------|-------------|
| `date` | Date of the observation |
| `observed` | Observed outcome value |
| `expected` | Counterfactual prediction (what the model predicts absent the event) |
| `expected_ci_lo` | Lower bound of the 95% eCI for the expected value |
| `expected_ci_hi` | Upper bound of the 95% eCI for the expected value |
| `excess` | `observed` − `expected` |
| `excess_ci_lo` | Lower bound of the 95% eCI for excess |
| `excess_ci_hi` | Upper bound of the 95% eCI for excess |
| `excess_pct` | `excess / expected × 100` |
| `excess_pct_ci_lo` | Lower bound of the 95% eCI for `excess_pct` |
| `excess_pct_ci_hi` | Upper bound of the 95% eCI for `excess_pct` |

---

### `{model}_excess_period.csv` — period-level aggregation

One row per defined period. By default, one row for "Full holdout" covering the entire
post-event window. Custom sub-periods are defined via `excess_periods` in the config;
each period is delimited in exactly one explicit unit family: `start_offset_days` /
`end_offset_days` (calendar days from the holdout start, inclusive end) or
`start_offset_obs` / `end_offset_obs` (observation counts into the holdout, inclusive
end). See the `excess_periods` example in `params.yaml`.

| Column | Description |
|--------|-------------|
| `period` | Period label (e.g., "Full holdout") |
| `start_date` | First date of the period |
| `end_date` | Last date of the period |
| `n_obs` | Number of observations (rows) in the period |
| `total_observed` | Sum of observed values over the period |
| `total_expected` | Sum of expected (counterfactual) values over the period |
| `total_excess` | `total_observed` − `total_expected` |
| `excess_ci_lo` | Lower bound of the 95% eCI for total excess |
| `excess_ci_hi` | Upper bound of the 95% eCI for total excess |
| `excess_pct` | `total_excess / total_expected × 100` |

---

### `{model}_ate_summary.csv` — average treatment effect summary

Two rows summarizing the overall effect.

| Column | Description |
|--------|-------------|
| `metric` | "Total ATE" or "Mean ATE per obs" |
| `estimate` | Point estimate |
| `ci_lo` | Lower bound of the 95% eCI |
| `ci_hi` | Upper bound of the 95% eCI |
| `n_obs` | Number of post-event observations in the summary |

Access programmatically:

```python
from its2s import calc_ate_summary

ate = calc_ate_summary(result.excess_table)
print(ate)
```

---

### `{model}_metrics.csv` — model performance on train and test windows

Each reported metric has one job:

| Column | Description |
|--------|-------------|
| `window` | "train" or "test" |
| `rmse` | Root mean square error: accuracy on the mean, the model-selection metric |
| `mae` | Mean absolute error: accuracy in native outcome units, robust |
| `mape` | Mean absolute percentage error: percentage communication. `NaN` (with a warning) when the window contains zero actuals — skipping zeros would silently drop the hardest observations |
| `mase` | Seasonal-naive benchmark ratio: model MAE over the in-sample MAE of the `m`-period seasonal naive. Below 1 = beats the benchmark. This is a benchmark comparison, not an accuracy metric |
| `mase_m` | The benchmark period `m`. Derived from the series frequency when `metrics.seasonality` is `"auto"` (daily 7, weekly 52, monthly 12) |
| `mase_denominator` | The benchmark's in-sample seasonal-naive MAE, in native units. The ratio is meaningless without it |

**How to use**: assess these before interpreting excess estimates. The test window
performance is the most important signal — it measures how well the model would have
tracked the outcome in the pre-event period it was not trained on. Large test RMSE
relative to the outcome's scale, or a test MASE near or above 1 (no better than
carrying forward the last seasonal cycle), indicates that the counterfactual is
unreliable.

---

### `{model}_diagnostics.csv` — residual diagnostics (tidy long format)

One row per diagnostic statistic; the full persisted ACF vector gets one row per
lag. The resolved series frequency, the seasonal period `m`, and the residual
count `n` repeat on every row, so each value is self-describing: a lag is always
in observations of the resolved frequency (on weekly data, lag 52 is 52 weeks),
never calendar days.

These diagnostics describe the train-only fit: the model fit on the training
window whose residuals are available before the event (GH #63 tracks the final
refit; once it lands, this file follows the final fit automatically).

| Column | Description |
|--------|-------------|
| `model_name` | Model the residuals come from |
| `section` | Row group: `summary`, `acf`, `ljung_box`, `shapiro`, or `params` |
| `statistic` | Statistic name (e.g., `residual_mean`, `acf`, `ljung_box_pvalue`, `shapiro_stat`, `max_lag`) |
| `lag` | Lag for `acf` rows; empty otherwise |
| `lag_units` | `observations` on lag-bearing rows (`acf`, `ljung_box_lags`, `max_lag`); empty otherwise |
| `value` | The statistic's value; empty when `status` is not `ok` |
| `status` | `ok` (computed, finite), `nan` (computed, result NaN), or `not_computed` (a precondition failed, e.g. the series is too short) |
| `note` | The reason on non-`ok` rows (e.g. `n <= 15: Ljung-Box skipped`, `key lag 52 exceeds max_lag=30`) |
| `freq_alias` | Resolved pandas frequency alias of the series (e.g. `D`, `W-SUN`) |
| `m` | Dominant seasonal period in observations (daily 7, weekly 52, monthly 12); empty when the frequency has no mapped cycle |
| `n` | Number of residuals after dropping NaNs |

`status` is the column to check before reading `value`: an empty `value` cell
alone does not distinguish a statistic that failed its precondition from one
that was computed and returned NaN.

**How to use**: the key lags `{1, m}` carry the pre-specified checks (see
[Diagnostics](diagnostics.md)); the rest of the ACF rows are the descriptive
record that makes structure at unexpected lags visible in the file itself.

---

### `{model}_counterfactual.png` — visual assessment

Displays the full observed outcome series alongside the counterfactual prediction with
a 95% eCI ribbon. The event period is shaded.

Visual inspection of the **pre-event portion** of this plot is the primary diagnostic
check: the observed and predicted lines should track closely before the event. If they
diverge substantially in the pre-event window, the counterfactual is not credible
regardless of what the numeric metrics show. See [Diagnostics](diagnostics.md) for
guidance on what to do when the fit is poor.

---

## Accessing outputs in Python

All outputs are also available on the returned `PipelineResult` object without writing
to disk:

```python
result = run_single_its(df, intervention_date="2022-06-01")

# Per-observation excess table (ExcessResult)
print(result.excess_table.obs_excess)

# Period-level aggregation
print(result.excess_table.period_excess)

# ATE summary
from its2s import calc_ate_summary
print(calc_ate_summary(result.excess_table))

# Train/test metrics
print(result.metrics_train)  # MetricsResult
print(result.metrics_test)

# Residual diagnostics
print(result.diagnostics)
```
