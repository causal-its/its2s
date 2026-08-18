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

| Column | Description |
|--------|-------------|
| `window` | "train" or "test" |
| `rmse` | Root mean square error |
| `mae` | Mean absolute error |
| `mape` | Mean absolute percentage error |
| `smape` | Symmetric MAPE |
| `mase` | Mean absolute scaled error |
| `r2` | R² (coefficient of determination) |

**How to use**: assess these before interpreting excess estimates. The test window
performance is the most important signal — it measures how well the model would have
tracked the outcome in the pre-event period it was not trained on. Poor test R² or
large test RMSE relative to the outcome's scale indicates that the counterfactual is
unreliable.

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
