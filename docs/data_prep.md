# Data Preparation

## DataFrame format

The pipeline expects a pandas DataFrame with at minimum:

- A **date column** (default name: `ds`) containing dates parseable as `datetime`.
- A **numeric outcome column** (default name: `y`) containing the time series values.

To use different column names, pass `date_col` and `target_col` to `run_single_its`:

```python
run_single_its(df, intervention_date="2022-06-01", date_col="date", target_col="count")
```

The series is sorted by `ds` inside the pipeline. If the input is already sorted,
no reordering occurs; if it is not, a `UserWarning` is emitted and the DataFrame is
sorted automatically.

---

## The three-period data structure

The two-stage framework partitions the time series into three non-overlapping periods,
all anchored on `intervention_date`:

| Period | Role |
|--------|------|
| **Training** | Historical baseline used to fit each candidate model. Ends where the test window begins. |
| **Testing** (pre-event) | Held-out pre-event window used for model selection only. Sits immediately before `intervention_date`. Must not include the event. |
| **Post-event** | During and after the intervention. The selected model predicts the counterfactual here. |

Window units are explicit per split method. `"percent"` and `"observations"` size
windows in observations (rows of the regular series); `"days"` sizes them in calendar
days, so on a weekly series `test_days=365` spans about 52 observations. Passing an
argument that belongs to a different `split_method` raises an error rather than being
silently ignored.

Key configuration parameters (set in `params.yaml` or via `config_overrides`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `periods.split_method` | `"percent"` | `"percent"` (default) sizes windows as fractions of the available observations; `"days"` uses calendar-day counts; `"observations"` uses explicit observation counts |
| `periods.test_pct` | 0.20 | Fraction of pre-intervention observations used as the test window (`split_method="percent"`) |
| `periods.holdout_pct` | 1.0 | Fraction of post-intervention observations used as the holdout window (`split_method="percent"`) |
| `periods.test_days` | 365 | Pre-event test window in calendar days (`split_method="days"`) |
| `periods.holdout_days` | 365 | Post-event projection window in calendar days (`split_method="days"`) |
| `periods.test_obs` | none (required) | Pre-event test window in observations (`split_method="observations"`) |
| `periods.holdout_obs` | none (required) | Post-event projection window in observations (`split_method="observations"`) |
| `periods.min_test_obs` | 30 | Warn when the realized test window has fewer observations, whatever the split method; 0 disables |

The percent-based default sizes the test window proportionally to the available
pre-intervention data, so the pipeline runs without manual tuning on short series. Use
`split_method="days"` when the test/holdout window length must match a fixed calendar
duration (e.g., a pre-registered analysis specifying a 365-day post-event window), and
`split_method="observations"` when it must contain an exact number of observations
(e.g., a 78-week test window on weekly data). The resulting train/test/holdout sizes
are logged at run time as observation counts and percentages, so a mis-sized window is
visible immediately. Independently of the split method, a warning fires when the
realized test window ends up smaller than `periods.min_test_obs` (default 30): test
metrics computed on very few points are unstable and can mislead model selection,
regardless of how the small window arose.

The test period is used only for model selection — it is never used to fit the final
model. The final model is calibrated on the full pre-event dataset (training + test)
before generating the counterfactual.

---

## Minimum data requirements

With percent-based defaults, there is no fixed day-count minimum. The training window
must still be long enough to:

1. Capture multiple full seasonal cycles (so the model can learn seasonal patterns).
2. Support a stable cross-validation budget (the default `tuning.min_train_pct=0.50`
   reserves 50% of the pre-intervention slice for the first fold's training window;
   `tuning.test_pct=0.10` per fold leaves room for five non-overlapping folds).

As a practical guide, the pre-event window should span at least two seasonal cycles of
the dominant seasonality (e.g., two years for a weekly or annual cycle on daily data).
Shorter series increase overfitting risk and reduce the reliability of bootstrap
confidence intervals.

---

## Temporal frequency

The package is tested and validated on daily data. It can be applied to other
frequencies, but requires configuration adjustments:

- **ARIMA seasonal period**: the default `m="auto"` derives the seasonal period from
  the resolved series frequency (daily 7, weekly 52, monthly 12). When the frequency
  is outside that mapping (e.g. quarterly) or the training window is shorter than
  `2m`, it falls back to `m=1` (non-seasonal) with a visible warning. An explicit
  integer (`config_overrides={"models": {"arima": {"m": 52}}}`) is always honored;
  it warns, but still fits as asked, if the training window is shorter than `2m`.
  Note the cost: on weekly data the resolved `m=52` can make the seasonal stepwise
  search substantially slower than the non-seasonal fit.

- **Series frequency**: resolved automatically from the date column and passed to any
  model that needs it (NeuralProphet's `freq`, ARIMA's `m`). There is no `freq` setting to
  declare. The resolver requires the series to be a complete, regularly spaced grid:
  gaps, duplicate dates, or irregular spacing raise an error naming the first
  offending timestamp. Note this also applies after `missing_data="drop"` removes
  rows -- a mid-series drop creates a gap; fill or aggregate to a regular grid
  instead.

- **Block length**: `bootstrap.block_length` is measured in observations (rows),
  never calendar days: the default 14 -- derived for daily data -- spans two weeks
  on a daily series but 14 weeks on a weekly one. For non-daily frequencies, this
  value may need to be adjusted manually.

- **NeuralProphet AR window**: `n_lags` likewise counts observations, not days:
  the default 14 is a two-week autoregressive window on daily data but a 14-week
  window on weekly data. Consider whether that window is what you mean.

---

## Covariates

Covariates are time-varying numeric columns in the same DataFrame as `ds` and `y`.
They are passed to model fitting and to the counterfactual prediction, so they
participate in both the pre-event model and the post-event projection.

Two requirements are critical:

1. **Full temporal coverage**: covariates must be present for every row — pre-event and
   post-event. A covariate that is only available for the pre-event period cannot be
   used, because the counterfactual prediction requires it in the post-event window.

2. **Not a mediator**: covariates should represent "business as usual" conditions
   (e.g., seasonal weather patterns, calendar effects) that would have continued
   regardless of the event. A variable that is itself affected by the event being
   studied would introduce bias into the counterfactual.

The pipeline raises a `ValueError` if any covariate column contains `NaN` values.

**Passing covariates:**

```python
# via function argument (one-time override)
run_single_its(df, intervention_date="2022-06-01", covariate_cols=["temp", "humidity"])

# via YAML config
# data:
#   covariate_cols: ["temp", "humidity"]
```

---

## Missing outcome values

By default, the pipeline raises a `ValueError` if the outcome column contains missing
values. Two automated strategies are available via the `data.missing_data` config key:

| Value | Behavior |
|-------|----------|
| `"error"` (default) | Raise `ValueError` on any missing outcome value |
| `"drop"` | Drop rows with missing outcome values before processing |
| `"interpolate"` | Linear interpolation of missing outcome values |

Set via `config_overrides={"data": {"missing_data": "drop"}}`, or impute externally
before calling `run_single_its`.

Note: the `"drop"` and `"interpolate"` strategies affect the temporal spacing of the
series. For models sensitive to regular spacing (ARIMA, NeuralProphet), prefer external
imputation with explicit handling.
