# FAQ

## Installation and setup

**Prophet is hanging on first import. What is happening?**

Prophet compiles a Stan model the first time it is imported. This compilation step
takes approximately 5–10 minutes and only happens once per environment. Subsequent
imports are fast. If the hang persists beyond 15 minutes, check that CmdStan installed
correctly: `import cmdstanpy; cmdstanpy.cmdstan_path()`.

---

**I get an XGBoost error about `libomp.dylib` not loaded on macOS with conda.**

This affects Intel Mac users who install the package via the conda path in the
[Setup guide](installation.md). The conda environment uses `pip install -e .` to
install dependencies, and the pip-distributed XGBoost binary on Intel Macs links
against OpenMP at a Homebrew path (`/usr/local/opt/libomp/`) that conda environments
cannot see. Running `brew install libomp` does not fix it — Homebrew and conda manage
libraries independently, and the library ends up in a location the conda environment
never searches.

After hitting the error, replace the pip-installed XGBoost with the conda-forge build,
which bundles OpenMP internally:

```bash
conda install -c conda-forge xgboost
```

If that does not resolve it, install OpenMP directly into the conda environment:

```bash
conda install -c conda-forge "libcxx<17"
```

Note: this issue does not affect Apple Silicon (M-series) Macs or users following the
`venv`-based installation path.

---

**I get an `ImportError` for NeuralProphet or PyTorch.**

NeuralProphet is an optional dependency (~1 GB including PyTorch) and is not installed
by default. Install it with:

```bash
pip install its2s[neural]
```

---

**I get an error about `pkg_resources` not found.**

The package pins `setuptools>=68,<71` because NeuralProphet uses `pkg_resources`,
which was removed in setuptools 71+. Check your setuptools version:

```bash
pip show setuptools
```

If it is 71 or higher, downgrade: `pip install "setuptools>=68,<71"`.

---

## Running the pipeline

**How do I know if my counterfactual is credible?**

Two checks in order of importance:

1. **Visual**: open `{model}_counterfactual.png` and inspect the pre-event period.
   The observed and expected lines should track closely before the event. Divergence
   before the event means the model did not capture the baseline trend.

2. **Numeric**: check `{model}_metrics.csv`. A low test R² (e.g., below 0.5) or a test
   RMSE that is large relative to the outcome's range signals poor generalization. The
   test window performance — not training performance — is the relevant signal.

If either check fails, try a different model via `compare_models()` or add covariates
that help explain the baseline trend.

---

**What block length should I use for the Moving Block Bootstrap?**

The default `block_length=14` is appropriate for daily data with moderate
autocorrelation (approximately two weeks of temporal dependence). It is not
automatically adapted to other data configurations.

For non-daily data or series with substantially different autocorrelation structure,
the default may produce CI coverage that is too narrow or too wide. Adjust via
`config_overrides={"bootstrap": {"block_length": <value>}}`. Automated block length
selection is not currently implemented.

---

**My bootstrap is producing many "simulation failure" warnings.**

Warnings about failed bootstrap simulations typically mean the model failed to converge
on some resampled series. Common causes:

- Training window is too short for reliable refitting on bootstrap resamples.
- The series contains extreme outliers that cause numerical instability.
- NeuralProphet training stochasticity on short windows.

If more than ~10% of simulations fail (the package warns at 50%), the CI coverage may
be unreliable. Check whether the full series has enough data and whether outliers should
be handled before running.

---

**Can I use this package on weekly or monthly data?**

Yes, but configuration adjustments are required:

- Set `m` for ARIMA: `config_overrides={"models": {"arima": {"m": 52}}}` for weekly,
  `{"m": 12}` for monthly.
- Series frequency itself is resolved automatically from the date column; the series
  must be a complete, regularly spaced grid (no gaps or duplicate dates), or the
  pipeline raises an error naming the first offending timestamp.
- Consider adjusting `block_length` for the MBB (default 14 is calibrated for daily
  data; block length is measured in observations, so 14 means 14 weeks on a weekly
  series).

---

**What is the minimum series length?**

There is no hard minimum, but the series must be long enough to:

1. Support the cross-validation framework (`min_train_days` + at least one fold of
   `test_days` + `skip_days`).
2. Provide a reliable test window (`test_days`, default 365 days) for model selection.

Series shorter than about two years of daily observations will typically produce
unstable tuning and cross-validation results. Shorter series are better served by
simpler models (ARIMA) with reduced CV requirements.

---

## Data

**My covariate is missing in part of the post-event window.**

Covariates must be present for every row, including the entire post-event projection
window. Options:

- Shorten `holdout_days` to the period for which the covariate is available.
- Drop the covariate if it cannot be extended.
- Impute the post-event covariate externally (only appropriate if the imputed values
  would not be affected by the event itself).

---

**I have missing values in my outcome column.**

By default, the pipeline raises a `ValueError` on missing outcome values. Two
automated strategies are available:

```python
# drop rows with missing outcome
run_single_its(df, ..., config_overrides={"data": {"missing_data": "drop"}})

# linear interpolation
run_single_its(df, ..., config_overrides={"data": {"missing_data": "interpolate"}})
```

For models sensitive to regular spacing (ARIMA, NeuralProphet), external imputation
with explicit handling is preferred.

---

**How do I switch to date-based splitting?**

The default `split_method="percent"` sizes test/holdout windows as fractions of the
available data. To pin the windows to fixed calendar durations instead:

```python
run_single_its(
    df, intervention_date="2022-06-01",
    config_overrides={
        "periods": {
            "split_method": "days",
            "test_days": 365,
            "holdout_days": 365,
        },
    },
)
```

In `"days"` mode, `test_days` must be strictly less than the number of pre-intervention
observations or the pipeline raises `ValueError` (a previously silent failure mode where
the training split came back empty).

---

## Configuration

**What is the `config_overrides` format?**

`config_overrides` is a nested dict that mirrors the structure of `params.yaml`. It is
merged on top of the package defaults at the highest priority:

```python
config_overrides = {
    "periods": {"test_days": 180, "holdout_days": 90},
    "bootstrap": {"n_sim": 500, "block_length": 14},
    "models": {"arima": {"m": 52}},
}
```

Inside `tune_model`, the search space also accepts double-underscore flattened keys
for nested model parameters (e.g., `"xgb__max_depth"` for XGBoost's `max_depth`).
