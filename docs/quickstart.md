# Quick Start

## Minimal example using simulated data

The pipeline expects a sorted time index and outcome column (defaults: `ds`, `y`). Defaults in `its2s/params.yaml` size the pre-intervention test window as 20% of the pre-event data and use the full post-event period as the holdout, with 1000 bootstrap draws—fine for real analyses but slow for a smoke test. The example below builds a fake daily series and passes **`config_overrides`** to pin the windows to fixed day counts and shrink the bootstrap so it finishes in a reasonable time.

```python
import numpy as np
import pandas as pd
from its2s import run_single_its

rng = np.random.default_rng(42)
intervention_date = "2022-06-01"

# using shorter windows + fewer bootstrap draws than package defaults for a faster demo.
# split_method="days" opts out of the percent default so test_days/holdout_days are honored.
config_overrides = {
    "periods": {"split_method": "days", "test_days": 90, "holdout_days": 90},
    "bootstrap": {"n_sim": 100},
}

dates = pd.date_range("2020-01-01", "2022-09-30", freq="D")
n = len(dates)
t = np.arange(n)
seasonal = 10 * np.sin(2 * np.pi * t / 365.25)
trend = 0.02 * t
noise = rng.normal(0, 2, n)
y = 100 + trend + seasonal + noise

# small level shift after the intervention so the excess is non-trivial.
intervention = pd.Timestamp(intervention_date)
y = y.astype(float)
y[dates >= intervention] += 5

df = pd.DataFrame({"ds": dates, "y": y})

result = run_single_its(
    df,
    intervention_date=intervention_date,
    model_name="prophet_xgb",
    config_overrides=config_overrides,
    seed=42,
)
```

Omit `output_dir` if you only need the returned results within your session. To save outputs:

```python
from pathlib import Path

result = run_single_its(
    df,
    intervention_date=intervention_date,
    model_name="prophet_xgb",
    config_overrides=config_overrides,
    output_dir=Path("./its2s_outputs"),
    seed=42,
)
```

## Your own data

**Loading:** The package does not read paths for you. Load with pandas, then pass the `DataFrame` to `run_single_its`. Paths can be absolute, relative to the script, or from an environment variable—whatever fits your project.

**Columns (defaults):**

| Role        | Default column name |
|-------------|---------------------|
| Time index  | `ds`                |
| Outcome     | `y`                 |

Defaults come from `its2s/params.yaml` under `data:`. To rename without editing that file:

```python
run_single_its(
    df,
    intervention_date="2022-06-01",
    date_col="my_date",
    target_col="outcome",
)
```

The date column is parsed as datetime and the series is sorted by time inside the pipeline.

## Outputs

Pass `output_dir` as a string or `pathlib.Path`; the directory is created if needed (`parents=True`).

**Files written** (names include `model_name`, e.g. `prophet_xgb`):

| File | Description |
|------|-------------|
| `{model_name}_counterfactual.png` | Counterfactual plot |
| `{model_name}_excess.csv` | Excess (observed vs counterfactual) table |
| `{model_name}_metrics.csv` | Train/test error metrics |
| `{model_name}_ate_summary.csv` | ATE-style summary (when the excess table is non-empty) |

All paths sit under the single `output_dir` you provide.

## Covariates for fitting and prediction

Covariates are **extra numeric (or otherwise model-supported) columns** in the same `DataFrame` as `ds` and `y`. They are passed through to model `fit`, `predict`, and the moving-block bootstrap so they participate in **both** pre- and post-intervention predictions where the model uses them.

**Two equivalent ways to specify them**

1. **In code** (overrides the default list for that run):

    ```python
    run_single_its(
        df,
        intervention_date="2022-06-01",
        covariate_cols=["temperature", "humidity", "holiday"],
    )
    ```

2. **In YAML** merged with defaults — set `data.covariate_cols` in a file you pass as `config_path`:

    ```yaml
    data:
      date_col: "ds"
      target_col: "y"
      covariate_cols: ["temperature", "humidity", "holiday"]
    ```

    ```python
    run_single_its(
        df,
        intervention_date="2022-06-01",
        config_path=Path("/path/to/my_its_config.yaml"),
    )
    ```

**Requirements**

- Every name in `covariate_cols` must exist as a column in `df`.
- Values should be defined for all rows in the **test + holdout** window (whose lengths are derived from `periods.test_pct` / `periods.holdout_pct` by default, from `periods.test_days` / `periods.holdout_days` when `split_method="days"`, or from `periods.test_obs` / `periods.holdout_obs` when `split_method="observations"`), so prediction and bootstrap over that horizon have the covariate path. Missing handling is model-specific; avoid unexpected NaNs in those windows unless your chosen model tolerates them.

To change any pipeline setting — window lengths, number of bootstrap draws, or model hyperparameters — use `config_path` (a YAML file) or `config_overrides` (a Python dict). Both are layered on top of the package defaults, which live in `its2s/params.yaml`. See [Configuration and models](#configuration-and-models) below.

## Configuration and models

- **`config_path`**: optional YAML merged on top of package defaults (`its2s/params.yaml`).
- **`config_overrides`**: optional dict merged last (highest priority).
- **`model_name`**: one of `prophet_xgb`, `neuralprophet`, `arima` (subject to optional dependencies).
