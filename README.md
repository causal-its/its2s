# its2s

Modular interrupted time series (ITS) counterfactual analysis with moving-block bootstrap confidence intervals.

This README covers **standalone use**: install the package into a Python environment, point it at your own tabular dataset (e.g. Parquet or CSV), and optionally save plots and tables to a folder you choose.

## Install

From the directory that contains this file (the repository root, where `pyproject.toml` lives):

```bash
pip install .
```

Editable install (changes to the source code are visible without reinstalling):

```bash
pip install -e .
```

Use the **same virtual environment** whenever you run your scripts. The folder you run from can be anywhere on your machine; imports work because `its2s` is installed into that environment, not because your script lives next to this repo.

Dependencies are declared in `pyproject.toml`. You can also mirror them with `pip install -r requirements.txt` if you use that file elsewhere.

## Minimal script

```python
from pathlib import Path

import pandas as pd
from its2s import run_single_its

# --- Input path: anywhere you like (absolute or relative to how you launch Python) ---
data_path = Path("/path/to/your/project/dummy_data.parquet")
df = pd.read_parquet(data_path)

result = run_single_its(
    df,
    intervention_date="2022-01-15",
    model_name="prophet_xgb",
    output_dir=Path("/path/to/your/project/its2s_outputs"),
    seed=42,
)
```

If `output_dir` is omitted, the pipeline still runs and returns a `PipelineResult`; **no files are written**.

## Input data: paths and column names

**Where you specify the input path**

- The package does not take a dataset path as a built-in CLI argument. You load the table yourself with pandas (or Polars, then `.to_pandas()`), then pass the `DataFrame` to `run_single_its`.
- Use any path that makes sense for your project: `Path(__file__).resolve().parent / "dummy_data.parquet"`, an absolute path, or an environment variable.

**Expected columns (defaults)**

Default settings assume:

| Role        | Default column name |
|------------|----------------------|
| Time index | `ds`                 |
| Outcome    | `y`                  |

Defaults come from `its2s/params.yaml` under `data:`. To use different names without editing that file, pass arguments:

```python
run_single_its(
    df,
    intervention_date="2022-01-15",
    date_col="my_date",
    target_col="outcome",
)
```

The date column is parsed as datetime and the series is sorted by time inside the pipeline.

## Output paths

**Where you specify the output location:** pass `output_dir` as a string or `pathlib.Path`. The directory is created if it does not exist (`parents=True`).

**Files written** (names include the `model_name` you passed, e.g. `prophet_xgb`):

| File | Description |
|------|-------------|
| `{model_name}_counterfactual.png` | Counterfactual plot |
| `{model_name}_excess.csv` | Excess (observed vs counterfactual) table |
| `{model_name}_metrics.csv` | Train/test error metrics |
| `{model_name}_ate_summary.csv` | Average treatment effect–style summary (written when daily excess is non-empty) |

All paths are under the single `output_dir` you provide.

## Covariates for fitting and prediction

Covariates are **extra numeric (or otherwise model-supported) columns** in the same `DataFrame` as `ds` and `y`. They are passed through to model `fit`, `predict`, and the moving-block bootstrap so they participate in **both** pre- and post-intervention predictions where the model uses them.

**Two equivalent ways to specify them**

1. **In code** (overrides the default list for that run):

   ```python
   run_single_its(
       df,
       intervention_date="2022-01-15",
       covariate_cols=["temperature", "humidity", "holiday"],
   )
   ```

2. **In YAML** copied or merged with the defaults — set `data.covariate_cols` in a file you pass as `config_path`:

   ```yaml
   data:
     date_col: "ds"
     target_col: "y"
     covariate_cols: ["temperature", "humidity", "holiday"]
   ```

   ```python
   run_single_its(
       df,
       intervention_date="2022-01-15",
       config_path=Path("/path/to/my_its_config.yaml"),
   )
   ```

**Requirements**

- Every name in `covariate_cols` must exist as a column in `df`.
- Values should be defined for all rows that fall in the **test + holdout** window used internally (from `periods.test_days` before the intervention through `periods.holdout_days` after), so prediction and bootstrap over that horizon have the covariate path. Missing handling is model-specific; avoid unexpected NaNs in those windows unless your chosen model tolerates them.

To tune horizons or bootstrap settings, merge a YAML file via `config_path` or pass a nested dict as `config_overrides` (see `run_single_its` in `its2s/pipeline.py`). The built-in default YAML is `its2s/params.yaml` inside the installed package.

## Configuration and models

- **`config_path`**: optional YAML merged on top of package defaults (`its2s/params.yaml`).
- **`config_overrides`**: optional dict merged last (highest priority).
- **`model_name`**: one of `arima`, `prophet_xgb`, `prophet_then_xgb`, `neuralprophet` (subject to optional dependencies).

## Public API

```python
from its2s import run_single_its, run_batch, load_config
```

For batch or replicate workflows, see `replicate_LA_WF/README.md` in this repository.
