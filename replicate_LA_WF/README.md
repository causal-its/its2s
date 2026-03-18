# replicate_LA_WF

Replicates an LA Wildfire ITS analysis using the its2s Python package.

## Input files

Place the following in `path_project/replicate_LA_WF/LA_wildfire_files/`:

- `df-predict-sf.parquet` -- full dataset including the holdout period
- `performance_metrics_*.csv` -- tuning results from the R model run

## Steps

### Step 1 -- Convert best tuning params to YAML

```
python 03_build_best_params.py <performance_metrics_csv>
```

Reads the R tuning CSV, converts hyperparameters to its2s format, and writes
`best_params.yaml` next to this script. Run `python 03_build_best_params.py --help`
for options (e.g. `--r-features`, `--out`).

### Step 2 -- Run ITS

```
python 02_run_its.py --best-params-yaml best_params.yaml
```

Runs the ITS pipeline for all series found in the parquet and saves results
under `path_project/replicate_LA_WF/<model>/`. Use `--enc-type` or `--exposure`
to restrict to a subset, and `--n-sim` to override the bootstrap count.

### Optional -- Validate input data

```
python 01_prep_data.py
```

Checks the parquet structure, date coverage, and data splits before running.

## Config

`config_replicate.yaml` controls bootstrap settings (n_sim, block length, CI
method) and holdout period definition. Edit this file to match a different
analysis configuration.
