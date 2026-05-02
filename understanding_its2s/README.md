# understanding_its2s

Explanatory notebooks for the `its2s` package. Each notebook is self-contained
and runs against the packaged dummy series in `data/dummy_data.csv` (daily,
2018-01-01 to 2022-04-25; simulated intervention on 2022-03-15 with a
42-day post-intervention effect window of +8/day).

## Recommended reading order

The notebooks follow the real workflow of a single ITS analysis: split the
data, design the validation scheme, tune hyperparameters, fit the final
model, attach uncertainty, then compose. Read them in numeric order.

| # | Notebook | What it covers |
|---|----------|----------------|
| 1 | `step1_data_splitting.ipynb` | `prepare_splits()`: carving the series into train / test / holdout around the intervention. |
| 2 | `step2_cross_validation.ipynb` | `time_series_cv()`: expanding-window CV, fold layout, `skip_days`, `cv_end_date`, and why temporal leakage matters. |
| 3 | `step3_hyperparameter_tuning.ipynb` | `tune_model()`: Latin hypercube search composed on top of step 2's CV. Demo on `prophet_xgb`; same mechanics apply to the other three models. |
| 4a | `step4a_model_prophet_xgb.ipynb` | Fit the Prophet + XGB **hybrid** (XGB on Prophet residuals). |
| 4b | `step4b_model_prophet_then_xgb.ipynb` | Fit the Prophet-then-XGB **sequential** model (XGB corrects a standalone Prophet forecast). |
| 4c | `step4c_model_neuralprophet.ipynb` | Fit `NeuralProphetModel`; discuss AR warmup and its effect on the MBB residual pool. |
| 4d | `step4d_model_arima.ipynb` | Fit `ARIMAModel` via `run_single_its()`; inspect `FitResult` and pipeline outputs. |
| 5 | `step5_bootstrap_mbb.ipynb` | Moving Block Bootstrap: residual blocks, `pred_matrix`, CI method choice, and `block_length` sensitivity. |
| 6 | `step6_full_workflow.ipynb` | Capstone: split -> CV sanity check -> tune -> fit -> MBB -> compare, end-to-end on the dummy series. |

The four step-4 notebooks are parallel variants; pick whichever architecture
you need first and come back to the others as required.

## Running the notebooks

The notebooks assume the `its2s` package is installed in the active
environment (`pip install -e .` from the repo root). Each notebook writes
figures into `figures/` alongside this README.

## Relationship to the package source

These notebooks reflect the state of `its2s/` including the hyperparameter
tuning framework (`tuning.py`) and the cross-validation fixes to
`cross_validation.py` (non-overlapping fold layout, `skip_days`,
`cv_end_date`). They are explanatory and do not themselves exercise the
package's test suite; see `tests/test_its2s.py` and `tests/test_tuning.py`
for that.
