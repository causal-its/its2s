# API Reference

Public entrypoints for the package. Each section below is generated from the live docstrings in the source code.

## Pipeline

::: its2s.run_single_its

::: its2s.run_batch

## Block length selection

The Moving Block Bootstrap block length is measured in observations (residual rows),
not calendar days; its implied calendar span depends on the series frequency. Set
`bootstrap.block_length` in the config to a fixed int (default `14`) or to `"auto"`
(the Politis-White rule, resolved at run time from the model residuals). To reproduce
the paper's CI-width-stability selection, run `calibrate_block_length` once and set
`bootstrap.block_length` to the integer it returns.

::: its2s.calibrate_block_length

::: its2s.bootstrap.block_length.auto_block_length

::: its2s.bootstrap.block_length.grid_search_block_length

::: its2s.bootstrap.block_length.fixed_block_length

::: its2s.bootstrap.block_length.resolve_block_length

## Cross-validation and tuning

::: its2s.time_series_cv

::: its2s.tune_model

::: its2s.TuningResult

## Comparison

::: its2s.compare_models

## Configuration

::: its2s.load_config