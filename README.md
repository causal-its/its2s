# its2s

`its2s` is an open-source Python package providing a modular framework for a two-stage interrupted time series (ITS) analysis pipeline using machine learning models. It includes tools for data splitting, cross-validation, hyperparameter tuning, different model types, moving-block bootstrap confidence intervals, and a full end-to-end workflow. For full reference material and documentation, please visit the [its2s website](https://causal-its.github.io/its2s/).

## All documentation, tutorials, and reference material

- **[Setup](https://causal-its.github.io/its2s/installation/)** — install the package into a Python environment.
- **[Quick Start](https://causal-its.github.io/its2s/quickstart/)** — minimal example using simulated data.
- **[Tutorials](https://causal-its.github.io/its2s/tutorials/step1_data_splitting/)** — six step-by-step notebooks.
- **[API Reference](https://causal-its.github.io/its2s/api/)** — public API docstrings.

## Installation quick reference

From the repository root:

```bash
conda env create -f environment.yml
conda activate its2s
```

…or, without conda, `python -m venv .venv && source .venv/bin/activate && pip install -e .`

## Public API quick reference

```python
from its2s import run_single_its, run_batch, compare_models, tune_model
```

## Getting help

Open an [issue](https://github.com/causal-its/its2s/issues) for bugs, questions, or feature requests.