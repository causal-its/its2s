# Description: Load and merge YAML configuration with defaults.
# Usage: from its2s.settings import load_config, get_model_config
# Dependencies: pyyaml

import copy
from pathlib import Path

import yaml

# params.yaml holds default (fallback) values, not hyperparameter search ranges.
# Tuning search spaces live in tuning.py (_ARIMA_SPACE, _NEURALPROPHET_SPACE, etc.).
_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "params.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path=None, overrides=None):
    """Load YAML config, merge with defaults, apply overrides.

    Parameters
    ----------
    path : str or Path, optional
        Path to a custom YAML config. Merged on top of defaults.
    overrides : dict, optional
        Runtime overrides applied last.

    Returns
    -------
    dict
    """
    with open(_DEFAULT_CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    if path is not None:
        with open(path) as f:
            user_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, user_config)

    if overrides:
        config = _deep_merge(config, overrides)

    return config


def get_model_config(config, model_name):
    """Extract model-specific config section.

    Parameters
    ----------
    config : dict
    model_name : str

    Returns
    -------
    dict
    """
    return copy.deepcopy(config.get("models", {}).get(model_name, {}))


def get_tuning_config(config):
    """Extract the tuning defaults section.

    Parameters
    ----------
    config : dict

    Returns
    -------
    dict
    """
    return copy.deepcopy(config.get("tuning", {}))
