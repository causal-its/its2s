# Description: Drawing candidate settings from a family's search space. A
#   Latin hypercube spreads n draws evenly over every dimension at once, and
#   flat "section__name" keys are nested into the dicts a family reads. Both
#   the old tuning path and step 1's screening draw through here, so the same
#   seed gives the same settings on either path.
# Usage: from its2s.search import sample_settings, nest_params
# Dependencies: scipy

import math

from scipy.stats import qmc


def sample_settings(search_space: dict, n_trials: int, seed: int) -> list[dict]:
    """Return n_trials parameter dicts sampled via Latin hypercube.

    Each parameter is scaled from the unit hypercube to its declared range.
    Log-scale parameters are sampled log-uniformly (uniform in log space).
    Integer parameters are rounded and clamped to [low, high].
    """
    keys = list(search_space.keys())
    sampler = qmc.LatinHypercube(d=len(keys), seed=seed)
    unit_samples = sampler.random(n=n_trials)  # shape (n_trials, n_params)

    trials = []
    for row in unit_samples:
        params = {}
        for j, key in enumerate(keys):
            low, high, dtype, scale = search_space[key]
            u = float(row[j])
            if scale == "log":
                val = math.exp(math.log(low) + u * (math.log(high) - math.log(low)))
            else:
                val = low + u * (high - low)
            if dtype == "int":
                val = int(round(val))
                val = max(int(low), min(int(high), val))
            params[key] = val
        trials.append(params)

    return trials


def nest_params(flat_params: dict) -> dict:
    """Convert double-underscore-prefixed keys into nested dicts.

    {"xgb__max_depth": 8, "prophet__changepoint_prior_scale": 0.05}
    -> {"xgb": {"max_depth": 8}, "prophet": {"changepoint_prior_scale": 0.05}}

    Keys without "__" stay at the top level.
    """
    result = {}
    for key, val in flat_params.items():
        if "__" in key:
            section, param = key.split("__", 1)
            result.setdefault(section, {})[param] = val
        else:
            result[key] = val
    return result
