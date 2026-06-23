# Description: Block length selection for MBB (fixed default, auto selector, grid stub).
# Usage: from its2s.bootstrap.block_length import fixed_block_length, auto_block_length
# Dependencies: numpy; arch (auto_block_length, lazy import)

import math

import numpy as np


def fixed_block_length(L=14):
    """Return a fixed block length for the moving block bootstrap.

    The block length is measured in OBSERVATIONS (consecutive rows of the
    residual series), not calendar time. Its implied calendar span depends on
    the series frequency: L=14 covers 14 days on a daily series but 14 weeks on
    a weekly series. The default of 14 matches the case study in Dey (2025).

    Parameters
    ----------
    L : int
        Block length, in number of observations.

    Returns
    -------
    int
    """
    return L


def auto_block_length(residuals):
    """Automatic moving-block-bootstrap block length (Politis and White, 2004).

    Wraps ``arch.bootstrap.optimal_block_length`` and returns the optimal length
    for the CIRCULAR block bootstrap -- the variant whose block resampling matches
    the moving block bootstrap used in this package. The result is in OBSERVATIONS
    (see :func:`fixed_block_length`), rounded up and floored at 1.

    NOTE: this is the Politis-White (2004) automatic rule, a DIFFERENT estimator
    than the Lahiri (2007) nonparametric plug-in (NPPI) used to pick L=14 in
    Dey (2025). It may not return 14; use :func:`grid_search_block_length` to
    reproduce the paper's CI-width-stability selection.

    Parameters
    ----------
    residuals : array-like
        Model residuals to estimate serial dependence from. Non-finite values
        (e.g. AR-warmup NaNs) are dropped defensively; pass the warmup-excluded
        finite residual vector that the bootstrap resamples.

    Returns
    -------
    int
        Optimal block length in observations (>= 1).

    Raises
    ------
    ValueError
        If fewer than 2 finite residuals are available.

    References
    ----------
    Politis, D. N., and White, H. (2004). Automatic block-length selection for
    the dependent bootstrap. Econometric Reviews, 23(1), 53-70.
    """
    try:
        from arch.bootstrap import optimal_block_length
    except ImportError as exc:  # pragma: no cover - exercised only without arch
        raise ImportError(
            "auto_block_length requires the 'arch' package. Install it with "
            "`pip install arch` (or `conda install -c conda-forge arch-py`)."
        ) from exc

    resid = np.asarray(residuals, dtype=float)
    resid = resid[np.isfinite(resid)]
    if resid.size < 2:
        raise ValueError(
            "auto_block_length needs at least 2 finite residuals to estimate the "
            "optimal block length."
        )

    # optimal_block_length returns a DataFrame with 'stationary' and 'circular'
    # columns; use 'circular' (circular block bootstrap), the MBB-matching variant.
    opt = optimal_block_length(resid)
    b_circular = float(np.asarray(opt["circular"])[0])
    return max(1, int(math.ceil(b_circular)))


def grid_search_block_length(model, train_df, L_range=None, n_sim=50):
    """Select block length via grid search over a range of values.

    Placeholder for future implementation
    (references d_select_block_length_MBB.R from codebase X).

    Parameters
    ----------
    model : BaseModel
        Fitted model instance.
    train_df : pd.DataFrame
        Training data.
    L_range : list[int], optional
        Range of block lengths to evaluate.
    n_sim : int
        Number of bootstrap simulations per candidate L.

    Returns
    -------
    int
    """
    raise NotImplementedError(
        "Grid search block length selection not yet implemented. "
        "Use fixed_block_length() instead."
    )
