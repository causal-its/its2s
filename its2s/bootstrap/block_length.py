# Description: Block length selection for MBB (fixed default, auto selector, grid search).
# Usage: from its2s.bootstrap.block_length import (fixed_block_length,
#        auto_block_length, grid_search_block_length)
# Dependencies: numpy, pandas; arch (auto_block_length, lazy import);
#               its2s.bootstrap.mbb (grid_search_block_length, lazy import)

import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


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


def _select_plateau_index(widths, tol, window):
    """Find the smallest index where the CI-width curve plateaus.

    A plateau starts at index ``i`` when the relative change in width is below
    ``tol`` for ``window`` consecutive steps starting at ``i``. The relative
    change into index ``j`` is ``|widths[j] - widths[j-1]| / |widths[j-1]|``;
    non-finite changes (NaN/inf, e.g. from a zero-width baseline or a failed L)
    never satisfy the threshold.

    Parameters
    ----------
    widths : np.ndarray
        CI width at each candidate L, in L order (may contain NaN for failed L).
    tol : float
        Relative-change threshold.
    window : int
        Number of consecutive sub-threshold steps required.

    Returns
    -------
    tuple[int | None, np.ndarray]
        ``(selected_index, rel_change)``. ``selected_index`` is None if no
        sustained plateau exists. ``rel_change`` is the per-index relative change
        (``rel_change[0]`` is NaN), returned for diagnostics.
    """
    widths = np.asarray(widths, dtype=float)
    rel_change = np.full(widths.shape[0], np.nan)
    if widths.shape[0] < 2:
        return None, rel_change

    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.abs(np.diff(widths)) / np.abs(widths[:-1])
    rel_change[1:] = step

    # step[i] is the change INTO L_list[i+1], so index i is a plateau start if
    # step[i], ..., step[i+window-1] are all finite and < tol.
    last_start = widths.shape[0] - 1 - window
    for i in range(0, last_start + 1):
        block = step[i:i + window]
        if block.size == window and np.all(np.isfinite(block)) and np.all(block < tol):
            return i, rel_change
    return None, rel_change


def grid_search_block_length(model, train_df, target_df, L_range=None, n_sim=500,
                             target_col="y", date_col="ds", covariate_cols=None,
                             ci_level=0.95, tol=0.05, window=5, seed=None,
                             n_jobs=1, return_diagnostics=False):
    """Select block length by CI-width stability (Dey 2025, Figure S2 criterion).

    Ports the grid search in ``d_select_block_length_MBB.R``: for each candidate
    block length L, run the moving block bootstrap and record the width of the
    95% CI for the SUMMED counterfactual ("expected") counts over the event
    window. The chosen L is the SMALLEST L at which that width plateaus -- the
    relative change between consecutive widths stays below ``tol`` for ``window``
    consecutive steps. This is CI-width STABILITY, not variance minimization.

    The OG R script only plotted the width-vs-L curve and picked the plateau by
    eye (reference lines at L=14 and L=93); the plateau rule here formalizes that
    visual choice. Its parameters (``tol``, ``window``) are exposed for
    transparency and reproducibility.

    Because observed counts are fixed, the width of the summed-expected CI equals
    the width of the summed-excess CI, so this reproduces the OG ``CI_diff`` curve.

    NOTE: the same ``seed`` is used for every L so width differences reflect the
    block length, not Monte-Carlo noise (matching the fixed seed in the OG R).

    Parameters
    ----------
    model : BaseModel
        Fitted model instance (must already be fitted; the bootstrap refits
        clones internally for each simulation).
    train_df : pd.DataFrame
        Pre-intervention training data the model was fitted on.
    target_df : pd.DataFrame
        Event-window prediction frame. The CI width is computed on the SUM of the
        counterfactual predictions across ALL its rows, so pass the
        post-intervention event window (the OG ``df_period``), not the full
        pre+post frame.
    L_range : iterable[int], optional
        Candidate block lengths, in observations. Defaults to ``range(1, 51)``
        (1..50). The OG swept 1..150; the default covers the L~14 region at lower
        cost but would not reach a later plateau (e.g. the OG's L=93 reference).
    n_sim : int
        MBB simulations per candidate L. Defaults to 500 (the OG value); this is
        the dominant cost (``len(L_range) * n_sim`` model refits) -- lowering it
        speeds the search but makes the width curve noisier.
    target_col, date_col, covariate_cols
        Column roles passed through to the bootstrap.
    ci_level : float
        Confidence level for the width measured at each L (default 0.95).
    tol : float
        Plateau threshold on the relative change in CI width between consecutive
        L (default 0.05 = 5%).
    window : int
        Number of consecutive steps the relative change must stay below ``tol``
        for a plateau to be declared (default 5).
    seed : int, optional
        Seed reused across all L for a clean, reproducible width curve.
    n_jobs : int
        Parallel jobs passed to the bootstrap.
    return_diagnostics : bool
        If True, return ``(L, diagnostics)`` where ``diagnostics`` is a DataFrame
        with columns ``L, ci_lo, ci_hi, ci_width, rel_change`` (the Figure S2
        evidence). If False (default), return only the selected int L.

    Returns
    -------
    int or tuple[int, pandas.DataFrame]
        The selected block length, or ``(L, diagnostics)`` if
        ``return_diagnostics`` is True.

    Raises
    ------
    ValueError
        If ``L_range`` is empty or contains a non-positive value.

    Warns
    -----
    UserWarning
        If no sustained plateau is found within ``L_range``; the LARGEST L is
        returned as a conservative fallback (largest L preserves the most
        residual autocorrelation -> widest CIs -> safest against undercoverage).

    References
    ----------
    Dey et al. (2025), "Two-Stage Interrupted Time Series Analysis with Machine
    Learning" (p.6, Figure S2): L=14 confirmed by a grid search selecting the
    smallest L with stable 95% CI width.
    """
    import warnings

    from .mbb import MovingBlockBootstrap

    if L_range is None:
        L_range = range(1, 51)
    L_list = [int(L) for L in L_range]
    if len(L_list) == 0:
        raise ValueError("L_range must contain at least one candidate block length.")
    if any(L < 1 for L in L_list):
        raise ValueError("All candidate block lengths must be >= 1.")

    alpha = 1 - ci_level
    lo_q = 100 * alpha / 2
    hi_q = 100 * (1 - alpha / 2)

    widths = np.full(len(L_list), np.nan)
    lows = np.full(len(L_list), np.nan)
    highs = np.full(len(L_list), np.nan)

    for idx, L in enumerate(L_list):
        mbb = MovingBlockBootstrap(n_sim=n_sim, block_length=L,
                                   ci_method="quantile", ci_level=ci_level,
                                   n_jobs=n_jobs)
        result = mbb.generate_cis(model, train_df, target_df,
                                  target_col=target_col, date_col=date_col,
                                  covariate_cols=covariate_cols, seed=seed)
        # Sum the counterfactual predictions across the event window per
        # simulation, then take the CI of that total (matches the OG df_period
        # expected_low / expected_up).
        sim_totals = np.nansum(result.pred_matrix, axis=0)
        sim_totals = sim_totals[np.isfinite(sim_totals)]
        if sim_totals.size == 0:
            logger.warning("Grid search: L=%d produced no usable simulations.", L)
            continue
        lo = float(np.nanpercentile(sim_totals, lo_q))
        hi = float(np.nanpercentile(sim_totals, hi_q))
        lows[idx] = lo
        highs[idx] = hi
        widths[idx] = hi - lo
        logger.info("Grid search: L=%d / %d -> CI width %.4g.",
                    L, L_list[-1], widths[idx])

    selected_idx, rel_change = _select_plateau_index(widths, tol, window)

    if selected_idx is None:
        selected_idx = len(L_list) - 1
        warnings.warn(
            f"No sustained CI-width plateau found over L_range "
            f"(tol={tol}, window={window}); returning the largest candidate "
            f"L={L_list[selected_idx]} as a conservative fallback (widest CIs). "
            "Inspect the width-vs-L curve and consider widening L_range or "
            "relaxing tol.",
            UserWarning,
            stacklevel=2,
        )

    selected_L = L_list[selected_idx]
    logger.info("Grid search selected block length L=%d.", selected_L)

    if return_diagnostics:
        diagnostics = pd.DataFrame({
            "L": L_list,
            "ci_lo": lows,
            "ci_hi": highs,
            "ci_width": widths,
            "rel_change": rel_change,
        })
        return selected_L, diagnostics
    return selected_L
