# Description: Block length selection for MBB (fixed default + stubs).
# Usage: from its2s.bootstrap.block_length import fixed_block_length
# Dependencies: none


def fixed_block_length(L=14):
    """Return a fixed block length (default = 14 days / 2 weeks).

    Parameters
    ----------
    L : int
        Block length.

    Returns
    -------
    int
    """
    return L


def nppi_block_length(residuals):
    """Estimate optimal block length via NPPI method.

    Placeholder for future implementation (references blocklength::nppi in R).

    Parameters
    ----------
    residuals : array-like
        Model residuals.

    Returns
    -------
    int
    """
    raise NotImplementedError(
        "NPPI block length estimation not yet implemented. "
        "Use fixed_block_length() or grid_search_block_length() instead."
    )


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
