# Description: Deterministic seed derivation using xxhash.
# Usage: from its2s.batch.seed_manager import derive_seed
# Dependencies: xxhash

import xxhash


def derive_seed(global_seed, *markers):
    """Derive a deterministic integer seed from a global seed and string markers.

    Replicates the gen_seed pattern: hash the concatenation of
    global_seed and all marker strings to produce a reproducible 32-bit integer.

    Parameters
    ----------
    global_seed : int
        Base seed for the analysis run.
    *markers : str
        Additional identifiers (e.g., series_id, model_name) to make the
        seed unique per combination.

    Returns
    -------
    int
        A non-negative 32-bit integer seed.
    """
    key = "|".join([str(global_seed)] + [str(m) for m in markers])
    return xxhash.xxh32(key.encode()).intdigest()
