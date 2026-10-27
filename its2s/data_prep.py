# Description: Training/test/post-event splitting for ITS analysis.
# Usage: from its2s.data_prep import prepare_splits
# Dependencies: pandas

import logging
import warnings
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

# Arguments that apply to each split method. Passing an argument that belongs
# to a different method is an error, never silently ignored (GH #28, #54).
_METHOD_ARGS = {
    "percent": ("test_pct", "post_event_pct"),
    "days": ("test_days", "post_event_days"),
    "observations": ("test_obs", "post_event_obs"),
}

# Config keys retired by the window rename (D-101). Reading only the current
# keys would let an old config fall back to the defaults without a word, so
# each retired key raises and names its replacement. Retires with the YAML
# configuration file (contract P11.04, section 11).
_RETIRED_KEYS = {
    "holdout_pct": "post_event_pct",
    "holdout_days": "post_event_days",
    "holdout_obs": "post_event_obs",
}


@dataclass
class TimeSeriesSplits:
    """Container for the ITS training, test and post-event windows."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    post_event_df: pd.DataFrame
    full_predict_df: pd.DataFrame
    intervention_date: pd.Timestamp


def resolve_split_config(periods_cfg, split_method=None):
    """Resolve (split_method, split_kwargs) from a config "periods" section.

    Only the arguments belonging to the resolved method are read and passed
    on, and config keys from a different method family raise here, mirroring
    prepare_splits' guard on its own keywords: a window setting the user
    wrote must never be silently ignored (GH #28, #54). The ``split_method``
    argument overrides the config key. ``periods_cfg`` is never mutated.

    Parameters
    ----------
    periods_cfg : dict
        The "periods" section of a loaded config.
    split_method : str, optional
        Override for ``periods_cfg["split_method"]``.

    Returns
    -------
    tuple[str, dict]
        The resolved split method and the keyword arguments to pass to
        ``prepare_splits`` for that method.
    """
    retired = {k: v for k, v in _RETIRED_KEYS.items() if k in periods_cfg}
    if retired:
        raise ValueError(
            f"periods config keys {sorted(retired)} were renamed; use "
            f"{[retired[k] for k in sorted(retired)]} instead."
        )
    method = (split_method if split_method is not None
              else periods_cfg.get("split_method", "percent"))
    if method in _METHOD_ARGS:
        foreign = [k for fam, keys in _METHOD_ARGS.items() if fam != method
                   for k in keys if k in periods_cfg]
        if foreign:
            raise ValueError(
                f"periods config keys {foreign} do not apply to "
                f"split_method={method!r}, which uses "
                f"{list(_METHOD_ARGS[method])}. Set the keys for the chosen "
                "split_method only; they are never silently ignored."
            )
    if method == "days":
        split_kwargs = {
            "test_days": periods_cfg.get("test_days", 365),
            "post_event_days": periods_cfg.get("post_event_days", 365),
        }
    elif method == "observations":
        split_kwargs = {
            "test_obs": periods_cfg.get("test_obs"),
            "post_event_obs": periods_cfg.get("post_event_obs"),
        }
    else:
        # "percent" is the default; an unknown method raises in prepare_splits.
        split_kwargs = {
            "test_pct": periods_cfg.get("test_pct", 0.20),
            "post_event_pct": periods_cfg.get("post_event_pct", 1.0),
        }
    return method, split_kwargs


def prepare_splits(df, intervention_date, date_col="ds",
                   split_method="percent",
                   test_pct=None, post_event_pct=None,
                   test_days=None, post_event_days=None,
                   test_obs=None, post_event_obs=None,
                   min_test_obs=30):
    """Split a time series DataFrame into training, test, and post-event windows.

    Window units are explicit per method. "percent" and "observations" size
    windows in observations (rows of the regular series); "days" sizes them in
    calendar days via ``pd.Timedelta``, so on a weekly series ``test_days=365``
    spans about 52 observations. Only the arguments belonging to the chosen
    ``split_method`` may be passed; arguments for another method raise
    ValueError rather than being silently ignored.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with at least a date column and target column.
    intervention_date : str or pd.Timestamp
        Date of the intervention.
    date_col : str
        Name of the date column.
    split_method : {"percent", "days", "observations"}
        "percent" (default): size the test/post-event windows as fractions of the
        pre-/post-intervention observation counts.
        "days": use explicit calendar-day windows `test_days`/`post_event_days`.
        "observations": use explicit observation-count windows
        `test_obs`/`post_event_obs`.
    test_pct : float
        Fraction of pre-intervention observations used as the test window.
        Defaults to 0.20. Only with ``split_method="percent"``.
    post_event_pct : float
        Fraction of post-intervention observations used as the post-event window.
        Defaults to 1.0. Only with ``split_method="percent"``.
    test_days : int
        Calendar days before the intervention used as the test window.
        Defaults to 365. Only with ``split_method="days"``.
    post_event_days : int
        Calendar days after the intervention used as the post-event window.
        Defaults to 365. Only with ``split_method="days"``.
    test_obs : int
        Number of observations before the intervention used as the test
        window. Required with ``split_method="observations"``; no default.
    post_event_obs : int
        Number of observations from the intervention onward used as the
        post-event window. Required with ``split_method="observations"``; no
        default.
    min_test_obs : int
        Warn when the realized test window has fewer observations than this,
        whatever the split method (GH #29). Applies to every method; set to 0
        to disable. Defaults to 30, a conservative floor below which test
        metrics such as MAPE are unstable and can mislead model selection.

    Returns
    -------
    TimeSeriesSplits
    """
    if split_method not in _METHOD_ARGS:
        raise ValueError(
            f"split_method must be 'percent', 'days', or 'observations', "
            f"got {split_method!r}."
        )

    passed = {"test_pct": test_pct, "post_event_pct": post_event_pct,
              "test_days": test_days, "post_event_days": post_event_days,
              "test_obs": test_obs, "post_event_obs": post_event_obs}
    allowed = _METHOD_ARGS[split_method]
    foreign = [name for name, value in passed.items()
               if value is not None and name not in allowed]
    if foreign:
        raise ValueError(
            f"Arguments {foreign} do not apply to split_method="
            f"{split_method!r}, which uses {list(allowed)}. Pass the "
            "arguments for the chosen split_method only."
        )

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    intervention_date = pd.Timestamp(intervention_date)

    pre_rows = df[df[date_col] < intervention_date]
    post_rows = df[df[date_col] >= intervention_date]
    n_pre = len(pre_rows)
    n_post = len(post_rows)

    if split_method == "days":
        test_days = 365 if test_days is None else test_days
        post_event_days = 365 if post_event_days is None else post_event_days
        # A non-positive window would place test_start at or beyond the
        # intervention, silently pulling post-event rows into training.
        if test_days < 1 or post_event_days < 1:
            raise ValueError(
                f"test_days and post_event_days must be >= 1, got "
                f"test_days={test_days}, post_event_days={post_event_days}."
            )
        test_start = intervention_date - pd.Timedelta(days=test_days)
        post_event_end = intervention_date + pd.Timedelta(days=post_event_days)
        train_df = df[df[date_col] < test_start].copy()
        test_df = df[(df[date_col] >= test_start)
                     & (df[date_col] < intervention_date)].copy()
        post_event_df = df[(df[date_col] >= intervention_date)
                           & (df[date_col] <= post_event_end)].copy()
    else:
        if split_method == "percent":
            test_pct = 0.20 if test_pct is None else test_pct
            post_event_pct = 1.0 if post_event_pct is None else post_event_pct
            if not (0 < test_pct <= 1) or not (0 < post_event_pct <= 1):
                raise ValueError(
                    f"test_pct and post_event_pct must be in (0, 1], got "
                    f"test_pct={test_pct}, post_event_pct={post_event_pct}."
                )
            n_test = max(1, int(round(test_pct * n_pre)))
            n_post_event = max(1, int(round(post_event_pct * n_post)))
        else:
            if test_obs is None or post_event_obs is None:
                raise ValueError(
                    "split_method='observations' requires explicit test_obs "
                    "and post_event_obs."
                )
            if test_obs < 1 or post_event_obs < 1:
                raise ValueError(
                    f"test_obs and post_event_obs must be >= 1, got "
                    f"test_obs={test_obs}, post_event_obs={post_event_obs}."
                )
            n_test = int(test_obs)
            n_post_event = int(post_event_obs)
        train_df = pre_rows.iloc[:max(0, n_pre - n_test)].copy()
        test_df = pre_rows.iloc[max(0, n_pre - n_test):].copy()
        # post_rows is every row from the intervention on; the post-event
        # window is the first n_post_event of them.
        post_event_df = post_rows.iloc[:n_post_event].copy()

    full_predict_df = pd.concat([test_df, post_event_df]).copy()

    logger.info(
        "Splits (%s): train=%d obs, test=%d obs (%.1f%% of %d pre), "
        "post-event=%d obs (%.1f%% of %d post)",
        split_method, len(train_df),
        len(test_df), 100 * len(test_df) / n_pre if n_pre else float("nan"),
        n_pre,
        len(post_event_df),
        100 * len(post_event_df) / n_post if n_post else float("nan"),
        n_post,
    )

    # Guardrail against silently degenerate test windows, whatever their
    # cause -- unit confusion, short pre-event series, etc. (GH #29). An
    # EMPTY window is the most degenerate case of all: downstream code
    # substitutes training metrics for the missing test metrics, so silence
    # here would present train performance as test performance.
    if min_test_obs > 0 and len(test_df) == 0:
        warnings.warn(
            "Test window is EMPTY: no observations fall in the test period. "
            "Downstream test metrics will silently repeat training metrics. "
            "Check the split settings (window size vs the span of "
            "pre-intervention data), or set min_test_obs=0 to silence this "
            "warning.",
            UserWarning,
            stacklevel=2,
        )
    elif 0 < len(test_df) < min_test_obs:
        warnings.warn(
            f"Test window has only {len(test_df)} observations "
            f"(< min_test_obs={min_test_obs}). Test metrics such as MAPE are "
            "unstable on so few points and can mislead model selection and "
            "interval calibration. Check the split settings, or set "
            "min_test_obs=0 to silence this warning.",
            UserWarning,
            stacklevel=2,
        )

    return TimeSeriesSplits(
        train_df=train_df,
        test_df=test_df,
        post_event_df=post_event_df,
        full_predict_df=full_predict_df,
        intervention_date=intervention_date,
    )
