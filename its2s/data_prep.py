# Description: Train/test/holdout splitting for ITS analysis.
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
    "percent": ("test_pct", "holdout_pct"),
    "days": ("test_days", "holdout_days"),
    "observations": ("test_obs", "holdout_obs"),
}


@dataclass
class TimeSeriesSplits:
    """Container for ITS train/test/holdout splits."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    holdout_df: pd.DataFrame
    full_predict_df: pd.DataFrame
    intervention_date: pd.Timestamp


def resolve_split_config(periods_cfg, split_method=None):
    """Resolve (split_method, split_kwargs) from a config "periods" section.

    Only the arguments belonging to the resolved method are read and passed
    on: prepare_splits raises on cross-method arguments (GH #28, #54). The
    ``split_method`` argument overrides the config key. ``periods_cfg`` is
    never mutated.

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
    method = (split_method if split_method is not None
              else periods_cfg.get("split_method", "percent"))
    if method == "days":
        split_kwargs = {
            "test_days": periods_cfg.get("test_days", 365),
            "holdout_days": periods_cfg.get("holdout_days", 365),
        }
    elif method == "observations":
        split_kwargs = {
            "test_obs": periods_cfg.get("test_obs"),
            "holdout_obs": periods_cfg.get("holdout_obs"),
        }
    else:
        # "percent" is the default; an unknown method raises in prepare_splits.
        split_kwargs = {
            "test_pct": periods_cfg.get("test_pct", 0.20),
            "holdout_pct": periods_cfg.get("holdout_pct", 1.0),
        }
    return method, split_kwargs


def prepare_splits(df, intervention_date, date_col="ds",
                   split_method="percent",
                   test_pct=None, holdout_pct=None,
                   test_days=None, holdout_days=None,
                   test_obs=None, holdout_obs=None,
                   min_test_obs=30):
    """Split a time series DataFrame into train, test, and holdout periods.

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
        "percent" (default): size the test/holdout windows as fractions of the
        pre-/post-intervention observation counts.
        "days": use explicit calendar-day windows `test_days`/`holdout_days`.
        "observations": use explicit observation-count windows
        `test_obs`/`holdout_obs`.
    test_pct : float
        Fraction of pre-intervention observations used as the test window.
        Defaults to 0.20. Only with ``split_method="percent"``.
    holdout_pct : float
        Fraction of post-intervention observations used as the holdout window.
        Defaults to 1.0. Only with ``split_method="percent"``.
    test_days : int
        Calendar days before the intervention used as the test window.
        Defaults to 365. Only with ``split_method="days"``.
    holdout_days : int
        Calendar days after the intervention used as the holdout window.
        Defaults to 365. Only with ``split_method="days"``.
    test_obs : int
        Number of observations before the intervention used as the test
        window. Required with ``split_method="observations"``; no default.
    holdout_obs : int
        Number of observations from the intervention onward used as the
        holdout window. Required with ``split_method="observations"``; no
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

    passed = {"test_pct": test_pct, "holdout_pct": holdout_pct,
              "test_days": test_days, "holdout_days": holdout_days,
              "test_obs": test_obs, "holdout_obs": holdout_obs}
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

    pre_df = df[df[date_col] < intervention_date]
    post_df = df[df[date_col] >= intervention_date]
    n_pre = len(pre_df)
    n_post = len(post_df)

    if split_method == "days":
        test_days = 365 if test_days is None else test_days
        holdout_days = 365 if holdout_days is None else holdout_days
        test_start = intervention_date - pd.Timedelta(days=test_days)
        holdout_end = intervention_date + pd.Timedelta(days=holdout_days)
        train_df = df[df[date_col] < test_start].copy()
        test_df = df[(df[date_col] >= test_start)
                     & (df[date_col] < intervention_date)].copy()
        holdout_df = df[(df[date_col] >= intervention_date)
                        & (df[date_col] <= holdout_end)].copy()
    else:
        if split_method == "percent":
            test_pct = 0.20 if test_pct is None else test_pct
            holdout_pct = 1.0 if holdout_pct is None else holdout_pct
            n_test = max(1, int(round(test_pct * n_pre)))
            n_holdout = max(1, int(round(holdout_pct * n_post)))
        else:
            if test_obs is None or holdout_obs is None:
                raise ValueError(
                    "split_method='observations' requires explicit test_obs "
                    "and holdout_obs."
                )
            if test_obs < 1 or holdout_obs < 1:
                raise ValueError(
                    f"test_obs and holdout_obs must be >= 1, got "
                    f"test_obs={test_obs}, holdout_obs={holdout_obs}."
                )
            n_test = int(test_obs)
            n_holdout = int(holdout_obs)
        train_df = pre_df.iloc[:max(0, n_pre - n_test)].copy()
        test_df = pre_df.iloc[max(0, n_pre - n_test):].copy()
        holdout_df = post_df.iloc[:n_holdout].copy()

    full_predict_df = pd.concat([test_df, holdout_df]).copy()

    logger.info(
        "Splits (%s): train=%d obs, test=%d obs (%.1f%% of %d pre), "
        "holdout=%d obs (%.1f%% of %d post)",
        split_method, len(train_df),
        len(test_df), 100 * len(test_df) / n_pre if n_pre else float("nan"),
        n_pre,
        len(holdout_df),
        100 * len(holdout_df) / n_post if n_post else float("nan"),
        n_post,
    )

    # Guardrail against silently degenerate test windows, whatever their
    # cause -- unit confusion, short pre-event series, etc. (GH #29).
    if 0 < len(test_df) < min_test_obs:
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
        holdout_df=holdout_df,
        full_predict_df=full_predict_df,
        intervention_date=intervention_date,
    )
