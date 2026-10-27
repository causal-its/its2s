# Description: The fixed map of shipped model families (contract P11.04 section
#   5.7). Each FamilySpec holds a family's name, status, default parameters,
#   search space and class location. Importing this module imports none of
#   prophet, pmdarima or neuralprophet, so a declaration can be checked before
#   any heavy import or fit.
# Usage: from its2s.models.families import get_family, load_family_class
# Dependencies: none (stdlib only)

import importlib
from dataclasses import dataclass, field

# Search spaces: param_name -> (low, high, dtype, scale)
# These are the tuning ranges used by the Latin hypercube sampler. They are
# independent of a family's defaults, which are the fixed (non-tuning) values.
#   dtype : "int" or "float"
#   scale : "linear" or "log" (log produces a log-uniform distribution)
#
# Params prefixed "section__" are nested into {"section": {param: value}} in the
# model config dict before injection via config_overrides.
#
# NOTE on ARIMA: the R reference tunes fixed ARIMA orders (p,d,q,P,D,Q). The Python
# ARIMAModel uses pmdarima.auto_arima, which accepts upper bounds (max_p, max_d, etc.)
# and selects the best order within those bounds. Tuning these bounds controls the
# search space available to auto_arima rather than selecting a fixed order -- a
# methodological difference from the R implementation that should be noted in
# any downstream reporting.

_ARIMA_SPACE = {
    "max_p": (1, 5, "int", "linear"),
    "max_d": (0, 2, "int", "linear"),
    "max_q": (1, 5, "int", "linear"),
    "max_P": (0, 3, "int", "linear"),
    "max_D": (0, 1, "int", "linear"),
    "max_Q": (0, 3, "int", "linear"),
}

_NEURALPROPHET_SPACE = {
    # n_lags bounds are OBSERVATION counts (7..30 rows), not calendar days.
    "n_lags":          (7,     30,   "int",   "linear"),
    "epochs":          (50,    200,  "int",   "linear"),
    "learning_rate":   (0.001, 0.1,  "float", "log"),
    "batch_size":      (16,    128,  "int",   "linear"),
}

# Mirrors R phxgb: 3 Prophet params + 7 XGBoost params = 10 total (R had 11,
# including growth=linear and seasonality_yearly=TRUE which are fixed here).
_PROPHET_XGB_SPACE = {
    "prophet__changepoint_prior_scale": (0.01,  0.5,  "float", "log"),
    "prophet__seasonality_prior_scale": (0.1,   10.0, "float", "log"),
    "prophet__changepoint_range":       (0.4,   0.9,  "float", "linear"),
    "xgb__n_estimators":                (50,    500,  "int",   "linear"),
    "xgb__max_depth":                   (4,     20,   "int",   "linear"),
    "xgb__learning_rate":               (0.001, 0.3,  "float", "log"),
    "xgb__min_child_weight":            (1,     12,   "int",   "linear"),
    "xgb__subsample":                   (0.6,   1.0,  "float", "linear"),
    "xgb__colsample_bytree":            (0.5,   1.0,  "float", "linear"),
    "xgb__gamma":                       (0.001, 5.0,  "float", "log"),
}

# Dimensions that act only on a seasonal component; they are dropped from the
# space when no cycle is declared, since they have nothing to act on.
_SEASONAL_ONLY = {"prophet__seasonality_prior_scale", "max_P", "max_D", "max_Q"}


@dataclass(frozen=True)
class FamilySpec:
    """Everything about a model family that is known without importing it.

    defaults hold the family's fixed parameters only. Seasonal structure is
    never a parameter: it comes from the analysis' Seasonality (contract 5.4).
    """

    name: str
    status: str                       # "available" | "withheld"
    withheld_reason: str | None
    withheld_issue: int | None
    defaults: dict
    search_space: dict
    starter_n_trials: int             # matches the R reference grid sizes
    class_path: tuple                 # (module, class name), imported on demand


FAMILIES = {
    "prophet_xgb": FamilySpec(
        name="prophet_xgb",
        status="available",
        withheld_reason=None,
        withheld_issue=None,
        defaults={
            # seasonality_prior_scale and changepoint_range at Prophet's own
            # defaults, so an untuned fit is unchanged by their wiring (#76).
            "prophet": {"changepoint_prior_scale": 0.05,
                        "seasonality_prior_scale": 10.0,
                        "changepoint_range": 0.8},
            "xgb": {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05,
                    "subsample": 0.8, "colsample_bytree": 0.8,
                    "min_child_weight": 3, "gamma": 1.0},
        },
        search_space=_PROPHET_XGB_SPACE,
        starter_n_trials=100,
        class_path=(".prophet_xgb", "ProphetXGBHybridModel"),
    ),
    "arima": FamilySpec(
        name="arima",
        status="available",
        withheld_reason=None,
        withheld_issue=None,
        # max_P, max_D, max_Q at auto_arima's own defaults, so an untuned fit
        # is unchanged by their wiring.
        defaults={"max_p": 5, "max_d": 2, "max_q": 5,
                  "max_P": 2, "max_D": 1, "max_Q": 2,
                  "stepwise": True, "suppress_warnings": True},
        search_space=_ARIMA_SPACE,
        starter_n_trials=100,
        class_path=(".arima", "ARIMAModel"),
    ),
    "neuralprophet": FamilySpec(
        name="neuralprophet",
        status="withheld",
        withheld_reason=("its prediction routine consumes the observed outcomes "
                         "of the window it forecasts, so every score and every "
                         "excess it produces is contaminated"),
        withheld_issue=53,
        # n_lags is in OBSERVATIONS, not days: two weeks on daily data, 14
        # weeks on weekly (GH #59).
        defaults={"n_lags": 14, "learning_rate": 0.01, "epochs": 100,
                  "batch_size": 64},
        search_space=_NEURALPROPHET_SPACE,
        starter_n_trials=75,
        class_path=(".neuralprophet", "NeuralProphetModel"),
    ),
}


def get_family(name):
    """Return the FamilySpec for name; an unknown name raises, listing the three."""
    try:
        return FAMILIES[name]
    except KeyError:
        raise ValueError(
            f"Unknown model family {name!r}. Known families: {list(FAMILIES)}."
        ) from None


def load_family_class(name):
    """Import and return the class of family name.

    Raises ImportError naming the family when its library is missing.
    """
    module_path, class_name = get_family(name).class_path
    try:
        module = importlib.import_module(module_path, package=__package__)
    except ImportError as e:
        raise ImportError(
            f"Model '{name}' is not available because a required "
            f"dependency could not be imported: {e}. "
            f"Check that all dependencies for this model are installed."
        ) from e
    return getattr(module, class_name)


@dataclass(frozen=True)
class ResolvedSpace:
    """A family's search space after fixed= and bounds= are applied.

    sample : dims still to search, name -> (low, high, dtype, scale)
    fixed : name -> value, held at that value in every candidate
    flags : human-readable notes of every departure from the shipped space
    """

    sample: dict
    fixed: dict
    flags: list = field(default_factory=list)


def _check_int(name, value, what):
    if not float(value).is_integer():
        raise ValueError(f"{name} is an integer dimension; {what} {value!r} is not an integer.")


def resolve_space(spec, seasonality, fixed=None, bounds=None):
    """Apply fixed= and bounds= to spec's search space (contract 5.7).

    Raises on a name the family does not have, a name in both fixed and
    bounds, low >= high, a non-positive bound on a log-scaled dimension, or a
    non-integer value on an integer dimension. Wider-than-shipped bounds are
    allowed. Seasonal-only dimensions vanish when no cycle is declared.
    """
    fixed, bounds = dict(fixed or {}), dict(bounds or {})
    space = dict(spec.search_space)
    unknown = sorted((set(fixed) | set(bounds)) - set(space))
    if unknown:
        raise ValueError(
            f"Family {spec.name!r} has no dimension(s) {unknown}; its dimensions "
            f"are {sorted(space)}.")
    both = sorted(set(fixed) & set(bounds))
    if both:
        raise ValueError(f"{both} given in both fixed= and bounds=; give each once.")

    flags = []
    if not seasonality.cycles:
        for name in sorted(_SEASONAL_ONLY & set(space)):
            if name in fixed or name in bounds:
                raise ValueError(
                    f"{name} acts only on a seasonal component, and no cycle is "
                    "declared.")
            del space[name]
            flags.append(f"{name} dropped: no seasonal cycle declared")

    for name, value in fixed.items():
        if space[name][2] == "int":
            _check_int(name, value, "fixed value")
            value = int(value)
        flags.append(f"{name} fixed at {value!r}")
        fixed[name] = value
        del space[name]

    for name, (low, high) in bounds.items():
        _, _, dtype, scale = space[name]
        if low >= high:
            raise ValueError(f"bounds for {name}: low {low!r} must be below high {high!r}.")
        if scale == "log" and low <= 0:
            raise ValueError(f"bounds for {name}: log-scaled, so low must be positive.")
        if dtype == "int":
            _check_int(name, low, "bound")
            _check_int(name, high, "bound")
            low, high = int(low), int(high)
        flags.append(f"{name} bounds {spec.search_space[name][:2]} -> {(low, high)}")
        space[name] = (low, high, dtype, scale)

    return ResolvedSpace(sample=space, fixed=fixed, flags=flags)
