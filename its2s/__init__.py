# Description: Public API for its2s package.
# Usage: from its2s import run_single_its, run_batch, load_config, tune_model

__version__ = "0.1.0"

from .batch.runner import run_batch
from .compare import compare_models
from .cross_validation import time_series_cv
from .design import (Design, DesignError, DesignWarning, FrozenDesignError,
                     SealError, declare_design)
from .candidates_record import CandidatesRecord
from .check import CheckError, CheckWarning, check_selected_model
from .check_record import CheckRecord
from .design_record import DesignRecord
from .explore import explore
from .metrics.excess import calc_ate_summary
from .settings import load_config
from .pipeline import run_single_its
from .screening import ScreeningError, screen_models
from .tuning import tune_model, TuningResult

__all__ = [
    "__version__",
    "run_single_its",
    "run_batch",
    "load_config",
    "time_series_cv",
    "compare_models",
    "tune_model",
    "TuningResult",
    "calc_ate_summary",
    "Design",
    "declare_design",
    "DesignError",
    "DesignWarning",
    "FrozenDesignError",
    "SealError",
    "DesignRecord",
    "explore",
    "screen_models",
    "CandidatesRecord",
    "ScreeningError",
    "check_selected_model",
    "CheckRecord",
    "CheckError",
    "CheckWarning",
]
