# Description: Output generation (plots and tables).

from .diagnostic_plots import (
    plot_residual_acf,
    plot_residual_diagnostics,
    plot_residual_pacf,
    plot_residual_qq,
    plot_residuals_over_time,
)
from .plots import plot_counterfactual
from .tables import (
    save_ate_summary,
    save_diagnostics_table,
    save_excess_table,
    save_metrics_table,
)

__all__ = [
    "plot_counterfactual",
    "plot_residual_acf",
    "plot_residual_pacf",
    "plot_residuals_over_time",
    "plot_residual_qq",
    "plot_residual_diagnostics",
    "save_excess_table",
    "save_metrics_table",
    "save_ate_summary",
    "save_diagnostics_table",
]
