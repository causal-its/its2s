import logging
from pathlib import Path

import numpy as np
import pandas as pd

from its2s import run_single_its

# Run artifacts go here (not under its2s/outputs/, which is library code: plots.py, tables.py).
OUT_DIR = Path(__file__).resolve().parent / "trace_run_outputs"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(name)s — %(message)s",
)

rng = np.random.default_rng(42)
intervention_date = "2022-06-01"
config_overrides = {
    "periods": {"test_days": 90, "holdout_days": 90},
    "bootstrap": {"n_sim": 100},
}

dates = pd.date_range("2020-01-01", "2022-09-30", freq="D")
n = len(dates)
t = np.arange(n)
seasonal = 10 * np.sin(2 * np.pi * t / 365.25)
trend = 0.02 * t
noise = rng.normal(0, 2, n)
y = 100 + trend + seasonal + noise
intervention = pd.Timestamp(intervention_date)
y = y.astype(float)
y[dates >= intervention] += 5
df = pd.DataFrame({"ds": dates, "y": y})

result = run_single_its(
    df,
    intervention_date=intervention_date,
    model_name="arima",
    config_overrides=config_overrides,
    output_dir=OUT_DIR,
    seed=42,
)

print("done:", result.model_name, "train rows", len(result.fit_result.fitted_values))
print("wrote:", OUT_DIR.resolve())