# =============================================================================
# understanding_its2s.py
# Interactive walkthrough of the its2s package — run section by section.
# =============================================================================

# =============================================================================
# STEP 1 — Data splitting (data_prep.py)
# Goal: understand how prepare_splits() carves a time series into
#       train / test / holdout windows around an intervention date.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from its2s.data_prep import prepare_splits
from paths import path_project

out_dir = path_project / "understanding_its2s"
out_dir.mkdir(parents=True, exist_ok=True)

# --- 1a. Simulate 5 years of daily data ---
# Seasonal sine wave (annual cycle) + small noise, mimicking e.g. daily counts
rng = np.random.default_rng(42)
dates = pd.date_range("2018-01-01", "2022-12-31", freq="D")
trend = np.linspace(0, 5, len(dates))
seasonality = 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
noise = rng.normal(0, 2, len(dates))
y = 50 + trend + seasonality + noise

df = pd.DataFrame({"ds": dates, "y": y})

print("=" * 60)
print("Simulated dataset")
print("=" * 60)
print(df.head())
print(f"\nShape : {df.shape}")
print(f"Date range: {df['ds'].min().date()} → {df['ds'].max().date()}")
print(f"y  mean={df['y'].mean():.2f}  std={df['y'].std():.2f}  "
      f"min={df['y'].min():.2f}  max={df['y'].max():.2f}")

# --- 1b. Call prepare_splits ---
# intervention_date: the date the 'treatment' happened
# test_days:    how many days *before* intervention become the held-out test window
# holdout_days: how many days *after*  intervention to track outcomes
INTERVENTION = "2022-01-01"

splits = prepare_splits(
    df,
    intervention_date=INTERVENTION,
    date_col="ds",
    test_days=365,
    holdout_days=365,
)

print("\n" + "=" * 60)
print("Split sizes")
print("=" * 60)
for name, part in [
    ("train",         splits.train_df),
    ("test",          splits.test_df),
    ("holdout",       splits.holdout_df),
    ("full_predict",  splits.full_predict_df),
]:
    if part.empty:
        date_range = "empty"
    else:
        date_range = f"{part['ds'].min().date()} → {part['ds'].max().date()}"
    print(f"  {name:<15}  {len(part):>4} rows   {date_range}")

# --- 1c. What is full_predict_df? ---
# It is test + holdout combined: the window the model will forecast over.
# The model never sees this data during fitting — it's purely for evaluation.
print("\n" + "=" * 60)
print("What is full_predict_df?")
print("=" * 60)
print("  full_predict = test + holdout  (used for bootstrapped forecasting)")
expected_len = len(splits.test_df) + len(splits.holdout_df)
print(f"  len(test) + len(holdout) = {len(splits.test_df)} + {len(splits.holdout_df)} = {expected_len}")
print(f"  len(full_predict_df)     = {len(splits.full_predict_df)}")
assert len(splits.full_predict_df) == expected_len, "Mismatch — check for off-by-one?"

# --- 1d. Visualise the splits ---
fig, ax = plt.subplots(figsize=(14, 4))

colors = {
    "train":    ("#4C72B0", "Train  (model fits here)"),
    "test":     ("#DD8452", "Test   (pre-intervention validation)"),
    "holdout":  ("#55A868", "Holdout (post-intervention outcomes)"),
}

for split_name, (color, label) in colors.items():
    part = getattr(splits, f"{split_name}_df")
    ax.plot(part["ds"], part["y"], color=color, linewidth=0.8, label=label)

ax.axvline(splits.intervention_date, color="red", linestyle="--", linewidth=1.5,
           label=f"Intervention  ({INTERVENTION})")
ax.axvline(splits.intervention_date - pd.Timedelta(days=365),
           color="gray", linestyle=":", linewidth=1, label="Test window start")

ax.set_title("ITS splits on simulated data", fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("y (daily outcome)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(out_dir / "step1_splits.png", dpi=150)
plt.show()
print(f"\nPlot saved -> {out_dir / 'step1_splits.png'}")
