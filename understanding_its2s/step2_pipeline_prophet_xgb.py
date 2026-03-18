# =============================================================================
# step2_pipeline_prophet_xgb.py
# Deep-dive into run_single_its() using the Prophet+XGB model.
#
# What this script exposes, step by step:
#   2a. Rebuild the same simulated dataset from step 1
#   2b. Fit the ProphetXGBHybridModel manually — inspect FitResult
#   2c. Understand the two-stage fit: Prophet first, then XGB on residuals
#   2d. Run the full pipeline via run_single_its()
#   2e. Inspect PipelineResult: metrics, excess table, ATE
#   2f. Recreate the counterfactual plot with interactive annotations
# =============================================================================

import logging
import sys
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")  # suppress Prophet/XGB verbosity

# Output directory — mirrors step1_data_splitting.py
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import path_project  # noqa: E402
out_dir = path_project / "understanding_its2s"
out_dir.mkdir(parents=True, exist_ok=True)

# Turn on INFO so you can watch the pipeline steps in the terminal
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# =============================================================================
# 2a — Simulate the same dataset as step 1
# =============================================================================
rng = np.random.default_rng(42)
dates = pd.date_range("2018-01-01", "2022-12-31", freq="D")
trend = np.linspace(0, 5, len(dates))
seasonality = 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
noise = rng.normal(0, 2, len(dates))
y = 50 + trend + seasonality + noise

INTERVENTION = "2022-01-01"

# Simulate a real intervention effect in the holdout period
# (+8 units/day starting from intervention) so the excess will be detectable
intervention_idx = np.searchsorted(dates, pd.Timestamp(INTERVENTION))
y[intervention_idx:] += 8

df = pd.DataFrame({"ds": dates, "y": y})

print("=" * 60)
print("2a. Simulated dataset  (with +8/day intervention effect)")
print("=" * 60)
print(df.tail())

# =============================================================================
# 2b — Manually fit ProphetXGBHybridModel on the training split
#       (replicates exactly what run_single_its does internally)
# =============================================================================
from its2s.data_prep import prepare_splits
from its2s.models.prophet_xgb import ProphetXGBHybridModel
from its2s.settings import get_model_config, load_config

config = load_config()  # loads params.yaml defaults
splits = prepare_splits(df, INTERVENTION)

model_params = get_model_config(config, "prophet_xgb")
model = ProphetXGBHybridModel(params=model_params)

print("\n" + "=" * 60)
print("2b. Fitting ProphetXGBHybridModel on training data ...")
print(f"    Training rows : {len(splits.train_df)}")
print(f"    Training range: {splits.train_df['ds'].min().date()} → {splits.train_df['ds'].max().date()}")
print("=" * 60)

fit_result = model.fit(splits.train_df, target_col="y", date_col="ds")

print("\nFitResult fields:")
print(f"  fitted_values  shape = {fit_result.fitted_values.shape}")
print(f"  residuals      shape = {fit_result.residuals.shape}")
print(f"  residuals  mean={fit_result.residuals.mean():.4f}  std={fit_result.residuals.std():.4f}")
print(f"  model_object keys: {list(fit_result.model_object.keys())}")

# =============================================================================
# 2c — Understand the two-stage fit
#       Prophet captures trend + seasonality → XGB fits the residuals
# =============================================================================
print("\n" + "=" * 60)
print("2c. Inside the hybrid: Prophet components vs XGB residuals")
print("=" * 60)

prophet_model = fit_result.model_object["prophet"]
xgb_model = fit_result.model_object["xgb"]

# Reconstruct Prophet in-sample predictions
prophet_input = splits.train_df[["ds"]].copy()
prophet_input["ds"] = pd.to_datetime(prophet_input["ds"])
prophet_pred = prophet_model.predict(prophet_input)

prophet_yhat = prophet_pred["yhat"].values
raw_y = splits.train_df["y"].values
stage1_residuals = raw_y - prophet_yhat     # what XGB was trained to predict
final_residuals = fit_result.residuals      # what remained after XGB too

print(f"  Stage-1 residuals (y - Prophet)  std = {stage1_residuals.std():.4f}")
print(f"  Stage-2 residuals (y - Prophet - XGB)  std = {final_residuals.std():.4f}")
print("  → XGB reduces residual variance from stage 1 to stage 2")

# XGB feature importances
xgb_feat_names = ["day_of_week", "day_of_year", "month", "week_of_year"]
importances = pd.Series(xgb_model.feature_importances_, index=xgb_feat_names).sort_values(ascending=False)
print("\n  XGB feature importances:")
print(importances.to_string())

# Quick residual plot: stage 1 vs stage 2
fig, axes = plt.subplots(1, 2, figsize=(13, 3.5))
for ax, resid, title, color in [
    (axes[0], stage1_residuals, "Stage-1 residuals (y − Prophet)", "#4C72B0"),
    (axes[1], final_residuals,  "Stage-2 residuals (y − Prophet − XGB)", "#DD8452"),
]:
    ax.plot(splits.train_df["ds"], resid, linewidth=0.6, color=color, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Residual")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.suptitle("Prophet+XGB: residuals before and after XGB stage", fontsize=11)
plt.tight_layout()
plt.savefig(out_dir / "step2_residuals.png", dpi=150)
plt.show()
print(f"\nPlot saved -> {out_dir / 'step2_residuals.png'}")

# =============================================================================
# 2d — Run the full pipeline via run_single_its()
# =============================================================================
print("\n" + "=" * 60)
print("2d. Running full pipeline via run_single_its() ...")
print("    (MBB bootstrap n_sim=100 for speed; set to 1000 for real use)")
print("=" * 60)

from its2s import run_single_its

result = run_single_its(
    df=df,
    intervention_date=INTERVENTION,
    model_name="prophet_xgb",
    config_overrides={"bootstrap": {"n_sim": 100}},  # fast for exploration
    output_dir=out_dir,
    seed=42,
)

print("\nPipelineResult fields:")
print(f"  model_name       : {result.model_name}")
print(f"  fit_result       : FitResult with {len(result.fit_result.fitted_values)} fitted values")
print(f"  bootstrap_result : BootstrapCIResult  pred_matrix shape = {result.bootstrap_result.pred_matrix.shape}")
print(f"  metrics_train    : {result.metrics_train}")
print(f"  metrics_test     : {result.metrics_test}")

# =============================================================================
# 2e — Inspect metrics and excess table
# =============================================================================
print("\n" + "=" * 60)
print("2e. Metrics — train vs test")
print("=" * 60)

metrics_df = pd.DataFrame({
    "RMSE":  [result.metrics_train.rmse,  result.metrics_test.rmse],
    "MAE":   [result.metrics_train.mae,   result.metrics_test.mae],
    "MAPE":  [result.metrics_train.mape,  result.metrics_test.mape],
    "SMAPE": [result.metrics_train.smape, result.metrics_test.smape],
    "R2":    [result.metrics_train.r2,    result.metrics_test.r2],
}, index=["Train", "Test"])
print(metrics_df.round(3).to_string())

print("\n" + "=" * 60)
print("2e. Excess table — period-level summary")
print("=" * 60)
print(result.excess_table.period_excess.to_string(index=False))

print("\n" + "=" * 60)
print("2e. Daily excess — first 10 holdout days")
print("=" * 60)
print(result.excess_table.daily_excess.head(10).to_string(index=False))

# ATE summary
from its2s.metrics.excess import calc_ate_summary
ate = calc_ate_summary(result.excess_table.daily_excess)
print("\n" + "=" * 60)
print("2e. Average Treatment Effect (ATE) summary")
print("=" * 60)
print(ate.to_string(index=False))
print("\n  → 'Total ATE'      = sum of daily excess over full holdout")
print("     'Mean Daily ATE' = average excess per day")
print("     The simulated effect was +8/day — check how close the estimate is.")

# =============================================================================
# 2f — Reproduce the counterfactual plot manually
#       (same data as the saved PNG, but annotated for learning)
# =============================================================================
print("\n" + "=" * 60)
print("2f. Counterfactual plot (annotated)")
print("=" * 60)

br = result.bootstrap_result
pred_dates = pd.to_datetime(br.dates)
intervention_ts = pd.Timestamp(INTERVENTION)

fig, ax = plt.subplots(figsize=(14, 5))

# Observed
for part in [splits.train_df, splits.test_df, splits.holdout_df]:
    ax.plot(part["ds"], part["y"], color="#333333", linewidth=0.6, alpha=0.7)
ax.plot([], [], color="#333333", linewidth=0.6, alpha=0.7, label="Observed")

# Counterfactual prediction + CI ribbon
ax.plot(pred_dates, br.predicted, color="#B2182B", linewidth=1.4,
        label="Counterfactual (no-intervention)")
ax.fill_between(pred_dates, br.conf_lo, br.conf_hi,
                color="#B2182B", alpha=0.15, label="95% CI (MBB)")

# Holdout shading
ax.axvspan(intervention_ts, splits.holdout_df["ds"].max(),
           color="#FEE08B", alpha=0.2, label="Holdout (post-intervention)")

# Intervention line
ax.axvline(intervention_ts, color="#4DAF4A", linestyle="--", linewidth=1.3,
           label="Intervention date")

# Annotate the gap on the last holdout day to make excess visible
last_date = pred_dates[pred_dates >= intervention_ts][-1]
last_obs = splits.holdout_df[splits.holdout_df["ds"] == last_date]["y"].values
last_pred = br.predicted[pred_dates == last_date]
if len(last_obs) and len(last_pred):
    ax.annotate(
        f"Excess ≈ {float(last_obs[0] - last_pred[0]):.1f}",
        xy=(last_date, float(last_pred[0])),
        xytext=(last_date - pd.Timedelta(days=60), float(last_pred[0]) + 5),
        arrowprops=dict(arrowstyle="->", color="black"),
        fontsize=9,
    )

ax.set_xlabel("Date")
ax.set_ylabel("y (daily outcome)")
ax.set_title(
    f"Prophet+XGB counterfactual  |  Test RMSE: {result.metrics_test.rmse:.2f}"
    f"  |  Test MAPE: {result.metrics_test.mape:.1f}%",
    fontsize=10,
)
ax.legend(loc="upper left", fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig(out_dir / "step2_counterfactual.png", dpi=150)
plt.show()
print(f"Plot saved -> {out_dir / 'step2_counterfactual.png'}")

print("\n" + "=" * 60)
print("Key takeaways from step 2")
print("=" * 60)
print("""
  1. ProphetXGBHybridModel.fit() runs TWO models in sequence:
       - Prophet fits trend + seasonality on (ds, y)
       - XGB is trained on (y − Prophet_yhat) using time features

  2. FitResult stores fitted_values and residuals — these residuals
     are the raw material for the Moving Block Bootstrap (step 3).

  3. run_single_its() orchestrates 7 steps in order:
       load_config → prepare_splits → fit → bootstrap → metrics → excess → save

  4. The PipelineResult dataclass bundles everything:
       fit_result, bootstrap_result, metrics_train, metrics_test,
       excess_table, config

  5. Excess = observed − counterfactual_predicted.
     With a true effect of +8/day, check that total_excess ≈ 8 × 365.
""")
