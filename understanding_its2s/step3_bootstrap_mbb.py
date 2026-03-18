# =============================================================================
# step3_bootstrap_mbb.py
# Deep-dive into the Moving Block Bootstrap (MBB) CI machinery.
# Self-contained: re-fits Prophet+XGB on the same simulated dataset.
#
# Sections:
#   3a. Fit model and inspect the raw residuals that MBB resamples
#   3b. Single simulation in slow motion — block resampling made visible
#   3c. Build the pred_matrix manually — watch the fan accumulate
#   3d. CI methods compared — quantile vs symmetric_sd
#   3e. Block length sensitivity — how block_length affects CI width
# =============================================================================

import logging
import sys
import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paths import path_project  # noqa: E402

out_dir = path_project / "understanding_its2s"
out_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Setup — same simulated dataset and model as step 2
# =============================================================================
from its2s.data_prep import prepare_splits
from its2s.models.prophet_xgb import ProphetXGBHybridModel
from its2s.settings import get_model_config, load_config

rng = np.random.default_rng(42)
dates = pd.date_range("2018-01-01", "2022-12-31", freq="D")
trend = np.linspace(0, 5, len(dates))
seasonality = 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)
noise = rng.normal(0, 2, len(dates))
y = 50 + trend + seasonality + noise
intervention_idx = np.searchsorted(dates, pd.Timestamp("2022-01-01"))
y[intervention_idx:] += 8
df = pd.DataFrame({"ds": dates, "y": y})

INTERVENTION = "2022-01-01"
config = load_config()
splits = prepare_splits(df, INTERVENTION)

print("Fitting Prophet+XGB on training data ...")
model = ProphetXGBHybridModel(params=get_model_config(config, "prophet_xgb"))
fit_result = model.fit(splits.train_df, target_col="y", date_col="ds")
print("Fit complete.\n")

# =============================================================================
# 3a — Inspect the raw residuals that MBB will resample
# =============================================================================
print("=" * 60)
print("3a. Raw residuals from the fitted model")
print("=" * 60)

residuals = fit_result.residuals
fitted_values = fit_result.fitted_values
train_dates = splits.train_df["ds"]

print(f"  n residuals : {len(residuals)}")
print(f"  mean        : {residuals.mean():.4f}  (should be near 0)")
print(f"  std         : {residuals.std():.4f}")
print(f"  min / max   : {residuals.min():.3f} / {residuals.max():.3f}")

# Autocorrelation at lag 1 and lag 7 — key motivation for block resampling
lag1_acf  = np.corrcoef(residuals[1:],  residuals[:-1])[0, 1]
lag7_acf  = np.corrcoef(residuals[7:],  residuals[:-7])[0, 1]
lag14_acf = np.corrcoef(residuals[14:], residuals[:-14])[0, 1]
print(f"\n  Autocorrelation of residuals:")
print(f"    lag-1  : {lag1_acf:.3f}")
print(f"    lag-7  : {lag7_acf:.3f}")
print(f"    lag-14 : {lag14_acf:.3f}")
print("  → Non-zero autocorrelation is WHY we use blocks, not iid resampling.")

fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)
axes[0].plot(train_dates, splits.train_df["y"].values, color="#333333",
             linewidth=0.6, alpha=0.8, label="Observed y")
axes[0].plot(train_dates, fitted_values, color="#2166AC",
             linewidth=0.9, alpha=0.8, label="Fitted values")
axes[0].set_ylabel("y")
axes[0].legend(fontsize=8)
axes[0].set_title("3a. Fitted values vs observed (training period)", fontsize=10)

axes[1].plot(train_dates, residuals, color="#B2182B",
             linewidth=0.6, alpha=0.8)
axes[1].axhline(0, color="black", linewidth=0.8, linestyle="--")
axes[1].set_ylabel("Residual")
axes[1].set_title(f"Residuals  (std={residuals.std():.3f},  lag-1 ACF={lag1_acf:.3f})",
                  fontsize=10)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig(out_dir / "step3a_residuals.png", dpi=150)
plt.show()
print(f"Plot saved -> {out_dir / 'step3a_residuals.png'}\n")

# =============================================================================
# 3b — Single simulation in slow motion: block resampling made visible
# =============================================================================
print("=" * 60)
print("3b. Single MBB simulation — block resampling in slow motion")
print("=" * 60)

from its2s.bootstrap.mbb import _resample_blocks

BLOCK_LENGTH = 14
sim_rng = np.random.default_rng(0)

resampled = _resample_blocks(residuals, BLOCK_LENGTH, sim_rng)

# Identify which block was placed at each position (for illustration)
# Re-run with a fresh rng to recover the start indices
viz_rng = np.random.default_rng(0)
import math
n = len(residuals)
n_blocks = math.ceil(n / BLOCK_LENGTH)
max_start = n - BLOCK_LENGTH
starts = viz_rng.integers(0, max_start + 1, size=n_blocks)

print(f"  block_length : {BLOCK_LENGTH}")
print(f"  n_blocks     : {n_blocks}  (to cover {n} residuals)")
print(f"  First 5 block start indices: {starts[:5].tolist()}")
print(f"  → each block copies a {BLOCK_LENGTH}-day contiguous chunk of residuals")

perturbed_y = fitted_values + resampled
perturbed_std = perturbed_y.std()
original_std  = splits.train_df["y"].values.std()
print(f"\n  Original training y std    : {original_std:.3f}")
print(f"  Perturbed training y std   : {perturbed_std:.3f}")
print("  → perturbed series is a plausible alternative history")

fig, axes = plt.subplots(2, 1, figsize=(13, 5), sharex=True)

# Shade the blocks on the residual panel to show what was picked
colors_cycle = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
for b_idx, start in enumerate(starts[:8]):  # show first 8 blocks only
    x_start = train_dates.iloc[start]
    x_end   = train_dates.iloc[min(start + BLOCK_LENGTH - 1, n - 1)]
    axes[0].axvspan(x_start, x_end,
                    color=colors_cycle[b_idx % len(colors_cycle)],
                    alpha=0.25)
axes[0].plot(train_dates, residuals,  color="#333333", linewidth=0.7,
             label="Original residuals")
axes[0].plot(train_dates, resampled,  color="#B2182B", linewidth=0.7,
             alpha=0.7, label="Resampled residuals")
axes[0].axhline(0, color="black", linewidth=0.6, linestyle="--")
axes[0].set_ylabel("Residual")
axes[0].legend(fontsize=8)
axes[0].set_title(f"3b. Original vs resampled residuals  (block_length={BLOCK_LENGTH},"
                  f" shaded = first 8 blocks picked)", fontsize=10)

axes[1].plot(train_dates, splits.train_df["y"].values, color="#333333",
             linewidth=0.6, label="Original training y")
axes[1].plot(train_dates, perturbed_y, color="#2166AC",
             linewidth=0.6, alpha=0.7, label="Perturbed training y")
axes[1].set_ylabel("y")
axes[1].legend(fontsize=8)
axes[1].set_title("Perturbed training series (fitted + resampled residuals)", fontsize=10)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig(out_dir / "step3b_block_resampling.png", dpi=150)
plt.show()
print(f"Plot saved -> {out_dir / 'step3b_block_resampling.png'}\n")

# =============================================================================
# 3c — Build the pred_matrix manually, watch the fan accumulate
# =============================================================================
print("=" * 60)
print("3c. Building pred_matrix manually — the fan of counterfactual worlds")
print("=" * 60)

from its2s.bootstrap.mbb import _single_mbb_sim

N_SIM = 100
base_rng = np.random.default_rng(42)
sim_seeds = base_rng.integers(0, 2**31, size=N_SIM)

n_dates = len(splits.full_predict_df)
pred_matrix = np.full((n_dates, N_SIM), np.nan)

print(f"  Running {N_SIM} simulations ...")
for i in range(N_SIM):
    preds = _single_mbb_sim(
        i, model, splits.train_df, splits.full_predict_df,
        fitted_values, residuals,
        BLOCK_LENGTH, "y", "ds", None, int(sim_seeds[i]),
    )
    pred_matrix[:, i] = preds

print(f"  pred_matrix shape : {pred_matrix.shape}  (n_dates × n_sims)")
print(f"  Each column is one bootstrapped counterfactual trajectory.")
print(f"  Spread across simulations at day 0  : std = {np.nanstd(pred_matrix[0, :]):.3f}")
print(f"  Spread across simulations at day 180: std = {np.nanstd(pred_matrix[180, :]):.3f}")
print("  → uncertainty typically grows further from the training period")

pred_dates = pd.to_datetime(splits.full_predict_df["ds"])
intervention_ts = pd.Timestamp(INTERVENTION)

fig, ax = plt.subplots(figsize=(13, 5))

# All 100 trajectories as a faint fan
for i in range(N_SIM):
    ax.plot(pred_dates, pred_matrix[:, i], color="#B2182B",
            linewidth=0.3, alpha=0.12)

# Quantile CIs derived from the matrix
conf_lo = np.nanpercentile(pred_matrix, 2.5,  axis=1)
conf_hi = np.nanpercentile(pred_matrix, 97.5, axis=1)
point_pred = model.predict(splits.full_predict_df, target_col="y", date_col="ds")

ax.fill_between(pred_dates, conf_lo, conf_hi,
                color="#B2182B", alpha=0.25, label="95% CI (quantile)")
ax.plot(pred_dates, point_pred.predicted, color="#B2182B",
        linewidth=1.4, label="Point prediction")

# Observed
for part in [splits.train_df, splits.test_df, splits.holdout_df]:
    ax.plot(part["ds"], part["y"], color="#333333", linewidth=0.6, alpha=0.7)
ax.plot([], [], color="#333333", linewidth=0.6, label="Observed")

ax.axvline(intervention_ts, color="#4DAF4A", linestyle="--",
           linewidth=1.2, label="Intervention")
ax.set_xlabel("Date")
ax.set_ylabel("y")
ax.set_title(f"3c. The bootstrap fan — {N_SIM} counterfactual trajectories"
             f"  (each faint red line = one simulation)", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

plt.tight_layout()
plt.savefig(out_dir / "step3c_bootstrap_fan.png", dpi=150)
plt.show()
print(f"Plot saved -> {out_dir / 'step3c_bootstrap_fan.png'}\n")

# =============================================================================
# 3d — CI methods compared: quantile vs symmetric_sd
# =============================================================================
print("=" * 60)
print("3d. CI method comparison: quantile vs symmetric_sd")
print("=" * 60)

from its2s.bootstrap.base import BaseBootstrap

ci_lo_q,   ci_hi_q   = BaseBootstrap.calculate_ci(pred_matrix, point_pred.predicted,
                                                    method="quantile",     level=0.95)
ci_lo_sd,  ci_hi_sd  = BaseBootstrap.calculate_ci(pred_matrix, point_pred.predicted,
                                                    method="symmetric_sd", level=0.95)

width_q  = (ci_hi_q  - ci_lo_q).mean()
width_sd = (ci_hi_sd - ci_lo_sd).mean()
print(f"  Mean CI width — quantile    : {width_q:.3f}")
print(f"  Mean CI width — symmetric_sd: {width_sd:.3f}")
print("  → if the bootstrap distribution is symmetric, both are similar.")
print("    if skewed, quantile CIs will be asymmetric — more honest.")

# Check skewness of bootstrap distribution at a holdout date
holdout_col_idx = (pred_dates >= intervention_ts).argmax()
boot_dist = pred_matrix[holdout_col_idx, :]
skew = float(pd.Series(boot_dist).skew())
print(f"\n  Bootstrap distribution skewness at first holdout date: {skew:.3f}")

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(pred_dates, point_pred.predicted, color="#333333",
        linewidth=1.2, label="Point prediction", zorder=3)
ax.fill_between(pred_dates, ci_lo_q,  ci_hi_q,
                color="#4C72B0", alpha=0.30, label=f"Quantile CI (width={width_q:.2f})")
ax.fill_between(pred_dates, ci_lo_sd, ci_hi_sd,
                color="#DD8452", alpha=0.30, label=f"Symmetric SD CI (width={width_sd:.2f})")
ax.axvline(intervention_ts, color="#4DAF4A", linestyle="--", linewidth=1.2,
           label="Intervention")
ax.set_xlabel("Date")
ax.set_ylabel("y")
ax.set_title("3d. Quantile CI vs symmetric SD CI", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig(out_dir / "step3d_ci_methods.png", dpi=150)
plt.show()
print(f"Plot saved -> {out_dir / 'step3d_ci_methods.png'}\n")

# =============================================================================
# 3e — Block length sensitivity
# =============================================================================
print("=" * 60)
print("3e. Block length sensitivity")
print("=" * 60)

BLOCK_LENGTHS = [3, 7, 14, 28]
bl_colors = ["#4C72B0", "#55A868", "#B2182B", "#8172B3"]
ci_results = {}

for bl in BLOCK_LENGTHS:
    print(f"  Running n_sim={N_SIM} with block_length={bl} ...")
    bl_rng = np.random.default_rng(42)
    bl_seeds = bl_rng.integers(0, 2**31, size=N_SIM)
    bl_matrix = np.full((n_dates, N_SIM), np.nan)

    for i in range(N_SIM):
        preds = _single_mbb_sim(
            i, model, splits.train_df, splits.full_predict_df,
            fitted_values, residuals,
            bl, "y", "ds", None, int(bl_seeds[i]),
        )
        bl_matrix[:, i] = preds

    lo, hi = BaseBootstrap.calculate_ci(bl_matrix, point_pred.predicted,
                                        method="quantile", level=0.95)
    ci_results[bl] = (lo, hi, (hi - lo).mean())
    print(f"    mean CI width = {ci_results[bl][2]:.3f}")

print("\n  Summary:")
print(f"  {'block_length':>14}  {'mean CI width':>14}")
for bl, (_, _, w) in ci_results.items():
    print(f"  {bl:>14}  {w:>14.3f}")
print("\n  → Too short: residual autocorrelation not preserved → CIs too narrow.")
print("    Too long:  few independent blocks → noisy quantile estimates → CIs unstable.")

fig, ax = plt.subplots(figsize=(13, 5))

for part in [splits.train_df, splits.test_df, splits.holdout_df]:
    ax.plot(part["ds"], part["y"], color="#333333", linewidth=0.5, alpha=0.6)
ax.plot([], [], color="#333333", linewidth=0.6, label="Observed")

ax.plot(pred_dates, point_pred.predicted, color="#333333",
        linewidth=1.2, linestyle="--", label="Point prediction", zorder=4)

for bl, color in zip(BLOCK_LENGTHS, bl_colors):
    lo, hi, w = ci_results[bl]
    ax.fill_between(pred_dates, lo, hi, color=color,
                    alpha=0.20, label=f"block_length={bl}  (mean width={w:.2f})")

ax.axvline(intervention_ts, color="black", linestyle="--",
           linewidth=1.0, label="Intervention")
ax.set_xlabel("Date")
ax.set_ylabel("y")
ax.set_title("3e. MBB CI width vs block length  (all at 95%, n_sim=100)", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.tight_layout()
plt.savefig(out_dir / "step3e_block_length_sensitivity.png", dpi=150)
plt.show()
print(f"Plot saved -> {out_dir / 'step3e_block_length_sensitivity.png'}\n")

print("=" * 60)
print("Key takeaways from step 3")
print("=" * 60)
print("""
  1. MBB resamples residuals in contiguous blocks (not iid) to preserve
     the autocorrelation structure of the time series.

  2. pred_matrix is (n_dates × n_sims) — each column is a full
     counterfactual trajectory from one bootstrap world.

  3. CIs are the 2.5th / 97.5th percentile of pred_matrix across
     simulations at each date. No parametric assumptions needed.

  4. "quantile" CI is preferable to "symmetric_sd" when the bootstrap
     distribution is skewed — it adapts to asymmetry automatically.

  5. block_length is a tuning parameter:
       - Too short → CIs underestimate uncertainty (autocorrelation lost)
       - Too long  → CIs become noisy (too few independent blocks)
       - Default of 14 (2 weeks) is a reasonable starting point for
         daily data with weekly seasonality.
""")
