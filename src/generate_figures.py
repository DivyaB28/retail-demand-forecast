# src/generate_figures.py
#
# Generates Figures 1-3 for the Assignment 6 Model Optimization Summary Report
# 
# Required input files (already on disk from prior steps):
#   reports/model_comparison_summary.json   (from model_comparison.py)
#   reports/lstm_results.json               (from train_lstm.py)
#   reports/series_segments.csv             (from segment_series.py)
#
# Output:
#   reports/figures/fig1_h1_model_comparison_bar.png
#   reports/figures/fig2_lstm_lowvolume_histogram.png
#   reports/figures/fig3_sarima_vs_prophet_scatter.png
#
# Run this from the project root (same place you run model_comparison.py from):
#   python src/generate_figures.py

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

Path("reports/figures").mkdir(parents=True, exist_ok=True)

with open("reports/model_comparison_summary.json") as f:
    results = json.load(f)
with open("reports/lstm_results.json") as f:
    lstm_data = json.load(f)
segments = pd.read_csv("reports/series_segments.csv")

sarima_wmape = results["sarima_wmape_by_series"]
prophet_wmape = results["prophet_wmape_by_series"]
lstm_wmape = results["lstm_wmape_by_series"]
h1_table = results["h1_comparison_table"]
lv_ids = lstm_data["lv_ids"]

# Regenerate ss_ids exactly as model_comparison.py did (same seed -> same 40 ids,
# no model fitting involved, just the same reproducible sample draw)
SAMPLE_PER_SEGMENT = 40
ss_ids = segments.loc[segments["segment_ss_lh"], "id"].sample(
    min(SAMPLE_PER_SEGMENT, segments["segment_ss_lh"].sum()), random_state=42
).tolist()

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

# ---------- Figure 1: Grouped bar chart, mean WMAPE by model x segment ----------
models = ["SARIMA", "Prophet", "LightGBM", "LSTM"]
segs = ["high_volume_non_intermittent", "low_volume_intermittent"]
seg_labels = ["High-volume,\nnon-intermittent", "Low-volume,\nintermittent"]

x = np.arange(len(models))
width = 0.35
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, (seg, label) in enumerate(zip(segs, seg_labels)):
    vals = [h1_table[seg][m] for m in models]
    bars = ax.bar(x + (i - 0.5) * width, vals, width, label=label)
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Mean WMAPE (%)")
ax.set_title("Mean WMAPE by Model and Series Segment")
ax.legend(title="Segment", frameon=False)
fig.tight_layout()
fig.savefig("reports/figures/fig1_h1_model_comparison_bar.png", dpi=200)
plt.close(fig)

# ---------- Figure 2: Histogram of per-series LSTM WMAPE, low-volume segment ----------
lv_lstm_vals = [lstm_wmape[i] for i in lv_ids if i in lstm_wmape and np.isfinite(lstm_wmape[i])]

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(lv_lstm_vals, bins=20, edgecolor="white")
ax.axvline(100, color="firebrick", linestyle="--", linewidth=1.5, label="100% (zero-prediction line)")
ax.set_xlabel("Per-Series LSTM WMAPE (%)")
ax.set_ylabel("Number of Series")
ax.set_title("Distribution of LSTM WMAPE, Low-Volume/Intermittent Segment\n(n = %d series)" % len(lv_lstm_vals))
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("reports/figures/fig2_lstm_lowvolume_histogram.png", dpi=200)
plt.close(fig)

# ---------- Figure 3: Paired scatter, SARIMA vs Prophet, strong-seasonality sample ----------
paired = [
    (sarima_wmape[i], prophet_wmape[i]) for i in ss_ids
    if i in sarima_wmape and i in prophet_wmape
    and np.isfinite(sarima_wmape[i]) and np.isfinite(prophet_wmape[i])
]
sarima_vals, prophet_vals = zip(*paired)

lim = max(max(sarima_vals), max(prophet_vals)) * 1.05
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(sarima_vals, prophet_vals, alpha=0.7, edgecolor="white", s=60)
ax.plot([0, lim], [0, lim], color="gray", linestyle="--", linewidth=1, label="Equal performance")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("SARIMA WMAPE (%)")
ax.set_ylabel("Prophet WMAPE (%)")
ax.set_title("SARIMA vs. Prophet WMAPE\nStrong-Seasonality, Low-Holiday Series (n = %d)" % len(paired))
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("reports/figures/fig3_sarima_vs_prophet_scatter.png", dpi=200)
plt.close(fig)

print("Saved:")
print("  reports/figures/fig1_h1_model_comparison_bar.png")
print("  reports/figures/fig2_lstm_lowvolume_histogram.png")
print("  reports/figures/fig3_sarima_vs_prophet_scatter.png")