# src/segment_series.py
import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/m5_feature_engineered.parquet")

def weekly_autocorr(g):
    s = g.sort_values("date")["sales"]
    if s.std() == 0 or len(s) < 14:
        return 0.0
    return s.autocorr(lag=7)

stats = df.groupby("id").apply(lambda g: pd.Series({
    "total_volume": g["sales"].sum(),
    "mean_sales": g["sales"].mean(),
    "cv": g["sales"].std() / g["sales"].mean() if g["sales"].mean() > 0 else np.nan,
    "zero_pct": (g["sales"] == 0).mean(),
    "weekly_autocorr": weekly_autocorr(g),
    "event_day_count": g["has_event"].sum() if "has_event" in g else (g["event_name_1"] != "none").sum(),
})).reset_index()

vol_q75 = stats["total_volume"].quantile(0.75)
vol_q25 = stats["total_volume"].quantile(0.25)
zero_q25 = stats["zero_pct"].quantile(0.25)
zero_q75 = stats["zero_pct"].quantile(0.75)
event_median = stats["event_day_count"].median()
autocorr_q75 = stats["weekly_autocorr"].quantile(0.75)

stats["segment_hv_ni"] = (stats["total_volume"] >= vol_q75) & (stats["zero_pct"] <= zero_q25)
stats["segment_lv_i"] = (stats["total_volume"] <= vol_q25) & (stats["zero_pct"] >= zero_q75)
stats["segment_ss_lh"] = (stats["weekly_autocorr"] >= autocorr_q75) & (stats["event_day_count"] <= event_median)

print("High-volume, non-intermittent series (H1):", stats["segment_hv_ni"].sum())
print("Low-volume, intermittent series (H1):", stats["segment_lv_i"].sum())
print("Strong-seasonality, low-holiday series (H2):", stats["segment_ss_lh"].sum())

stats.to_csv("reports/series_segments.csv", index=False)
print("\nSaved reports/series_segments.csv")
print(stats[["id","total_volume","zero_pct","weekly_autocorr","segment_hv_ni","segment_lv_i","segment_ss_lh"]].head(10))