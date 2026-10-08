# src/model_comparison.py
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
# import shap
from scipy.stats import wilcoxon
from pmdarima import auto_arima
from prophet import Prophet

df = pd.read_parquet("data/processed/m5_feature_engineered.parquet")
df = df.sort_values(["id", "date"]).reset_index(drop=True)
segments = pd.read_csv("reports/series_segments.csv")

df_model = df.dropna(subset=["sales_lag_7","sales_lag_28","rolling_mean_7","rolling_std_7"]).copy()
cutoff_date = df_model["date"].max() - pd.Timedelta(days=28)
train_full = df_model[df_model["date"] <= cutoff_date]
test_full = df_model[df_model["date"] > cutoff_date]

def wmape(actual, pred):
    return np.sum(np.abs(actual - pred)) / np.sum(np.abs(actual)) * 100

feature_cols = ["sell_price","wday","month","has_event","is_weekend","snap_CA","snap_TX",
                 "sales_lag_7","sales_lag_28","rolling_mean_7","rolling_std_7",
                 "cat_id_HOBBIES","dept_id_HOBBIES_1","state_id_TX","state_id_WI"]

results = {"per_series": {}}


print("=== Tuning HistGradientBoosting ===")
param_dist = {
    "max_leaf_nodes": [15, 31, 63],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_iter": [100, 300, 500],
    "max_depth": [None, 5, 10],
}
tscv = TimeSeriesSplit(n_splits=3)
search = RandomizedSearchCV(
    HistGradientBoostingRegressor(random_state=42), param_dist, n_iter=10,
    scoring="neg_mean_absolute_error", cv=tscv, random_state=42, n_jobs=-1
)
search.fit(train_full[feature_cols], train_full["sales"])
best_lgb = search.best_estimator_
print("Best params:", search.best_params_)

hgb_pred_test = best_lgb.predict(test_full[feature_cols])
hgb_pred_train = best_lgb.predict(train_full[feature_cols])
results["hgb_best_params"] = search.best_params_
results["hgb_test_wmape"] = wmape(test_full["sales"].values, hgb_pred_test)
results["hgb_train_wmape"] = wmape(train_full["sales"].values, hgb_pred_train)
results["hgb_test_mae"] = mean_absolute_error(test_full["sales"], hgb_pred_test)
print(f"LightGBM train WMAPE: {results['hgb_train_wmape']:.2f}% | test WMAPE: {results['hgb_test_wmape']:.2f}%")

print("=== Computing permutation importance ===")
perm_result = permutation_importance(
    best_lgb, test_full[feature_cols], test_full["sales"],
    n_repeats=10, random_state=42, n_jobs=-1, scoring="neg_mean_absolute_error"
)
importance_summary = dict(sorted(
    zip(feature_cols, perm_result.importances_mean.tolist()),
    key=lambda x: -x[1]
))
results["hgb_permutation_importance"] = importance_summary
print("Top features by permutation importance:", list(importance_summary.items())[:5])

test_full = test_full.copy()
test_full["hgb_pred"] = hgb_pred_test

def hgb_series_wmape(series_id):
    g = test_full[test_full["id"] == series_id]
    return wmape(g["sales"].values, g["hgb_pred"].values) if len(g) else np.nan

with open("reports/lstm_results.json") as f:
    lstm_data = json.load(f)
lstm_wmape = {k: v for k, v in lstm_data["lstm_wmape_by_series"].items()}
hv_ids = lstm_data["hv_ids"]
lv_ids = lstm_data["lv_ids"]

# ---------- SARIMA + Prophet, on the same sampled series used for LSTM, plus a capped H2 sample ----------
SAMPLE_PER_SEGMENT = 40
ss_ids = segments.loc[segments["segment_ss_lh"], "id"].sample(
    min(SAMPLE_PER_SEGMENT, segments["segment_ss_lh"].sum()), random_state=42).tolist()

relevant_ids = sorted(set(hv_ids) | set(lv_ids) | set(ss_ids))
print(f"\n=== Fitting SARIMA + Prophet on {len(relevant_ids)} sampled series ===")

sarima_wmape, prophet_wmape = {}, {}
for sid in relevant_ids:
    g_train = train_full[train_full["id"] == sid].sort_values("date")
    g_test = test_full[test_full["id"] == sid].sort_values("date")
    if len(g_train) < 60 or len(g_test) == 0:
        continue
    try:
        model = auto_arima(g_train["sales"].values, seasonal=True, m=7, suppress_warnings=True, error_action="ignore")
        fc = model.predict(n_periods=len(g_test))
        sarima_wmape[sid] = wmape(g_test["sales"].values, np.array(fc))
    except Exception as e:
        print(f"SARIMA failed for {sid}: {type(e).__name__}: {e}")
        sarima_wmape[sid] = np.nan
    try:
        pdf = g_train.rename(columns={"date": "ds", "sales": "y"})[["ds","y","sell_price","snap_CA","snap_TX","has_event"]]
        m = Prophet(weekly_seasonality=True, yearly_seasonality=False)
        for reg in ["sell_price","snap_CA","snap_TX","has_event"]:
            m.add_regressor(reg)
        m.fit(pdf)
        future = g_test.rename(columns={"date": "ds"})[["ds","sell_price","snap_CA","snap_TX","has_event"]]
        fcst = m.predict(future)
        prophet_wmape[sid] = wmape(g_test["sales"].values, fcst["yhat"].clip(lower=0).values)
    except Exception as e:
        prophet_wmape[sid] = np.nan

results["sarima_wmape_by_series"] = sarima_wmape
results["prophet_wmape_by_series"] = prophet_wmape
print(f"SARIMA mean WMAPE (fitted series): {np.nanmean(list(sarima_wmape.values())):.2f}%")
print(f"Prophet mean WMAPE (fitted series): {np.nanmean(list(prophet_wmape.values())):.2f}%")

# ---------- H2: Wilcoxon signed-rank test, SARIMA vs Prophet on strong-seasonality/low-holiday series ----------
# ss_ids = segments.loc[segments["segment_ss_lh"], "id"].tolist()
paired = [(sarima_wmape[i], prophet_wmape[i]) for i in ss_ids
          if i in sarima_wmape and i in prophet_wmape
          and not np.isnan(sarima_wmape[i]) and not np.isnan(prophet_wmape[i])]
if len(paired) >= 10:
    sarima_vals, prophet_vals = zip(*paired)
    stat, pval = wilcoxon(sarima_vals, prophet_vals)
    results["h2_wilcoxon"] = {"n_pairs": len(paired), "statistic": float(stat), "p_value": float(pval)}
    print(f"\n=== H2 Wilcoxon test (n={len(paired)}) ===")
    print(f"Statistic: {stat:.4f}, p-value: {pval:.4f}")
    print("Fail to reject H0 (no significant difference)" if pval > 0.05 else "Reject H0 (significant difference)")
else:
    print(f"\nOnly {len(paired)} valid pairs for H2 test — too few for a reliable Wilcoxon result.")

# ---------- LSTM: lightweight, on H1 segments only ----------
print("\n=== Training lightweight LSTM on H1 segments ===")
# h1_ids = segments.loc[segments["segment_hv_ni"] | segments["segment_lv_i"], "id"].tolist()
seq_features = ["sales","sell_price","wday","snap_CA","snap_TX","has_event"]
LOOKBACK = 28
# lstm_wmape = {}

results["lstm_wmape_by_series"] = lstm_wmape
hv_ni_lstm = [lstm_wmape[i] for i in segments.loc[segments["segment_hv_ni"], "id"] if i in lstm_wmape]
lv_i_lstm = [lstm_wmape[i] for i in segments.loc[segments["segment_lv_i"], "id"] if i in lstm_wmape]
print(f"LSTM mean WMAPE, high-volume/non-intermittent (n={len(hv_ni_lstm)}): {np.mean(hv_ni_lstm):.2f}%" if hv_ni_lstm else "No HV/NI LSTM results")
print(f"LSTM mean WMAPE, low-volume/intermittent (n={len(lv_i_lstm)}): {np.mean(lv_i_lstm):.2f}%" if lv_i_lstm else "No LV/I LSTM results")

# ---------- H1 comparison table ----------
def seg_mean(d, ids):
    vals = [d[i] for i in ids if i in d and np.isfinite(d[i])]
    return float(np.mean(vals)) if vals else None

# hv_ids = segments.loc[segments["segment_hv_ni"], "id"].tolist()
# lv_ids = segments.loc[segments["segment_lv_i"], "id"].tolist()

h1_table = {
    "high_volume_non_intermittent": {
        "SARIMA": seg_mean(sarima_wmape, hv_ids),
        "Prophet": seg_mean(prophet_wmape, hv_ids),
        "LightGBM": seg_mean({i: hgb_series_wmape(i) for i in hv_ids}, hv_ids),
        "LSTM": seg_mean(lstm_wmape, hv_ids),
    },
    "low_volume_intermittent": {
        "SARIMA": seg_mean(sarima_wmape, lv_ids),
        "Prophet": seg_mean(prophet_wmape, lv_ids),
        "LightGBM": seg_mean({i: hgb_series_wmape(i) for i in lv_ids}, lv_ids),
        "LSTM": seg_mean(lstm_wmape, lv_ids),
    },
}
results["h1_comparison_table"] = h1_table
print("\n=== H1 COMPARISON TABLE ===")
print(json.dumps(h1_table, indent=2))

with open("reports/model_comparison_summary.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved reports/model_comparison_summary.json")