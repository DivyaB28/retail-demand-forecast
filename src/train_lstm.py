import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models

df = pd.read_parquet("data/processed/m5_feature_engineered.parquet")
df = df.sort_values(["id", "date"]).reset_index(drop=True)
segments = pd.read_csv("reports/series_segments.csv")

df_model = df.dropna(subset=["sales_lag_7","sales_lag_28","rolling_mean_7","rolling_std_7"]).copy()
cutoff_date = df_model["date"].max() - pd.Timedelta(days=28)

SAMPLE_PER_SEGMENT = 40
hv_ids = segments.loc[segments["segment_hv_ni"], "id"].sample(
    min(SAMPLE_PER_SEGMENT, segments["segment_hv_ni"].sum()), random_state=42).tolist()
lv_ids = segments.loc[segments["segment_lv_i"], "id"].sample(
    min(SAMPLE_PER_SEGMENT, segments["segment_lv_i"].sum()), random_state=42).tolist()
h1_ids = sorted(set(hv_ids) | set(lv_ids))

def wmape(actual, pred):
    return np.sum(np.abs(actual - pred)) / np.sum(np.abs(actual)) * 100

seq_features = ["sales","sell_price","wday","snap_CA","snap_TX","has_event"]
LOOKBACK = 28
lstm_wmape = {}

print(f"Training LSTM on {len(h1_ids)} series...")
for n, sid in enumerate(h1_ids, 1):
    g = df_model[df_model["id"] == sid].sort_values("date")
    if len(g) < LOOKBACK + 35:
        continue
    scaler = MinMaxScaler()
    g_train = g[g["date"] <= cutoff_date]
    g_test = g[g["date"] > cutoff_date]
    scaler.fit(g_train[seq_features])
    scaled_all = scaler.transform(g[seq_features])
    scaled_df = pd.DataFrame(scaled_all, columns=seq_features, index=g.index)

    X, y = [], []
    for i in range(LOOKBACK, len(g)):
        X.append(scaled_df.iloc[i-LOOKBACK:i].values)
        y.append(scaled_df.iloc[i]["sales"])
    X, y = np.array(X), np.array(y)
    split_idx = len(g_train) - LOOKBACK
    if split_idx <= 0 or split_idx >= len(X):
        continue
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test = X[split_idx:]

    model = models.Sequential([
        layers.Input(shape=(LOOKBACK, len(seq_features))),
        layers.LSTM(16),
        layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mae")
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    pred_scaled = model.predict(X_test, verbose=0).flatten()

    sales_min = scaler.data_min_[seq_features.index("sales")]
    sales_max = scaler.data_max_[seq_features.index("sales")]
    pred_actual = pred_scaled * (sales_max - sales_min) + sales_min
    actual_test = g_test["sales"].values[:len(pred_actual)]
    if len(actual_test):
        lstm_wmape[sid] = wmape(actual_test, np.clip(pred_actual, 0, None))
    if n % 10 == 0:
        print(f"  {n}/{len(h1_ids)} series done")

hv_ni_lstm = [lstm_wmape[i] for i in hv_ids if i in lstm_wmape]
lv_i_lstm = [lstm_wmape[i] for i in lv_ids if i in lstm_wmape]
print(f"LSTM mean WMAPE, high-volume/non-intermittent (n={len(hv_ni_lstm)}): "
      f"{np.mean(hv_ni_lstm):.2f}%" if hv_ni_lstm else "No HV/NI results")
print(f"LSTM mean WMAPE, low-volume/intermittent (n={len(lv_i_lstm)}): "
      f"{np.mean(lv_i_lstm):.2f}%" if lv_i_lstm else "No LV/I results")

with open("reports/lstm_results.json", "w") as f:
    json.dump({"lstm_wmape_by_series": lstm_wmape, "hv_ids": hv_ids, "lv_ids": lv_ids}, f, indent=2)
print("Saved reports/lstm_results.json")