import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/m5_clean_long.parquet")
df = df.sort_values(by=["id", "date"]).reset_index(drop=True)

print("BEFORE FEATURE ENGINEERING")
print("Shape:", df.shape)
print("Columns:", list(df.columns))

# --- Step 2 rubric: derived / new features ---
df["has_event"] = df["event_name_1"].notna().astype(int)
df["is_weekend"] = df["wday"].isin([1, 7]).astype(int)  # M5 convention: 1=Sat, 7=Fri
df["price_tier"] = pd.qcut(df["sell_price"], 4, labels=["low", "mid", "high", "premium"])

# --- Lag / rolling features (leakage-safe: shift(1) before rolling) ---
grp = df.groupby("id")["sales"]
df["sales_lag_7"] = grp.shift(7)
df["sales_lag_28"] = grp.shift(28)
df["rolling_mean_7"] = grp.transform(lambda s: s.shift(1).rolling(7).mean())
df["rolling_std_7"] = grp.transform(lambda s: s.shift(1).rolling(7).std())

# --- Encoding: one-hot for low-cardinality categoricals ---
df_encoded = pd.get_dummies(df, columns=["cat_id", "dept_id", "state_id"], drop_first=True)

print("\n AFTER FEATURE ENGINEERING ")
print("Shape:", df_encoded.shape)
print("New columns added:", [c for c in df_encoded.columns if c not in df.columns] )

print("\n NaN INTRODUCED BY LAG/ROLLING FEATURES ")
for col in ["sales_lag_7", "sales_lag_28", "rolling_mean_7", "rolling_std_7"]:
    print(f"{col}: {df[col].isna().mean()*100:.2f}% missing "
          f"({df[col].isna().sum()} rows, out of {len(df)})")

print("\n SAMPLE OF NEW FEATURES ")
print(df[["id", "date", "sales", "sales_lag_7", "rolling_mean_7", "has_event", "is_weekend", "price_tier"]].head(10))

print("\n PRICE TIER DISTRIBUTION ")
print(df["price_tier"].value_counts())

# Diagnose
print(df["event_name_1"].unique()[:20])
print(df["event_name_1"].apply(type).value_counts())

# Fix: treat empty string / literal "nan" as missing too
df["event_name_1"] = df["event_name_1"].replace("none", np.nan)
df["has_event"] = df["event_name_1"].notna().astype(int)
print(df["has_event"].value_counts(normalize=True))

print("\n HAS_EVENT / IS_WEEKEND COUNTS ")
print(df["has_event"].value_counts())
print(df["is_weekend"].value_counts())

# Save the finalized feature-engineered dataset 
out_path = "data/processed/m5_feature_engineered.parquet"
df_encoded.to_parquet(out_path, index=False, compression="snappy")
print(f"\nSaved feature-engineered dataset to {out_path}, size on disk:")
import os
print(f"{os.path.getsize(out_path) / (1024*1024):.2f} MB")