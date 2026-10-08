import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_parquet("data/processed/m5_feature_engineered.parquet")
df = df.sort_values(["id", "date"]).reset_index(drop=True)

# Drop rows with NaN from lag/rolling features (can't train on them)
df_model = df.dropna(subset=["sales_lag_7", "sales_lag_28", "rolling_mean_7", "rolling_std_7"]).copy()
print("Rows after dropping lag/rolling NaNs:", df_model.shape)

# Chronological split: last 28 days = test
cutoff_date = df_model["date"].max() - pd.Timedelta(days=28)
train = df_model[df_model["date"] <= cutoff_date]
test = df_model[df_model["date"] > cutoff_date]
print("Train rows:", len(train), "| Test rows:", len(test))
print("Train date range:", train["date"].min(), "to", train["date"].max())
print("Test date range:", test["date"].min(), "to", test["date"].max())

feature_cols = [
    "sell_price", "wday", "month", "has_event", "is_weekend",
    "snap_CA", "snap_TX", "sales_lag_7", "sales_lag_28",
    "rolling_mean_7", "rolling_std_7",
    "cat_id_HOBBIES", "dept_id_HOBBIES_1", "state_id_TX", "state_id_WI",
]

X_train, y_train = train[feature_cols], train["sales"]
X_test, y_test = test[feature_cols], test["sales"]

# --- Model 1: Naive seasonal baseline ---
naive_pred = test["sales_lag_7"]
naive_mae = mean_absolute_error(y_test, naive_pred)
naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
print("\n=== NAIVE BASELINE (sales = sales_lag_7) ===")
print(f"MAE: {naive_mae:.4f}")
print(f"RMSE: {naive_rmse:.4f}")

# --- Model 2: Linear Regression ---
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
print("\n=== LINEAR REGRESSION ===")
print(f"MAE: {lr_mae:.4f}")
print(f"RMSE: {lr_rmse:.4f}")
print(f"% negative predictions: {(lr_pred < 0).mean()*100:.2f}%")

print("\n=== COEFFICIENTS ===")
for name, coef in sorted(zip(feature_cols, lr.coef_), key=lambda x: -abs(x[1])):
    print(f"{name}: {coef:.4f}")
print(f"Intercept: {lr.intercept_:.4f}")

print("\n=== SAMPLE PREDICTIONS vs ACTUAL ===")
sample = test[["id", "date", "sales"]].copy()
sample["lr_pred"] = lr_pred
sample["naive_pred"] = naive_pred.values
print(sample.head(15))

sns.set_style("whitegrid")
test_out = test[["id", "date", "sales", "cat_id_HOBBIES"]].copy()
test_out["lr_pred"] = lr_pred
test_out["naive_pred"] = naive_pred.values
test_out["lr_error"] = test_out["sales"] - test_out["lr_pred"]
test_out["naive_error"] = test_out["sales"] - test_out["naive_pred"]

# 1. Predicted vs Actual scatter (LR)
plt.figure(figsize=(6,6))
plt.scatter(test_out["sales"], test_out["lr_pred"], alpha=0.05, s=10)
plt.plot([0, test_out["sales"].max()], [0, test_out["sales"].max()], "r--", label="Perfect prediction")
plt.xlabel("Actual sales"); plt.ylabel("Predicted sales")
plt.title("Linear Regression: Predicted vs Actual (Test Set)")
plt.legend(); plt.tight_layout()
plt.savefig("reports/figures/fig1_pred_vs_actual.png", dpi=150)
plt.close()

# 2. MAE/RMSE comparison bar chart
metrics_df = pd.DataFrame({
    "Model": ["Naive (lag-7)", "Linear Regression"],
    "MAE": [naive_mae, lr_mae],
    "RMSE": [naive_rmse, lr_rmse],
})
metrics_df.plot(x="Model", y=["MAE", "RMSE"], kind="bar", figsize=(6,4))
plt.title("Baseline Model Comparison"); plt.ylabel("Error")
plt.xticks(rotation=0); plt.tight_layout()
plt.savefig("reports/figures/fig2_metric_comparison.png", dpi=150)
plt.close()

# 3. Time series: actual vs predicted for one sample series over the test window
sample_id = test_out["id"].unique()[0]
s = test_out[test_out["id"] == sample_id].sort_values("date")
plt.figure(figsize=(10,4))
plt.plot(s["date"], s["sales"], label="Actual", marker="o")
plt.plot(s["date"], s["lr_pred"], label="LR Predicted", marker="x")
plt.plot(s["date"], s["naive_pred"], label="Naive Predicted", linestyle="--")
plt.xticks(rotation=45); plt.legend(); plt.title(f"Actual vs Predicted — {sample_id}")
plt.tight_layout()
plt.savefig("reports/figures/fig3_series_forecast.png", dpi=150)
plt.close()

# 4. Coefficient magnitude bar chart
coef_df = pd.DataFrame({"feature": feature_cols, "coef": lr.coef_}).sort_values("coef")
plt.figure(figsize=(7,6))
plt.barh(coef_df["feature"], coef_df["coef"])
plt.axvline(0, color="black", linewidth=0.8)
plt.title("Linear Regression Coefficients")
plt.tight_layout()
plt.savefig("reports/figures/fig4_coefficients.png", dpi=150)
plt.close()

# 5. Error distribution: LR vs Naive
plt.figure(figsize=(7,4))
plt.hist(test_out["lr_error"], bins=40, alpha=0.6, label="LR error")
plt.hist(test_out["naive_error"], bins=40, alpha=0.6, label="Naive error")
plt.axvline(0, color="black", linewidth=0.8)
plt.legend(); plt.title("Prediction Error Distribution")
plt.tight_layout()
plt.savefig("reports/figures/fig5_error_distribution.png", dpi=150)
plt.close()

# 6. Category-stratified error (ties back to your WMAPE category point from EDA)
test_out["category"] = np.where(test_out["cat_id_HOBBIES"] == 1, "HOBBIES", "FOODS")
cat_mae = test_out.groupby("category").apply(
    lambda g: mean_absolute_error(g["sales"], g["lr_pred"])
)
print("\n MAE BY CATEGORY ")
print(cat_mae)