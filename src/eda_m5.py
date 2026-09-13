"""
eda_m5.py
 
Exploratory Data Analysis on the cleaned, stratified-sample M5 dataset
(m5_clean_long.parquet). 
 
BEFORE RUNNING:
    pip install pandas matplotlib seaborn pyarrow
 
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_PATH = Path("data/processed/m5_clean_long.parquet")
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True) 
sns.set_style("whitegrid")  # Set seaborn style for plots

df = pd.read_parquet(DATA_PATH)
df["cat_id"] = df["cat_id"].astype(str)   # strips any leftover categorical metadata

# STEP 2: Describe the dataset
print("\nStep 2:Dataset Overview")
print(f"Rows: {len(df):,}, Columns: {(df.shape[1])}")
print(f"Unique series (id): {df['id'].nunique():,}")
print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"Categories: {sorted(df['cat_id'].unique())}")
print(f"Department: {sorted(df['dept_id'].unique())}")
print(f"Stores: {sorted(df['store_id'].unique())}")
print(f"States: {sorted(df['state_id'].unique())}")

# STEP 3: Descriptive statistics & Summary analysis

print("\nStep 3: Descriptive Statistics")
print("\n--- Sales: central tendency & spread ---")
print(df["sales"].describe())
print(f"Skewness: {df['sales'].skew():.3f}")
print(f"Zero-sales days: {(df['sales'] == 0).mean() * 100:.2f}%")

print("\n--- Sell price: central tendency & spread ---")
print(df["sell_price"].describe())

print("\n--- Frequency distribution: series count per category ---")
print(df.groupby('cat_id')["id"].nunique())

print("\n--- Sales share by category ---")
cat_sales = df.groupby('cat_id')["sales"].sum().sort_values(ascending=False)
print(cat_sales)
print((cat_sales / cat_sales.sum() * 100).round(2).astype(str) + "%")

print("\n--- Weekly seasonality: mean sales by day-of-week code ---")
print(df.groupby("wday")["sales"].mean().round(3))


print("\n--- Monthly seasonality: mean sales by month ---")
print(df.groupby("month")["sales"].mean().round(3))

print("\n--- SNAP effect on FOODS category sales ---")
foods = df[df["cat_id"] == "FOODS"]
print(f"Mean FOODS sales, SNAP_CA=1: {foods[foods['snap_CA'] == 1]['sales'].mean():.3f}")
print(f"Mean FOODS sales, SNAP_CA=0: {foods[foods['snap_CA'] == 0]['sales'].mean():.3f}")

print("\n--- Correlation matrix (sales, price, calendar, SNAP) ---")
corr_cols = ["sales", "sell_price", "wday", "month", "snap_CA", "snap_TX"]
corr = df[corr_cols].corr()
print(corr.round(3))

# STEP 4: Visualizations

print("STEP 4: GENERATING VISUALIZATIONS ->", FIG_DIR)

# 1. Histogram
fig, ax = plt.subplots(figsize = (7, 4.5))
df[df["sales"] < 15]["sales"].hist(bins=16, ax=ax, color='#4C72B0', edgecolor='white')
ax.set_title("Distribution of Daily Unit Sales (capped at 15 for readability)", fontsize=12)
ax.set_ylabel("Frequency")
ax.set_xlabel("Units sold per day")
plt.tight_layout()
plt.savefig(FIG_DIR / "01_sales_histogram.png", dpi=150)
plt.close()

# 2. Line plot - aggregate daily sales trend over time
daily = df.groupby("date")["sales"].sum().reset_index()
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(daily["date"], daily["sales"], linewidth=0.8, color='#4C72B0')
ax.set_title("Aggregate Daily Sales Across Sampled Series (2011, 2016)", fontsize=12)
ax.set_ylabel("Total Units Sold")
ax.set_xlabel("Date")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_trend.png", dpi=150)
plt.close()

# 3. Boxplot - weekly seasonality
fig, ax = plt.subplots(figsize=(7, 4.5))
sns.boxplot(data = df, x="wday", y = "sales", ax=ax, showfliers=False, color="#55A868")
ax.set_title("Sales Distribution by Day of Week", fontsize=12)
ax.set_ylabel("Units Sold")
ax.set_xlabel("wday code (as per M5 calendar convention)")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_weekday_boxplot.png", dpi=150)
plt.close()

# 4. Scatter plot - price vs. sales (sampled for point-count readability)
sample = df.sample(min(20000, len(df)), random_state=42)
fig, ax = plt.subplots(figsize=(7, 4.5))
for cat, color in zip(["FOODS", "HOBBIES"], ["#4C72B0", "#55A868"]):
    subset = sample[sample["cat_id"] == cat]
    ax.scatter(subset["sell_price"], subset["sales"], alpha=0.3, label=cat, color=color)
ax.set_title(f"Sell Price vs. Units Sold ({len(sample):,}-row sample)", fontsize=12)
ax.set_ylabel("Units Sold")
ax.set_xlabel("Sell Price ($)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "04_price_vs_sales_scatter.png", dpi=150)
plt.close()

# 5. Correlation heatmap
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation Heatmap: Sales, Price, Calendar & SNAP Features")
plt.tight_layout()
plt.savefig(FIG_DIR / "05_corr_heatmap.png", dpi=150)
plt.close()

# 6. Stacked bar chart — total sales by store and category
fig, ax = plt.subplots(figsize=(8, 4.5))
store_cat = df.groupby(["store_id", "cat_id"])["sales"].sum().unstack()
store_cat.plot(kind="bar", stacked=True, ax=ax, color=["#C44E52", "#4C72B0"])
ax.set_title("Total Sales by Store and Category")
ax.set_xlabel("Store")
ax.set_ylabel("Total units sold")
plt.tight_layout()
plt.savefig(FIG_DIR / "06_store_category_bar.png", dpi=150)
plt.close()

print(f"Saved 6 figures to {FIG_DIR}/")

# STEP 5 & 6: KEY FINDINGS + DATA ISSUES (computed, not just narrated)

df["z"] = df.groupby("id")["sales"].transform(
    lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) > 0 else 1)
)
spikes = df[df["z"].abs() > 4]
explained = (
    (spikes["event_name_1"] != "none")
    | (spikes["snap_CA"] == 1)
    | (spikes["snap_TX"] == 1)
)
print(f"Total spike rows (|z| > 4): {len(spikes):,}")
print(f"Spikes on a known calendar event day: {(spikes['event_name_1'] != 'none').sum():,}")
print(f"Spikes on a SNAP day (CA or TX): {((spikes['snap_CA']==1)|(spikes['snap_TX']==1)).sum():,}")
print(f"Spikes explained by event OR SNAP: {explained.sum():,} ({explained.mean()*100:.1f}%)")
print(f"Spikes with NO identified explanation: {(~explained).sum():,} ({(1-explained.mean())*100:.1f}%)")

 