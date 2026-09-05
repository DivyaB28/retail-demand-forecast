"""
clean_m5.py
 
Data cleaning and preprocessing pipeline for the M5 Forecasting - Accuracy
dataset (Kaggle: https://www.kaggle.com/competitions/m5-forecasting-accuracy).
 
REAL DATASET DIMENSIONS (for reference — confirmed against the official
Kaggle competition data as of this writing):
    sales_train_validation.csv : 30,490 rows x 1,919 cols (id + 6 keys + d_1..d_1913)
    sales_train_evaluation.csv : 30,490 rows x 1,947 cols (id + 6 keys + d_1..d_1941)
    calendar.csv                : 1,969 rows x 14 cols (covers train + 28-day
                                   forecast horizon; event_name_1/2 are ~92%
                                   empty by design — most days have no event)
    sell_prices.csv              : 6,841,121 rows x 4 cols
 
USAGE
-----
1. Download sales_train_validation.csv (or sales_train_evaluation.csv for
   the final 1941-day version), calendar.csv, and sell_prices.csv from the
   Kaggle competition page and place them in data/raw/. sample_submission.csv
   is a submission-format template, not raw data — it is intentionally
   excluded from this cleaning pipeline.
2. Run: python src/clean_m5.py
3. Cleaned, merged, long-format output is written to
   data/processed/m5_clean_long.csv, and a data-quality summary is
   printed to the console (and saved to reports/data_quality_summary.json).

WHAT THIS SCRIPT DOES
----------------------
1. Loads the three raw M5 files.
2. Standardizes formats (dates, dtypes, whitespace, currency strings).
3. Detects and removes exact duplicate records.
4. Melts the wide sales_train_validation table (one column per day) into
   long format (one row per item-store-day) — required for time-series
   modeling with Prophet/ARIMA/LSTM.
5. Merges sales + calendar + prices into a single analysis-ready table.
6. Identifies and documents missing values, distinguishing genuine gaps
   from structurally expected ones (e.g., no price before an item's
   launch week).
7. Flags statistical outliers (demand spikes) without deleting them,
   since in retail demand data, spikes are usually real signal (holidays,
   promotions) rather than errors — deleting them would bias the models
   this cleaned data will feed.
8. Applies a stratified sample scope (per the project proposal) so the
   output stays computationally manageable.
"""

from calendar import calendar
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

quality_log = {} # collects everything we find, for the write-up

def log(section, key, value):
    quality_log.setdefault(section, {})[key] = value


# 1. Load the three raw M5 files.
def load_raw():
    sales = pd.read_csv(RAW_DIR / "sales_train_validation.csv")
    calendar = pd.read_csv(RAW_DIR / "calendar.csv")
    prices = pd.read_csv(RAW_DIR / "sell_prices.csv")
    print("prices cols:", prices.columns.tolist())
    print("prices dtypes:\n", prices.dtypes)
    print("\ncalendar cols:", calendar.columns.tolist())
    print("calendar dtypes:\n", calendar.dtypes)
    print("\nsales cols sample:", sales.columns[:12].tolist())
    log("overview", "sales_shape_raw", list(sales.shape))
    log("overview", "calendar_shape_raw", list(calendar.shape))
    log("overview", "prices_shape_raw", list(prices.shape))
    return sales, calendar, prices

# 2. Quality checks
def audit_missing_values(df, name):
    missing = df.isna().sum()
    missing = missing[missing > 0]
    pct = (missing / len(df) * 100).round(2)
    result = {col: {"n_missing": int(missing[col]), "pct_missing": float(pct[col])} for col in missing.index}
    log("missing_values", name, result)
    return result

def audit_duplicates(df, name, subset= None):
    n_dupes = int(df.duplicated(subset=subset).sum())
    log("duplicates", name, n_dupes)
    return n_dupes

def audit_format_inconsistencies(calendar, prices):
    issues = {}
    # Date format check: anything that doesn't match strict YYYY-MM-DD after strip
    bad_dates = calendar["date"].astype(str).str.strip()
    non_iso = bad_dates[~bad_dates.str.match(r"^\d{4}-\d{2}-\d{2}$")]
    issues["calendar_non_iso_dates"] = non_iso.tolist()

# Price format check: values that aren't cleanly numeric (currency symbols, stray whitespace)
    def is_dirty(x):
        if isinstance(x, str):
            return not re.match(r"^-?\d+(\.\d+)?$", x.strip())
        return False
    dirty_prices = prices["sell_price"].apply(is_dirty)
    issues["prices_non_numeric_count"] = int(dirty_prices.sum())
    log("format_inconsistencies", "details", issues)
    return issues

def audit_outliers_and_invalid(sales_long):
    negative = sales_long[sales_long["sales"] < 0]
    log("outliers", "negative_sales_count", int(len(negative)))
    print(sales_long.dtypes)
    print(sales_long["sales"].head(10))
    print(sales_long.groupby("id").size().head(10)) 
    # z-score based spike flag, computed per item-store series.
    # Make a defensive copy, ensure numeric float, compute per-series mean/std
    sales_long = sales_long.copy()
    sales_long["sales"] = pd.to_numeric(sales_long["sales"], errors="coerce").fillna(0).astype(float)

    grp = sales_long.groupby("id")["sales"]
    means = grp.transform("mean")
    stds = grp.transform(lambda s: s.std(ddof=0))
    # Avoid division by zero for constant series
    stds = stds.replace(0, 1.0)

    sales_long["z"] = (sales_long["sales"] - means) / stds

    spikes = sales_long[sales_long["z"].abs() > 4]
    log("outliers", "spike_count_z_gt_4", int(len(spikes)))
    return negative, spikes

# 3. CLEANING
def clean_calendar(calendar):
    calendar = calendar.copy()
    before = len(calendar)
    calendar = calendar.drop_duplicates()
    log("cleaning_actions", "calendar_duplicates_removed", before - len(calendar))
 
    # Standardize date strings: strip whitespace, convert slash format to ISO
    calendar["date"] = calendar["date"].astype(str).str.strip()
    calendar["date"] = pd.to_datetime(calendar["date"], errors="coerce", format="mixed")
    n_unparsed = int(calendar["date"].isna().sum())
    log("cleaning_actions", "calendar_dates_unparseable", n_unparsed)
 
    # Event columns: real M5 uses NaN to mean "no event" — this is a
    # legitimate/expected missing value, not a data quality defect.
    # Fill with an explicit label so downstream encoding doesn't silently
    # drop the "no event" category.
    for col in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        if col in calendar.columns:
            calendar[col] = calendar[col].fillna("none")
 
    calendar["wday"] = calendar["wday"].astype("int8")
    calendar["month"] = calendar["month"].astype("int8")
    calendar["year"] = calendar["year"].astype("int16")
    return calendar

def clean_prices(prices):
    prices = prices.copy()
    before = len(prices)
    prices = prices.drop_duplicates()
    log("cleaning_actions", "prices_duplicates_removed", before - len(prices))
 
    def parse_price(x):
        if isinstance(x, str):
            x = x.strip().replace("$", "")
        try:
            return float(x)
        except (ValueError, TypeError):
            return np.nan
 
    prices["sell_price"] = prices["sell_price"].apply(parse_price)
    n_unparsed = int(prices["sell_price"].isna().sum())
    log("cleaning_actions", "prices_unparseable_after_cleaning", n_unparsed)
    prices = prices.dropna(subset=["sell_price"])
    return prices

 
def melt_and_clean_sales(sales, calendar):
    sales = sales.copy()
    before = len(sales)
    sales = sales.drop_duplicates()
    log("cleaning_actions", "sales_duplicates_removed", before - len(sales))
 
    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]
 
    long = sales.melt(id_vars=id_vars, value_vars=day_cols,
                       var_name="d", value_name="sales")
 
    # Merge day -> actual calendar date
    # Normalize join keys and only select calendar columns that exist
    long["d"] = long["d"].astype(str).str.strip()
    calendar = calendar.copy()
    if "d" not in calendar.columns:
        raise KeyError("calendar DataFrame must contain 'd' column for merging with sales long format")
    calendar["d"] = calendar["d"].astype(str).str.strip()
    cal_cols = ["d", "date", "wm_yr_wk", "wday", "month", "year",
                "event_name_1", "event_type_1", "snap_CA", "snap_TX"]
    cal_cols = [c for c in cal_cols if c in calendar.columns]
    long = long.merge(calendar[cal_cols], on="d", how="left")
 
    # Negative sales are data-entry errors (physically impossible for unit
    # sales in this dataset) — clip to zero rather than drop the row, so we
    # don't lose the rest of that day's legitimate record for the series.
    n_negative = int((long["sales"] < 0).sum())
    long["sales"] = long["sales"].clip(lower=0)
    log("cleaning_actions", "negative_sales_clipped_to_zero", n_negative)
 
    long["sales"] = long["sales"].astype("int32")
    return long
 
def merge_prices(long, prices):
    # before merged = long.merge(...)
    long["store_id"] = long["store_id"].astype(str).str.strip()
    long["item_id"] = long["item_id"].astype(str).str.strip()
    long["wm_yr_wk"] = pd.to_numeric(long["wm_yr_wk"], errors="coerce").astype("Int64").astype("int64")

    prices["store_id"] = prices["store_id"].astype(str).str.strip()
    prices["item_id"] = prices["item_id"].astype(str).str.strip()
    prices["wm_yr_wk"] = pd.to_numeric(prices["wm_yr_wk"], errors="coerce").astype("Int64").astype("int64")

    merged = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    # Missing sell_price here is structurally expected: it means the item
    # had not yet launched at that store in that week. This is NOT a
    # random missing-data problem — document it rather than imputing blindly.
    n_missing_price = int(merged["sell_price"].isna().sum())
    pct_missing_price = round(n_missing_price / len(merged) * 100, 2)
    log("missing_values", "sell_price_missing_pre_launch",
        {"n_missing": n_missing_price, "pct_missing": pct_missing_price})
    return merged

def trim_pre_launch_rows(merged):
    """Remove rows before each item-store series' actual launch date.
 
    Rationale: sell_price is null for any date before an item was first
    sold at a given store (see merge_prices()). Leaving those rows in
    means every affected series starts with a long, artificial run of
    zero sales that reflects "not yet stocked," not real demand — this
    would bias any trend/seasonality model into learning a false pattern
    at the start of the series. Trimming to first-launch date per series
    removes that artifact while keeping every subsequent zero (a
    legitimate no-sale day) intact.
    """
    before = len(merged)
    launch_date = (merged.dropna(subset=["sell_price"])
                   .groupby("id")["date"].min()
                   .rename("launch_date"))
    merged = merged.merge(launch_date, on="id", how="left")
    trimmed = merged[merged["date"] >= merged["launch_date"]].drop(columns=["launch_date"])
    log("cleaning_actions", "pre_launch_rows_trimmed", before - len(trimmed))
    return trimmed

def apply_stratified_sample(merged, categories=None, max_items_per_cat=None):
    """Per the project proposal's feasibility scoping, keep a representative
    stratified sample rather than processing the full ~30,490-series
    hierarchy. On the REAL dataset, set max_items_per_cat (e.g., 50) to
    control size; leave None to keep everything (used here on the small
    synthetic sample, which is already tiny)."""
    if categories:
        merged = merged[merged["cat_id"].isin(categories)]
    if max_items_per_cat:
        keep_ids = (merged[["cat_id", "item_id"]].drop_duplicates()
                    .groupby("cat_id").head(max_items_per_cat)["item_id"])
        merged = merged[merged["item_id"].isin(keep_ids)]
    return merged

# MAIN

def main():
    sales, calendar, prices = load_raw()

    # --- audit raw files before touching them ---
    audit_missing_values(sales, "sales_raw")
    audit_missing_values(calendar, "calendar_raw")
    audit_missing_values(prices, "prices_raw")
    audit_duplicates(sales, "sales_raw")
    audit_duplicates(calendar, "calendar_raw")
    audit_duplicates(prices, "prices_raw")
    audit_format_inconsistencies(calendar, prices)
    
    # --- clean each table ---
    calendar_clean = clean_calendar(calendar)
    prices_clean = clean_prices(prices)
    sales_long = melt_and_clean_sales(sales, calendar_clean)

    # --- merge ---
    merged = merge_prices(sales_long, prices_clean)

    # --- outlier audit (post-merge, on cleaned long table) ---
    negative, spikes = audit_outliers_and_invalid(merged)

    # --- trim pre-launch rows (see function docstring for rationale) ---
    merged = trim_pre_launch_rows(merged)

    # --- apply proposal's feasibility scope ---
    # This scoped sample (not the full 46M-row table) is what should be
    # committed to the repo / submitted for the assignment — small enough
    # for GitHub, and it's the actual data the modeling phase will use.
    final = apply_stratified_sample(
        merged,
        categories=["FOODS", "HOBBIES"],   # adjust to your chosen categories
        max_items_per_cat=50,              # adjust for your target sample size
    )

    log("overview", "final_shape", list(final.shape))
    log("overview", "final_n_unique_series", int(final["id"].nunique()))
    log("overview", "final_date_range", [str(final["date"].min()), str(final["date"].max())])

    out_path = PROCESSED_DIR / "m5_clean_long.csv"
    final_out = final.drop(columns=["z"], errors="ignore")
    final_out.to_csv(out_path, index=False)

    # Parquet copy: columnar + compressed, typically 8-15x
    # smaller than the CSV for data this repetitive (many repeated
    # categorical/integer values), and it preserves dtypes exactly on
    # reload (no re-parsing dates/ints from text). Recommended as the
    # primary working format going forward; keep the CSV only if a
    # specific tool requires plain text.
    
    parquet_path = PROCESSED_DIR / "m5_clean_long.parquet"
    final_out.to_parquet(parquet_path, index=False, compression="snappy")
    csv_mb = out_path.stat().st_size / (1024 ** 2)
    parquet_mb = parquet_path.stat().st_size / (1024 ** 2)
    log("overview", "csv_size_mb", round(csv_mb, 1))
    log("overview", "parquet_size_mb", round(parquet_mb, 1))
 

    with open(REPORTS_DIR / "data_quality_summary.json", "w") as f:
        json.dump(quality_log, f, indent=2, default=str)
 
    print("\n=== DATA QUALITY SUMMARY ===")
    print(json.dumps(quality_log, indent=2, default=str))
    print(f"\nCleaned data written to: {out_path} ({csv_mb:.1f} MB)")
    print(f"Parquet copy written to: {parquet_path}  ({parquet_mb:.1f} MB)")
    print(f"Quality summary written to: {REPORTS_DIR / 'data_quality_summary.json'}")
 

if __name__ == "__main__":
    main()