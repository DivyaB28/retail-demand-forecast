# Retail Demand Forecasting with Explainable ML

Forecasting weekly store/department sales, comparing a naive baseline, Prophet,
and LightGBM, with SHAP-based explainability and a Streamlit dashboard.

## Problem Statement

Retailers lose money two ways: **overstocking** (wasted capital, markdowns) and
**understocking** (lost sales, unhappy customers). This project forecasts
weekly demand at the store/department level and explains _why_ the model
predicts what it predicts, so the output is usable for real inventory
decisions, not just a chart.

**Business question:** Can we forecast next week's sales accurately enough to
inform inventory planning, and can we explain which factors (seasonality,
holidays, markdowns) drive that forecast?

## Data

Source: [Walmart Recruiting - Store Sales Forecasting (Kaggle)](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data)

Place the raw CSVs in `data/raw/` (see Step 2 in the project write-up).

## Repo Structure

```
retail-demand-forecast/
├── data/
│   ├── raw/              # original, untouched downloaded data
│   └── processed/        # cleaned/merged data ready for modeling
├── notebooks/            # exploratory analysis, one notebook per phase
├── src/                  # reusable Python modules (data prep, models, eval)
├── dashboard/            # Streamlit app
├── reports/
│   └── figures/          # saved plots for the final paper
├── tests/                # basic unit tests for src/ functions
├── requirements.txt
└── README.md
```

## Models (in order of complexity)

1. **Naive seasonal baseline** — same week, previous year
2. **Prophet** — statistical model with built-in trend/seasonality/holiday decomposition
3. **LightGBM** — gradient boosting with engineered lag/rolling features + SHAP explainability
4. _(Stretch goal)_ LSTM — deep learning comparison

## Metrics

- **WMAPE** (Weighted MAPE) — primary metric, volume-weighted
- **MASE** — scaled error vs. naive baseline

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
# or, to launch the dashboard:
streamlit run dashboard/app.py
```

## Status

- [ ] Data collected & cleaned
- [ ] EDA complete
- [ ] Naive baseline
- [ ] Prophet model
- [ ] LightGBM + SHAP
- [ ] Streamlit dashboard
- [ ] Final report
