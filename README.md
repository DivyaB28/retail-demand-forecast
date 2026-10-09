# Comparative Time-Series Forecasting of Retail Demand

**Evaluating Classical, Statistical, and Deep Learning Approaches on the M5 Dataset**

Capstone project for MSDS-699-A01: Data Science Capstone: Practicum, University of the Cumberlands.
Author: Divya Basavaraju

This project compares four ways of forecasting short-horizon (28-day) daily demand for
Walmart item-store series from the M5 Forecasting - Accuracy dataset: **SARIMA**,
**Prophet**, **gradient boosting**, and a lightweight **LSTM**, against a naive seasonal
baseline and a linear regression baseline.

## Problem Statement

Retailers lose money two ways: **overstocking** (tied-up capital, waste, markdowns) and
**understocking** (lost sales, unhappy customers). No single forecasting method suits
every product, because fast-selling staples and rarely bought items behave very
differently. This project asks which method a retailer should rely on for which kind of
item, and how far those recommendations can be trusted.

**Research question:** Among classical statistical (SARIMA), modern statistical (Prophet),
and deep learning (LSTM) approaches, which most accurately predicts 7 to 28 day demand
for individual Walmart store-department series, and how do series characteristics
(volume, intermittency, seasonality strength) relate to each model's relative
performance?

**Hypotheses**

- **H1:** LSTM achieves lower WMAPE than SARIMA and Prophet on high-volume,
  non-intermittent series, but not on low-volume, intermittent series.
- **H2 (null):** There is no significant difference in WMAPE between SARIMA and Prophet on
  series with strong, regular weekly seasonality and minimal holiday effects.

## Data

Source: [M5 Forecasting - Accuracy (Kaggle)](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data),
released by Walmart for the fifth Makridakis forecasting competition.

| Stage                                                 | Extent                                                               |
| ----------------------------------------------------- | -------------------------------------------------------------------- |
| Raw wide sales table                                  | 30,490 series x 1,913 days                                           |
| Merged long format                                    | 58,327,370 rows                                                      |
| After launch-date trimming (full analysis-ready data) | 46,027,957 rows                                                      |
| Stratified sample used for modelling                  | 1,901,730 rows, 1,000 series (500 FOODS_1, 500 HOBBIES_1; 10 stores) |
| Feature-engineered sample                             | 26 columns                                                           |

The raw M5 files and the full 46-million-row dataset are **not** included in this
repository because of Kaggle's data terms and file size. To reproduce the pipeline,
download the M5 files from the link above and place them in `data/raw/`
(this folder is git-ignored). The cleaned and feature-engineered sample files are stored
as Parquet in `data/processed/`.

**Scope note:** the sample covers two of the seven M5 departments (FOODS_1 and
HOBBIES_1) and data ending April 2016, so results are a methodological demonstration,
not a claim about current Walmart operations or the full product catalogue.

## Repository Structure

```
retail-demand-forecast/
├── data/
│   ├── raw/                  # original M5 files (not tracked; download from Kaggle)
│   └── processed/            # cleaned and feature-engineered Parquet files
├── src/                      # pipeline scripts (see "How to Run")
├── reports/
│   ├── series_segments.csv           # per-series statistics and segment flags
│   ├── lstm_results.json             # per-series LSTM WMAPE and sampled series ids
│   ├── model_comparison_summary.json # all model results, H1 table, H2 test
│   └── figures/                      # figures used in the report
├── .gitignore
├── Requirements.txt
└── README.md
```

## Pipeline and Scripts

| Step                    | Script                                 | Output                                         |
| ----------------------- | -------------------------------------- | ---------------------------------------------- |
| 1. Clean and sample     | `src/clean_m5.py`                      | cleaned stratified sample (Parquet)            |
| 2. Exploratory analysis | `src/eda_m5.py`                        | EDA figures and statistics                     |
| 3. Feature engineering  | [feature engineering script in `src/`] | `data/processed/m5_feature_engineered.parquet` |
| 4. Baseline models      | [baseline script in `src/`]            | naive and linear regression metrics            |
| 5. Segment series       | `src/segment_series.py`                | `reports/series_segments.csv`                  |
| 6. Train LSTM           | `src/train_lstm.py`                    | `reports/lstm_results.json`                    |
| 7. Compare all models   | `src/model_comparison.py`              | `reports/model_comparison_summary.json`        |
| 8. Generate figures     | `src/generate_figures.py`              | `reports/figures/fig1-3 *.png`                 |

## Methods

- **Target:** daily unit sales (a zero-inflated, right-skewed count variable; 58.70% of
  series-days are zero).
- **Features:** one-hot encoded category, department and state; weekend, event and price-tier
  flags; sales lagged 7 and 28 days; trailing 7-day rolling mean and standard deviation
  (computed on shifted data to avoid leakage).
- **Validation:** chronological split; the final 28 days (28 March to 24 April 2016) are
  held out, and all models are scored on the same window.
- **Segments:** high-volume/non-intermittent (206 series), low-volume/intermittent (225),
  strong-seasonality/low-holiday (250). SARIMA, Prophet and LSTM are fitted per series on a
  seeded sample of 40 series per segment (`random_state = 42`).
- **Models:** naive seasonal rule, linear regression, SARIMA (`pmdarima.auto_arima`,
  seasonal period 7), Prophet (weekly seasonality plus price, SNAP and event regressors),
  `HistGradientBoostingRegressor` (tuned with `RandomizedSearchCV` and `TimeSeriesSplit`),
  and a single-layer LSTM (16 units, 28-day look-back, 10 epochs).
- **Metrics:** **WMAPE** (primary; sum of absolute errors divided by sum of actual sales),
  plus MAE and RMSE for the baseline stage. H2 is tested with a Wilcoxon signed-rank test.
- **Explainability:** permutation importance on the gradient-boosting model.

**Tool substitutions.** scikit-learn's `HistGradientBoostingRegressor` replaced LightGBM
(which could not be compiled on the development machine), and permutation importance
replaced SHAP (SHAP's `TreeExplainer` is incompatible with this model). Both are
documented in the final report.

## Results Summary

Mean WMAPE (%) on the 28-day holdout, 40 series per segment:

| Segment                       | SARIMA | Prophet | Gradient boosting | LSTM      |
| ----------------------------- | ------ | ------- | ----------------- | --------- |
| High-volume, non-intermittent | 96.60  | 95.85   | 91.23             | **83.02** |
| Low-volume, intermittent      | 177.89 | 179.24  | 231.79            | 104.45    |

- **H1:** supported in substance. LSTM is best on high-volume series. Its low error on
  intermittent series is an artefact of predicting close to zero (per-series errors cluster
  around 100%, the score of an all-zero forecast), so SARIMA or Prophet is the safer choice
  there. H1 was assessed by comparing averages, not by a formal significance test.
- **H2:** not rejected (Wilcoxon W = 373, p = .627, n = 40 pairs).
- **Predictors:** the 7-day rolling mean of sales dominates feature importance; calendar and
  SNAP indicators contribute almost nothing once recent sales are known.
- **Baselines:** linear regression improved on the naive rule (MAE 1.431 to 1.194, RMSE
  2.997 to 2.274).

Full details are in the final project report.

## How to Run

The project uses **two Python environments**, because TensorFlow requires `numpy<2`,
which conflicts with other packages.

**Main environment** (cleaning, EDA, baselines, SARIMA, Prophet, gradient boosting, figures):

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r Requirements.txt
```

**LSTM environment** (only for `src/train_lstm.py`):

```bash
python -m venv venv_lstm
source venv_lstm/bin/activate
pip install "numpy<2" tensorflow==2.15.0 pandas scikit-learn pyarrow
```

Then run the scripts from the project root in the order shown in the pipeline table. In
particular, run `src/train_lstm.py` in `venv_lstm` **before** `src/model_comparison.py`,
which reads `reports/lstm_results.json`. `src/model_comparison.py` fits SARIMA and Prophet
on roughly 119 series and takes a long time (about one to two hours on a laptop).
`src/generate_figures.py` only reads existing result files and runs in seconds.

## Status

- [x] Data collected and cleaned
- [x] EDA complete
- [x] Feature engineering
- [x] Naive and linear regression baselines
- [x] SARIMA, Prophet, gradient boosting and LSTM comparison
- [x] Hypothesis tests (H1, H2)
- [x] Final report
- [ ] Future work: all-department rolling-origin evaluation, intermittent-demand methods,
      tuned or global LSTM, SHAP-based local explanations, interactive dashboard

## References

Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy competition:
Results, findings, and conclusions. _International Journal of Forecasting, 38_(4),
1346-1364. https://doi.org/10.1016/j.ijforecast.2021.11.013
