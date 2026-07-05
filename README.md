# GHI Prediction System

Production pipeline and web app for predicting Global Horizontal Irradiance
(GHI) from daily weather and cloud-sensor data.

## Problem & approach

- **Target**: `GHI_W_Avg_mean` (daily mean solar irradiance, W/m²).
- **Data quality**: the target is missing for a full sensor-outage year in
  the source data; those rows are dropped, not imputed, since there's no
  ground truth to train against.
- **Split**: chronological (not random) 85/15 train/test — daily weather
  rows are autocorrelated, so a random split would leak neighboring days
  into evaluation and overstate accuracy.
- **Models compared**: Random Forest, XGBoost, LightGBM, a simple average
  ensemble, and a Ridge-stacked ensemble. Random Forest wins on this
  dataset size (under ~1,000 usable rows, 70+ features) — heavily
  regularized models generalize best here, verified empirically rather
  than assumed.
- **Two deployed variants**:
  - **Full model** — uses recorded sunshine duration (`Suntime_Tot_sum`),
    the dominant predictor. Highest accuracy; requires a ground station.
  - **Weather-only model** — drops sunshine duration, for use when
    predicting from forecast data alone. Reported honestly as lower
    accuracy, not hidden.
- **Explainability**: feature importances + SHAP values are used to sanity
  check that predictions follow physically expected relationships (cloud
  cover negative, visibility/temperature positive).

## Project structure

```
ghi_prediction/       # importable package
  config.py           # target/feature/hyperparameter config
  data_loader.py       # loading + schema validation
  model.py             # GHIPredictor — preprocessing + trained model
  train.py             # candidate comparison, chronological split, CLI
app/
  streamlit_app.py     # deployable web app
tests/
  conftest.py           # synthetic-data fixtures (no real data needed)
  test_predictor.py
  test_data_validation.py
  test_model_regression.py
notebooks/
  GHI_Prediction_System_final.ipynb   # original exploratory notebook
```

## Setup

```bash
pip install -r requirements.txt
```

## Train a model (CLI)

```bash
python -m ghi_prediction.train path/to/Final_Daily.xlsx --out ghi_predictor.pkl
```

## Run the tests

```bash
PYTHONPATH=. pytest -v
```

Tests run against a synthetic dataset (`tests/conftest.py`) that mimics the
real schema, including the outage-year pattern and legitimately-missing
cloud-layer columns — no proprietary data required to validate the pipeline.

Includes:
- **Unit tests** for `GHIPredictor` (fit/predict contracts, pickling,
  weather-only mode, graceful handling of unseen/missing columns).
- **Data validation tests** (schema checks, no train/test date overlap, no
  leakage columns reaching the feature matrix).
- **Model regression test** — fails CI if holdout MAE drifts past a
  tolerance band on a fixed dataset, catching silent accuracy regressions.

## Run the app

```bash
streamlit run app/streamlit_app.py
```

Upload your `Final_Daily.xlsx` in the sidebar. The app trains in-session and
gives you:
- **Predict** — manual entry or batch CSV upload, either model variant.
- **Performance** — MAE/RMSE/R² and predicted-vs-actual charts on the
  chronological holdout.
- **Feature Importance** — top drivers for each model variant.
- **Data Quality** — missing-target-by-year and raw GHI-over-time view.

## Deploy

**Streamlit Community Cloud**: push this repo to GitHub, point Streamlit
Cloud at `app/streamlit_app.py`.

**Docker**:
```bash
docker build -t ghi-prediction .
docker run -p 8501:8501 ghi-prediction
```

## CI

`.github/workflows/ci.yml` runs the full test suite and a syntax check on
every push/PR to `main`.
