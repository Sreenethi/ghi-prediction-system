# GHI Prediction System

Production pipeline and web app for predicting Global Horizontal Irradiance
(GHI) from daily weather and cloud-sensor data.

## Motivation

Global Horizontal Irradiance is normally measured directly with a
**pyranometer** — an accurate but expensive instrument that requires
calibration and maintenance. As a result, most sites (weather stations,
farms, candidate solar installation sites) never have one installed, even
though basic weather data — temperature, humidity, cloud cover, visibility,
sometimes sunshine duration — is far more commonly available from cheaper
sensors or public weather services.

This project asks a practical question: **can GHI be estimated accurately
enough from that cheaper, more widely available weather data, instead of
requiring a pyranometer at every site?**

The pipeline trains on data from a location that *does* have a pyranometer,
then lets that model predict GHI at locations that *don't* — using only
weather variables. This is useful for:

- **Solar site assessment** — estimating solar potential at a candidate site
  before investing in pyranometer infrastructure there.
- **Filling sensor gaps** — if a pyranometer goes offline (as happened for a
  full year in the dataset this was built on), GHI can still be estimated
  from the weather data that kept recording.
- **Scaling to many locations cheaply** — install pyranometers at a few
  reference sites, train a model, and extrapolate to nearby regions using
  weather data alone.

Two model variants reflect two real-world equipment situations:

| Variant | Requires | Accuracy | Use case |
|---|---|---|---|
| **Full model** | Sunshine-duration sensor (cheap) | Higher | Site has basic solar monitoring, but no pyranometer |
| **Weather-only model** | General weather data only | Lower, but non-zero | Site has no solar-specific equipment at all |

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
