# ☀️ GHI Prediction System

<p align="center">

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red)]()
[![Machine Learning](https://img.shields.io/badge/Machine-Learning-success)]()
[![License](https://img.shields.io/badge/Status-Production-brightgreen)]()

</p>

<p align="center">

### Predict **Global Horizontal Irradiance (GHI)** using inexpensive weather measurements instead of an expensive pyranometer.

</p>

<p align="center">

### 🚀 **Live Demo**

[![Open App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ghi-prediction-system-7xhs7emrslrvdlhzccop25.streamlit.app/)

**🔗 https://ghi-prediction-system-7xhs7emrslrvdlhzccop25.streamlit.app/**

</p>

---

# 📖 Overview

Global Horizontal Irradiance (**GHI**) is one of the most important measurements for evaluating solar energy potential.

Normally, GHI is measured using a **pyranometer**, an accurate but expensive instrument that requires regular calibration and maintenance.

Most weather stations, farms, and potential solar installation sites **do not own a pyranometer**, but they often record:

* 🌡 Temperature
* 💧 Humidity
* ☁ Cloud Cover
* 🌫 Visibility
* 🌞 Sunshine Duration
* 🌬 Wind Information

This project answers a practical question:

> **Can we accurately estimate GHI using only these cheaper weather measurements?**

The answer is **yes**.

A machine learning model is trained on data collected from locations that already have a pyranometer and then predicts GHI for locations where only weather observations are available.

---

# 🎯 Applications

✅ Solar Farm Site Assessment

Estimate solar potential before installing expensive equipment.

---

✅ Missing Sensor Recovery

If a pyranometer fails (as happened for an entire year in this dataset), the model can still estimate GHI.

---

✅ Low-Cost Large Scale Deployment

Train on a few reference stations and estimate GHI across many nearby locations using weather data alone.

---

# 🏗 Model Variants

| Model                     | Requires                 | Accuracy   | Best For                            |
| ------------------------- | ------------------------ | ---------- | ----------------------------------- |
| 🌞 **Full Model**         | Sunshine Duration Sensor | ⭐ Highest  | Sites with basic solar monitoring   |
| 🌦 **Weather-only Model** | Weather Variables Only   | ⭐ Moderate | Locations without any solar sensors |

---

# ⚙ Problem Formulation

### 🎯 Target Variable

```text
GHI_W_Avg_mean
```

Daily Mean Global Horizontal Irradiance (W/m²)

---

### 📅 Data Split

Instead of a random split, the project uses an **85/15 chronological split**.

Why?

Weather observations are time-dependent.

Random splitting would leak neighboring days into both train and test sets, resulting in unrealistically optimistic performance.

---

### 🧹 Data Cleaning

The original dataset contains one complete year where the pyranometer failed.

Since there is **no ground truth**, those rows are:

* ❌ Not imputed
* ❌ Not interpolated
* ✅ Removed from training

---

# 🤖 Models Compared

Several machine learning algorithms were evaluated.

| Model               | Used |
| ------------------- | ---- |
| 🌳 Random Forest    | ✅    |
| ⚡ XGBoost           | ✅    |
| 💡 LightGBM         | ✅    |
| 🤝 Average Ensemble | ✅    |
| 🧠 Ridge Stacking   | ✅    |

After experimentation, **Random Forest** consistently achieved the best balance of:

* Generalization
* Stability
* Lowest MAE
* Best performance on ~1000 daily observations with 70+ features

---

# 🔍 Explainability

The project is not a black box.

Model behavior is verified using:

* 📈 Feature Importance
* 🔎 SHAP Values

Expected physical relationships are preserved.

| Variable          | Effect on GHI     |
| ----------------- | ----------------- |
| Cloud Cover       | ⬇ Negative        |
| Visibility        | ⬆ Positive        |
| Temperature       | ⬆ Positive        |
| Sunshine Duration | ⬆ Strong Positive |

---

# 📂 Project Structure

```text
ghi_prediction/
│
├── config.py
├── data_loader.py
├── model.py
├── train.py
│
app/
├── streamlit_app.py
│
tests/
├── conftest.py
├── test_predictor.py
├── test_data_validation.py
├── test_model_regression.py
│
notebooks/
└── GHI_Prediction_System_final.ipynb
```

---

# 🚀 Installation

```bash
pip install -r requirements.txt
```

---

# 🏋 Train the Model

```bash
python -m ghi_prediction.train path/to/Final_Daily.xlsx --out ghi_predictor.pkl
```

---

# 🧪 Run Tests

```bash
PYTHONPATH=. pytest -v
```

The testing suite uses a **synthetic dataset** that reproduces:

* Missing GHI outage year
* Weather schema
* Missing cloud-layer columns
* Data validation rules

No proprietary dataset is required.

### Test Coverage

✅ Predictor functionality

✅ Pickling

✅ Weather-only inference

✅ Missing column handling

✅ Schema validation

✅ Train/Test leakage checks

✅ Regression testing for MAE stability

---

# 🌐 Run the Web Application

```bash
streamlit run app/streamlit_app.py
```

Upload your **Final_Daily.xlsx** file and explore:

### 🔮 Prediction

* Manual prediction
* Batch CSV prediction
* Full model
* Weather-only model

---

### 📊 Performance Dashboard

* MAE
* RMSE
* R² Score
* Predicted vs Actual plots

---

### 📈 Feature Importance

Visualize the most influential weather variables.

---

### 🧹 Data Quality

Inspect:

* Missing target by year
* Raw GHI timeline
* Dataset completeness

---

# 🐳 Docker

Build:

```bash
docker build -t ghi-prediction .
```

Run:

```bash
docker run -p 8501:8501 ghi-prediction
```

---

# ☁ Deployment

The application is ready for deployment on:

* ✅ Streamlit Community Cloud
* ✅ Docker
* ✅ GitHub

Deploy to Streamlit by pointing the service to:

```text
app/streamlit_app.py
```

---

# 🔄 Continuous Integration

GitHub Actions automatically runs on every push or pull request.

Pipeline includes:

* ✅ Unit Tests
* ✅ Regression Tests
* ✅ Data Validation
* ✅ Syntax Checking

Configuration:

```text
.github/workflows/ci.yml
```

---

# 📌 Highlights

✔ Production-ready ML pipeline

✔ Chronological evaluation (no temporal leakage)

✔ Weather-only inference mode

✔ Explainable AI with SHAP

✔ Interactive Streamlit dashboard

✔ Docker support

✔ Automated testing

✔ Continuous Integration

✔ Modular package architecture

---

# 📜 License

This project is intended for educational and research purposes.
