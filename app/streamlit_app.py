"""GHI Prediction System — Streamlit app.

Run locally:
    streamlit run app/streamlit_app.py

Deploy: push this repo to GitHub and point Streamlit Community Cloud at
app/streamlit_app.py, or build the included Dockerfile.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ghi_prediction.config import SUNTIME_COL, TARGET
from ghi_prediction.data_loader import SchemaError, load_daily_data, missing_target_report
from ghi_prediction.model import GHIPredictor
from ghi_prediction.train import chronological_split

st.set_page_config(page_title="GHI Prediction System", page_icon="☀️", layout="wide")


# --------------------------------------------------------------------------- #
# Data / model loading
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Training model...")
def train_predictor(file_bytes: bytes, sheet_name: str) -> tuple[GHIPredictor, pd.DataFrame]:
    tmp_path = Path("/tmp/_uploaded_ghi_data.xlsx")
    tmp_path.write_bytes(file_bytes)
    df = load_daily_data(tmp_path, sheet_name=sheet_name)
    data = df.dropna(subset=[TARGET]).reset_index(drop=True)
    predictor = GHIPredictor().fit(data)
    return predictor, data


@st.cache_data(show_spinner=False)
def get_holdout_predictions(_predictor: GHIPredictor, data: pd.DataFrame):
    X = data.drop(columns=["Timestamp", TARGET] + GHIPredictor.LEAK_COLS)
    y = data[TARGET]
    _, X_test, _, y_test, split_idx = chronological_split(X, y)
    test_rows = data.iloc[split_idx:]
    preds = _predictor.predict(test_rows)
    return test_rows["Timestamp"].values, y_test.values, preds


# --------------------------------------------------------------------------- #
# Sidebar — data source
# --------------------------------------------------------------------------- #
st.sidebar.title("☀️ GHI Prediction System")
st.sidebar.markdown("Upload your daily weather/solar dataset to train and use the model.")

uploaded = st.sidebar.file_uploader("Final_Daily.xlsx", type=["xlsx"])
sheet_name = st.sidebar.text_input("Sheet name", value="Daily_Data")

if uploaded is None:
    st.title("☀️ GHI Prediction System")
    st.info(
        "👈 Upload your `Final_Daily.xlsx` in the sidebar to get started. "
        "The app will train the model in-session and let you explore predictions, "
        "performance, and feature importance."
    )
    st.markdown(
        """
        **What this app does**
        - Trains a Random Forest model to predict Global Horizontal Irradiance (GHI)
          from daily weather and cloud-sensor data.
        - Two model variants: a **full model** (uses recorded sunshine duration) and a
          **weather-only model** (usable when predicting from forecast data alone).
        - Chronological train/test split — avoids leaking future days into evaluation.
        """
    )
    st.stop()

try:
    predictor, data = train_predictor(uploaded.getvalue(), sheet_name)
except SchemaError as e:
    st.error(f"Data schema problem: {e}")
    st.stop()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load/train on this file: {e}")
    st.stop()

st.sidebar.success(f"Model trained on {len(data)} usable rows.")

tab_predict, tab_performance, tab_importance, tab_data = st.tabs(
    ["🔮 Predict", "📊 Performance", "🧠 Feature Importance", "🗂️ Data Quality"]
)

# --------------------------------------------------------------------------- #
# Tab: Predict
# --------------------------------------------------------------------------- #
with tab_predict:
    st.header("Make a prediction")
    mode = st.radio(
        "Model variant",
        ["Full model (uses sunshine sensor)", "Weather-only model (forecast-friendly)"],
        horizontal=True,
    )
    weather_only = mode.startswith("Weather-only")

    input_method = st.radio("Input method", ["Manual entry", "Batch upload (CSV)"], horizontal=True)

    if input_method == "Manual entry":
        st.markdown("Enter values for the key drivers; other features default to training medians.")
        cols = st.columns(3)
        row = {}
        key_features = [c for c in (predictor.feature_cols_ns_ if weather_only else predictor.feature_cols_)]
        # Surface the top-importance features as editable inputs; rest default silently.
        top_feats = predictor.feature_importance(weather_only=weather_only).head(6).index.tolist()

        defaults = data[[c for c in top_feats if c in data.columns]].median(numeric_only=True)
        for i, feat in enumerate(top_feats):
            with cols[i % 3]:
                default_val = float(defaults.get(feat, 0.0))
                row[feat] = st.number_input(feat, value=round(default_val, 2))

        if st.button("Predict GHI", type="primary"):
            # Build a full-width row using training medians, then overwrite with user input.
            base_row = data.drop(columns=["Timestamp", TARGET] + GHIPredictor.LEAK_COLS).median(numeric_only=True)
            full_row = base_row.copy()
            for k, v in row.items():
                full_row[k] = v
            input_df = pd.DataFrame([full_row])
            pred = predictor.predict(input_df, weather_only=weather_only)[0]
            st.metric("Predicted GHI", f"{pred:.1f} W/m²")

    else:
        st.markdown("Upload a CSV with the same columns as the training data (extra columns are ignored).")
        batch_file = st.file_uploader("Batch CSV", type=["csv"], key="batch")
        if batch_file is not None:
            batch_df = pd.read_csv(batch_file)
            try:
                preds = predictor.predict(batch_df, weather_only=weather_only)
                result = batch_df.copy()
                result["Predicted_GHI"] = preds
                st.dataframe(result, use_container_width=True)
                st.download_button(
                    "Download predictions as CSV",
                    result.to_csv(index=False).encode(),
                    file_name="ghi_predictions.csv",
                    mime="text/csv",
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Prediction failed: {e}")

# --------------------------------------------------------------------------- #
# Tab: Performance
# --------------------------------------------------------------------------- #
with tab_performance:
    st.header("Holdout performance (chronological split)")
    timestamps, y_test, preds = get_holdout_predictions(predictor, data)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mae:.2f} W/m²")
    c2.metric("RMSE", f"{rmse:.2f} W/m²")
    c3.metric("R²", f"{r2:.3f}")

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.scatter(
            x=y_test, y=preds, labels={"x": "Actual GHI", "y": "Predicted GHI"},
            title="Predicted vs. Actual",
        )
        lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
        fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines", name="Perfect prediction",
                                  line=dict(dash="dash", color="red")))
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=timestamps, y=y_test, name="Actual", mode="lines+markers"))
        fig2.add_trace(go.Scatter(x=timestamps, y=preds, name="Predicted", mode="lines+markers"))
        fig2.update_layout(title="Test period: Actual vs Predicted over time", yaxis_title="GHI (W/m²)")
        st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------------------------------- #
# Tab: Feature importance
# --------------------------------------------------------------------------- #
with tab_importance:
    st.header("What drives the prediction?")
    which = st.radio("Model", ["Full model", "Weather-only model"], horizontal=True)
    importances = predictor.feature_importance(weather_only=(which == "Weather-only model")).head(15)
    fig = px.bar(
        x=importances.values[::-1], y=importances.index[::-1], orientation="h",
        labels={"x": "Importance", "y": ""}, title=f"Top 15 features — {which}",
    )
    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------- #
# Tab: Data quality
# --------------------------------------------------------------------------- #
with tab_data:
    st.header("Data quality overview")
    st.write(f"**{len(data)}** usable rows (rows with a valid target).")
    st.subheader("Missing target by year (in the raw uploaded file)")
    raw_df = load_daily_data(Path("/tmp/_uploaded_ghi_data.xlsx"), sheet_name=sheet_name)
    st.bar_chart(missing_target_report(raw_df))
    st.subheader("GHI over time")
    fig = px.scatter(raw_df, x="Timestamp", y=TARGET, title="GHI over time — gaps show missing/unusable periods")
    st.plotly_chart(fig, use_container_width=True)
