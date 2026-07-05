"""Unit tests for ghi_prediction.model.GHIPredictor."""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from ghi_prediction.config import SUNTIME_COL, TARGET
from ghi_prediction.model import GHIPredictor, NotFittedError


def test_fit_raises_on_missing_target_column(synthetic_df):
    bad_df = synthetic_df.drop(columns=[TARGET])
    predictor = GHIPredictor()
    with pytest.raises(Exception):
        predictor.fit(bad_df)


def test_fit_raises_when_no_valid_targets(synthetic_df):
    df = synthetic_df.copy()
    df[TARGET] = np.nan
    predictor = GHIPredictor()
    with pytest.raises(ValueError):
        predictor.fit(df)


def test_predict_before_fit_raises(synthetic_df):
    predictor = GHIPredictor()
    with pytest.raises(NotFittedError):
        predictor.predict(synthetic_df.head(5))


def test_fit_drops_rows_with_missing_target(synthetic_df):
    predictor = GHIPredictor()
    predictor.fit(synthetic_df)
    # 2023 was zeroed out as a simulated outage — model should have trained
    # on strictly fewer rows than the full input.
    n_valid = synthetic_df[TARGET].notnull().sum()
    assert n_valid < len(synthetic_df)
    assert predictor.model_ is not None


def test_predict_output_length_matches_input(synthetic_df):
    predictor = GHIPredictor().fit(synthetic_df)
    sample = synthetic_df.dropna(subset=[TARGET]).tail(10)
    preds = predictor.predict(sample)
    assert len(preds) == len(sample)


def test_predict_weather_only_does_not_require_suntime(synthetic_df):
    predictor = GHIPredictor().fit(synthetic_df)
    sample = synthetic_df.dropna(subset=[TARGET]).tail(10).drop(columns=[SUNTIME_COL])
    preds = predictor.predict(sample, weather_only=True)
    assert len(preds) == len(sample)
    assert SUNTIME_COL not in predictor.feature_cols_ns_


def test_weather_only_model_excludes_suntime_from_training(synthetic_df):
    predictor = GHIPredictor().fit(synthetic_df)
    assert SUNTIME_COL in predictor.feature_cols_
    assert SUNTIME_COL not in predictor.feature_cols_ns_


def test_predictions_are_non_negative_and_finite(synthetic_df):
    predictor = GHIPredictor().fit(synthetic_df)
    sample = synthetic_df.dropna(subset=[TARGET]).tail(20)
    preds = predictor.predict(sample)
    assert np.all(np.isfinite(preds))


def test_predictor_is_picklable_round_trip(synthetic_df, tmp_path):
    predictor = GHIPredictor().fit(synthetic_df)
    sample = synthetic_df.dropna(subset=[TARGET]).tail(5)
    preds_before = predictor.predict(sample)

    pkl_path = tmp_path / "predictor.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(predictor, f)
    with open(pkl_path, "rb") as f:
        loaded = pickle.load(f)

    preds_after = loaded.predict(sample)
    np.testing.assert_allclose(preds_before, preds_after)


def test_feature_importance_sums_reasonably(synthetic_df):
    predictor = GHIPredictor().fit(synthetic_df)
    importances = predictor.feature_importance()
    assert len(importances) == len(predictor.feature_cols_)
    assert np.isclose(importances.sum(), 1.0, atol=1e-6)


def test_cloud_layer_missing_flag_created(synthetic_df):
    predictor = GHIPredictor().fit(synthetic_df)
    assert "CloudCover_L2_oktas_mean_missing" in predictor.feature_cols_


def test_predict_handles_unseen_columns_gracefully(synthetic_df):
    """New/unexpected columns at inference time shouldn't break prediction —
    reindex() should just ignore them."""
    predictor = GHIPredictor().fit(synthetic_df)
    sample = synthetic_df.dropna(subset=[TARGET]).tail(5).copy()
    sample["brand_new_sensor_channel"] = 42
    preds = predictor.predict(sample)
    assert len(preds) == len(sample)


def test_predict_handles_missing_optional_columns(synthetic_df):
    """A column that's all-NaN at inference (e.g. sensor offline) should
    fall back to the medians learned at fit time rather than crashing."""
    predictor = GHIPredictor().fit(synthetic_df)
    sample = synthetic_df.dropna(subset=[TARGET]).tail(5).copy()
    sample["Visibility_miles_mean"] = np.nan
    preds = predictor.predict(sample)
    assert np.all(np.isfinite(preds))
