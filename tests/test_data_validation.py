"""Tests for data_loader schema validation."""
from __future__ import annotations

import pandas as pd
import pytest

from ghi_prediction.data_loader import SchemaError, missing_target_report, validate_schema
from ghi_prediction.config import TARGET


def test_validate_schema_passes_on_good_data(synthetic_df):
    validate_schema(synthetic_df)  # should not raise


def test_validate_schema_fails_on_missing_target(synthetic_df):
    bad = synthetic_df.drop(columns=[TARGET])
    with pytest.raises(SchemaError):
        validate_schema(bad)


def test_validate_schema_fails_on_missing_timestamp(synthetic_df):
    bad = synthetic_df.drop(columns=["Timestamp"])
    with pytest.raises(SchemaError):
        validate_schema(bad)


def test_validate_schema_fails_on_empty_dataframe(synthetic_df):
    empty = synthetic_df.iloc[0:0]
    with pytest.raises(SchemaError):
        validate_schema(empty)


def test_validate_schema_fails_on_non_numeric_target(synthetic_df):
    bad = synthetic_df.copy()
    bad[TARGET] = "not a number"
    with pytest.raises(SchemaError):
        validate_schema(bad)


def test_missing_target_report_groups_by_year(synthetic_df):
    report = missing_target_report(synthetic_df)
    # The synthetic fixture simulates a full 2023 outage.
    assert report.loc[2023] == (synthetic_df["Timestamp"].dt.year == 2023).sum()


def test_no_leakage_columns_reach_feature_matrix(synthetic_df):
    """Regression guard: LEAK_COLS must never end up in the trained feature set."""
    from ghi_prediction.model import GHIPredictor

    predictor = GHIPredictor().fit(synthetic_df)
    for col in predictor.LEAK_COLS:
        assert col not in predictor.feature_cols_
        assert col not in predictor.feature_cols_ns_


def test_no_train_test_date_overlap(synthetic_df):
    """Regression guard for the chronological split: train and test dates
    must not overlap (would indicate an accidental random shuffle)."""
    from ghi_prediction.train import chronological_split
    from ghi_prediction.config import TARGET as T

    data = synthetic_df.dropna(subset=[T]).reset_index(drop=True)
    X = data.drop(columns=["Timestamp"])
    y = data[T]
    X_train, X_test, y_train, y_test, split_idx = chronological_split(X, y)

    train_dates = data["Timestamp"].iloc[:split_idx]
    test_dates = data["Timestamp"].iloc[split_idx:]
    assert train_dates.max() <= test_dates.min()
