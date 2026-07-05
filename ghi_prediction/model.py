"""Deployable GHIPredictor — preprocessing + trained model in one object.

This is a direct, hardened port of the `GHIPredictor` class from the
exploratory notebook: same cleaning logic (cloud-layer-aware imputation,
leakage-column dropping), same two-model design (full model using the
sunshine-duration sensor, weather-only model for forecast-only inputs).
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ghi_prediction.config import DEFAULT_MODEL_PARAMS, LEAK_COLS, SUNTIME_COL, TARGET
from ghi_prediction.data_loader import SchemaError, validate_schema

logger = logging.getLogger(__name__)


def _default_ghi_model() -> RandomForestRegressor:
    """Module-level factory (not a lambda) so GHIPredictor stays picklable."""
    return RandomForestRegressor(**DEFAULT_MODEL_PARAMS)


class NotFittedError(RuntimeError):
    """Raised when predict()/feature_importance() is called before fit()."""


class GHIPredictor:
    """Production wrapper for GHI prediction.

    Usage
    -----
    >>> predictor = GHIPredictor()
    >>> predictor.fit(df)                          # df = raw Daily_Data dataframe
    >>> predictor.predict(new_df)                  # full model (needs Suntime_Tot_sum)
    >>> predictor.predict(new_df, weather_only=True)  # no Suntime needed
    """

    LEAK_COLS = LEAK_COLS
    TARGET = TARGET
    SUNTIME_COL = SUNTIME_COL

    def __init__(
        self,
        model_factory: Optional[Callable] = None,
        model_factory_ns: Optional[Callable] = None,
    ):
        self.model_factory = model_factory or _default_ghi_model
        self.model_factory_ns = model_factory_ns or self.model_factory
        self.model_: Optional[RandomForestRegressor] = None
        self.model_ns_: Optional[RandomForestRegressor] = None
        self.feature_cols_: Optional[list[str]] = None
        self.feature_cols_ns_: Optional[list[str]] = None
        self._impute_medians: dict = {}
        self._height_max: dict = {}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _clean(self, df: pd.DataFrame, fit_mode: bool = False) -> pd.DataFrame:
        df = df.copy()
        drop_cols = [c for c in self.LEAK_COLS + ["Timestamp", self.TARGET] if c in df.columns]
        X = df.drop(columns=drop_cols)

        # Cloud layer 2/3: missing = layer wasn't detected, not a sensor gap.
        l2l3_cover = [c for c in X.columns if ("CloudCover_L2" in c or "CloudCover_L3" in c)]
        l2l3_height = [c for c in X.columns if ("CloudBaseHeight_L2" in c or "CloudBaseHeight_L3" in c)]

        for c in l2l3_cover:
            X[c + "_missing"] = X[c].isnull().astype(int)
            X[c] = X[c].fillna(0)
        for c in l2l3_height:
            if fit_mode:
                self._height_max[c] = X[c].max()
            X[c] = X[c].fillna(self._height_max.get(c, X[c].max()))

        # Remaining genuine single-day gaps -> median impute (medians are
        # learned at fit time and frozen for inference, to avoid leakage).
        remaining = X.columns[X.isnull().any()]
        for c in remaining:
            if fit_mode:
                self._impute_medians[c] = X[c].median()
            X[c] = X[c].fillna(self._impute_medians.get(c, X[c].median()))

        return X

    def _check_fitted(self) -> None:
        if self.model_ is None or self.model_ns_ is None:
            raise NotFittedError("Call .fit(df) before .predict() or .feature_importance().")

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame) -> "GHIPredictor":
        """Fit both the full and weather-only models on `df`.

        Rows with a missing target are dropped (no ground truth to train on).
        """
        if self.TARGET not in df.columns:
            raise SchemaError(f"Training data must contain target column '{self.TARGET}'")

        data = df.dropna(subset=[self.TARGET]).reset_index(drop=True)
        if data.empty:
            raise ValueError("No rows with a valid target were found — nothing to train on.")

        logger.info("Fitting GHIPredictor on %d usable rows", len(data))

        X = self._clean(data, fit_mode=True)
        y = data[self.TARGET]

        self.feature_cols_ = X.columns.tolist()
        self.model_ = self.model_factory()
        self.model_.fit(X, y)

        X_ns = X.drop(columns=[self.SUNTIME_COL]) if self.SUNTIME_COL in X.columns else X
        self.feature_cols_ns_ = X_ns.columns.tolist()
        self.model_ns_ = self.model_factory_ns()
        self.model_ns_.fit(X_ns, y)

        return self

    def predict(self, df: pd.DataFrame, weather_only: bool = False):
        """Predict GHI for new rows.

        weather_only=True uses the model that doesn't require the
        sunshine-duration sensor (use when predicting from forecast data).
        """
        self._check_fitted()
        X = self._clean(df, fit_mode=False)
        if weather_only:
            X = X.reindex(columns=self.feature_cols_ns_, fill_value=0)
            return self.model_ns_.predict(X)
        X = X.reindex(columns=self.feature_cols_, fill_value=0)
        return self.model_.predict(X)

    def feature_importance(self, weather_only: bool = False) -> pd.Series:
        self._check_fitted()
        model = self.model_ns_ if weather_only else self.model_
        cols = self.feature_cols_ns_ if weather_only else self.feature_cols_
        return pd.Series(model.feature_importances_, index=cols).sort_values(ascending=False)
