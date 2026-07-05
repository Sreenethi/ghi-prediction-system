"""Candidate model training, chronological evaluation, and CLI entry point."""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ghi_prediction.config import RANDOM_STATE, SPLIT_FRAC, TARGET
from ghi_prediction.data_loader import load_daily_data
from ghi_prediction.model import GHIPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def chronological_split(X: pd.DataFrame, y: pd.Series, split_frac: float = SPLIT_FRAC):
    """Split by time order (not randomly) — daily weather rows are
    autocorrelated, so a random split would leak neighboring days into the
    test set and overstate accuracy."""
    n = len(X)
    split_idx = int(n * split_frac)
    return (
        X.iloc[:split_idx], X.iloc[split_idx:],
        y.iloc[:split_idx], y.iloc[split_idx:],
        split_idx,
    )


def evaluate(name, model, Xtr, ytr, Xte, yte, verbose: bool = True):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    mae = mean_absolute_error(yte, pred)
    rmse = mean_squared_error(yte, pred) ** 0.5
    r2 = r2_score(yte, pred)
    if verbose:
        logger.info("%-20s MAE=%7.2f  RMSE=%7.2f  R2=%.3f", name, mae, rmse, r2)
    return model, pred, {"mae": mae, "rmse": rmse, "r2": r2}


def candidate_models() -> dict:
    return {
        "RandomForest": RandomForestRegressor(
            n_estimators=400, max_depth=7, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "XGBoost": xgb.XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=1.0, random_state=RANDOM_STATE,
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=1.0, random_state=RANDOM_STATE, verbose=-1,
        ),
    }


def run_comparison(X_train, y_train, X_test, y_test) -> tuple[pd.DataFrame, dict]:
    """Train all candidates, return a results table sorted by MAE and the
    dict of fitted models."""
    results, fitted_models = {}, {}
    for name, model in candidate_models().items():
        fitted, _, metrics = evaluate(name, model, X_train, y_train, X_test, y_test)
        results[name] = metrics
        fitted_models[name] = fitted

    avg_pred = np.mean([m.predict(X_test) for m in fitted_models.values()], axis=0)
    results["Ensemble (avg)"] = {
        "mae": mean_absolute_error(y_test, avg_pred),
        "rmse": mean_squared_error(y_test, avg_pred) ** 0.5,
        "r2": r2_score(y_test, avg_pred),
    }

    results_df = pd.DataFrame(results).T.sort_values("mae")
    return results_df, fitted_models


def main():
    parser = argparse.ArgumentParser(description="Train the GHI prediction model.")
    parser.add_argument("data_path", help="Path to Final_Daily.xlsx")
    parser.add_argument("--sheet-name", default="Daily_Data")
    parser.add_argument("--out", default="ghi_predictor.pkl", help="Output pickle path")
    args = parser.parse_args()

    df = load_daily_data(args.data_path, sheet_name=args.sheet_name)
    data = df.dropna(subset=[TARGET]).reset_index(drop=True)
    logger.info("Usable rows after dropping missing-target rows: %d", len(data))

    predictor = GHIPredictor()
    predictor.fit(data)

    out_path = Path(args.out)
    with open(out_path, "wb") as f:
        pickle.dump(predictor, f)
    logger.info("Saved trained predictor to %s", out_path)

    logger.info("Top 10 features (full model):\n%s", predictor.feature_importance().head(10))


if __name__ == "__main__":
    main()
