"""Model-quality regression test.

Guards against silent accuracy regressions: if someone edits feature
engineering or swaps a model and holdout MAE gets meaningfully worse on a
FIXED synthetic dataset, this test fails CI. On the real dataset, replace
`EXPECTED_MAE_CEILING` with your own measured baseline from the notebook's
holdout evaluation (see section 6 of the notebook).
"""
from __future__ import annotations

from sklearn.metrics import mean_absolute_error

from ghi_prediction.config import REGRESSION_TOLERANCE, TARGET
from ghi_prediction.model import GHIPredictor
from ghi_prediction.train import chronological_split

# Baseline MAE measured once on the synthetic fixture at commit time.
# Regenerate with: pytest tests/test_model_regression.py -s to print the
# current value, then update this constant if the change was intentional.
EXPECTED_MAE_CEILING = 25.0


def test_holdout_mae_within_tolerance(synthetic_df):
    data = synthetic_df.dropna(subset=[TARGET]).reset_index(drop=True)
    X = data.drop(columns=["Timestamp", TARGET] + GHIPredictor.LEAK_COLS)
    y = data[TARGET]

    X_train, X_test, y_train, y_test, _ = chronological_split(X, y)

    predictor = GHIPredictor()
    predictor.fit(data.iloc[: len(X_train)])
    preds = predictor.predict(data.iloc[len(X_train):])

    mae = mean_absolute_error(y_test, preds)
    print(f"\nHoldout MAE: {mae:.2f} (ceiling: {EXPECTED_MAE_CEILING * REGRESSION_TOLERANCE:.2f})")

    assert mae < EXPECTED_MAE_CEILING * REGRESSION_TOLERANCE, (
        f"Holdout MAE {mae:.2f} exceeds allowed ceiling "
        f"{EXPECTED_MAE_CEILING * REGRESSION_TOLERANCE:.2f} — check for a "
        "regression in feature engineering or model config."
    )
