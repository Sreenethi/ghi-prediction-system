"""Central configuration for the GHI prediction pipeline.

Keeping these as module-level constants (rather than hardcoding them inline)
means the whole pipeline can be re-pointed at a different target/schema by
editing one file.
"""

# Target column and columns that leak the target (derived from it, must
# never be used as model inputs).
TARGET = "GHI_W_Avg_mean"
LEAK_COLS = ["GHI_W_Avg_min", "GHI_W_Avg_max", "GHI_W_Avg_std"]

# Feature used only in the "full" model — dropped for the weather-only
# variant since it requires a ground-station sunshine sensor.
SUNTIME_COL = "Suntime_Tot_sum"

# Chronological train/test split fraction (most recent (1 - frac) held out).
SPLIT_FRAC = 0.85

# Random seed used everywhere for reproducibility.
RANDOM_STATE = 42

# Default hyperparameters for the production model (RandomForest was the
# empirical winner in the exploratory notebook on this dataset size).
DEFAULT_MODEL_PARAMS = dict(
    n_estimators=400,
    max_depth=7,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# Regression-test tolerance: CI fails if MAE on the fixed holdout drifts
# worse than this multiplier of the last known-good MAE.
REGRESSION_TOLERANCE = 1.15
