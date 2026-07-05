"""Loading and schema validation for the daily GHI/weather dataset."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ghi_prediction.config import LEAK_COLS, TARGET

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "Timestamp",
    TARGET,
    *LEAK_COLS,
    "Suntime_Tot_sum",
    "AirTemp_C_mean",
    "RH_pct_mean",
]


class SchemaError(ValueError):
    """Raised when an input dataframe doesn't match the expected schema."""


def load_daily_data(path: str | Path, sheet_name: str = "Daily_Data") -> pd.DataFrame:
    """Load the raw daily dataset from an Excel file and validate its schema.

    Parameters
    ----------
    path: path to the .xlsx file (e.g. Final_Daily.xlsx)
    sheet_name: sheet containing the daily-resolution data

    Returns
    -------
    DataFrame sorted chronologically by Timestamp.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info("Loading %s (sheet=%s)", path, sheet_name)
    df = pd.read_excel(path, sheet_name=sheet_name)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    validate_schema(df)

    logger.info(
        "Loaded %d rows, %d columns (%s to %s)",
        len(df),
        df.shape[1],
        df["Timestamp"].min().date(),
        df["Timestamp"].max().date(),
    )
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Raise SchemaError if required columns are missing or mistyped.

    This is intentionally strict about *presence* of required columns but
    permissive about extra columns, so the pipeline tolerates new sensor
    channels being added upstream without breaking.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")

    if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
        try:
            pd.to_datetime(df["Timestamp"])
        except Exception as exc:  # noqa: BLE001
            raise SchemaError(f"Timestamp column is not parseable as dates: {exc}") from exc

    numeric_required = [c for c in REQUIRED_COLUMNS if c != "Timestamp"]
    non_numeric = [c for c in numeric_required if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise SchemaError(f"Expected numeric columns are non-numeric: {non_numeric}")

    if df.empty:
        raise SchemaError("Dataframe is empty")


def missing_target_report(df: pd.DataFrame) -> pd.Series:
    """Return count of missing target values grouped by year — the same
    diagnostic the exploratory notebook uses to justify dropping 2024."""
    return df.groupby(df["Timestamp"].dt.year)[TARGET].apply(lambda s: s.isnull().sum())
