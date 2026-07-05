"""Shared pytest fixtures.

We don't ship the real Final_Daily.xlsx (proprietary sensor data), so tests
run against a synthetic dataset that mimics its schema and quirks:
  - a target column with realistic missingness (including an all-missing year,
    like the real 2024 sensor outage)
  - cloud layer 2/3 columns that are legitimately missing (layer not detected)
  - cyclical date features already present (as they are in the real file)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ghi_prediction.config import TARGET


def _make_synthetic_daily_data(n_days: int = 730, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=n_days, freq="D")
    doy = dates.dayofyear.values

    suntime = rng.uniform(0, 12, n_days)
    air_temp = 25 + 5 * np.sin(2 * np.pi * doy / 365) + rng.normal(0, 1, n_days)
    rh = np.clip(80 + rng.normal(0, 10, n_days), 30, 100)
    visibility = np.clip(rng.normal(8, 2, n_days), 0, 15)

    ghi = 15 * suntime + 0.5 * air_temp - 0.3 * rh + rng.normal(0, 10, n_days)
    ghi = np.clip(ghi, 0, None)

    df = pd.DataFrame({
        "Timestamp": dates,
        "year": dates.year,
        "month": dates.month,
        "day": dates.day,
        "day_of_year": doy,
        "GHI_W_Avg_mean": ghi,
        "GHI_W_Avg_min": ghi * 0.0,
        "GHI_W_Avg_max": ghi * 1.4,
        "GHI_W_Avg_std": np.abs(rng.normal(150, 30, n_days)),
        "Suntime_Tot_sum": suntime,
        "AirTemp_C_mean": air_temp,
        "AirTemp_C_std": np.abs(rng.normal(2, 0.5, n_days)),
        "RH_pct_mean": rh,
        "RH_pct_std": np.abs(rng.normal(10, 3, n_days)),
        "Visibility_miles_mean": visibility,
        "Visibility_miles_max": visibility + rng.uniform(0, 2, n_days),
        "CloudCover_L1_oktas_mean": rng.uniform(0, 8, n_days),
        # Legitimately missing ~30% of the time (layer not detected that day).
        "CloudCover_L2_oktas_mean": np.where(
            rng.random(n_days) < 0.3, np.nan, rng.uniform(0, 8, n_days)
        ),
        "CloudCover_L3_oktas_mean": np.where(
            rng.random(n_days) < 0.5, np.nan, rng.uniform(0, 8, n_days)
        ),
        "CloudBaseHeight_L2_ft_mean": np.where(
            rng.random(n_days) < 0.3, np.nan, rng.uniform(2000, 10000, n_days)
        ),
        "CloudBaseHeight_L3_ft_mean": np.where(
            rng.random(n_days) < 0.5, np.nan, rng.uniform(5000, 10000, n_days)
        ),
        "humidity_range": np.abs(rng.normal(30, 10, n_days)),
        "cloud_transmission_index": rng.uniform(0, 1, n_days),
        "cloud_total_mean": rng.uniform(0, 8, n_days),
        "month_sin": np.sin(2 * np.pi * dates.month / 12),
        "month_cos": np.cos(2 * np.pi * dates.month / 12),
        "doy_sin": np.sin(2 * np.pi * doy / 365),
        "doy_cos": np.cos(2 * np.pi * doy / 365),
    })

    # A few genuine single-day gaps in an otherwise-complete column.
    genuine_gap_idx = rng.choice(n_days, size=3, replace=False)
    df.loc[genuine_gap_idx, "AirTemp_C_mean"] = np.nan

    # Simulate a full-year sensor outage (like real 2024) in year 2 of the range.
    outage_mask = df["Timestamp"].dt.year == 2023
    df.loc[outage_mask, [TARGET, "GHI_W_Avg_min", "GHI_W_Avg_max", "GHI_W_Avg_std"]] = np.nan

    return df


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    return _make_synthetic_daily_data()


@pytest.fixture
def synthetic_df_small() -> pd.DataFrame:
    return _make_synthetic_daily_data(n_days=120, seed=1)
