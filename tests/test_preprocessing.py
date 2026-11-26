"""
Tests for Data Preprocessing, Feature Engineering, and Splitting
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing import handle_missing_values
from feature_engineering import create_features
from split_data import time_series_split


@pytest.fixture
def sample_raw_dataframe():
    """Generates synthetic hourly air quality data with known missing values."""
    idx = pd.date_range("2024-01-01 00:00:00", periods=72, freq="h")
    df = pd.DataFrame(
        {
            "No": range(1, 73),
            "year": idx.year,
            "month": idx.month,
            "day": idx.day,
            "hour": idx.hour,
            "PM2.5": np.linspace(20, 80, 72),
            "PM10": np.linspace(30, 120, 72),
            "SO2": np.random.uniform(2, 20, 72),
            "NO2": np.random.uniform(10, 50, 72),
            "CO": np.random.uniform(200, 1500, 72),
            "O3": np.random.uniform(10, 100, 72),
            "TEMP": np.random.uniform(5, 25, 72),
            "PRES": np.random.uniform(1005, 1025, 72),
            "DEWP": np.random.uniform(-5, 15, 72),
            "RAIN": np.zeros(72),
            "wd": ["N", "NE", "E", "SE", "S", "SW", "W", "NW"] * 9,
            "WSPM": np.random.uniform(0.5, 5.0, 72),
            "station": ["Dongsi"] * 72,
        },
        index=idx,
    )
    return df


class TestPreprocessing:
    def test_handle_missing_values_interpolates_short_gaps(self, sample_raw_dataframe):
        """Short missing gaps (<= 6 hours) must be linearly interpolated."""
        df = sample_raw_dataframe.copy()
        # Create a 3-hour gap
        df.iloc[10:13, df.columns.get_loc("PM2.5")] = np.nan

        cleaned = handle_missing_values(df, target_col="PM2.5", max_gap_hours=6)

        assert len(cleaned) == len(df)
        assert not cleaned["PM2.5"].isnull().any()
        # Verify linear interpolation between row 9 and 13
        expected_val = (sample_raw_dataframe["PM2.5"].iloc[9] + sample_raw_dataframe["PM2.5"].iloc[13]) / 2
        assert np.isclose(cleaned["PM2.5"].iloc[11], expected_val, atol=1e-3)

    def test_handle_missing_values_drops_long_target_gaps(self, sample_raw_dataframe):
        """Missing target gaps > max_gap_hours should not be interpolated and must be dropped."""
        df = sample_raw_dataframe.copy()
        # Create an 8-hour gap in target
        df.iloc[20:28, df.columns.get_loc("PM2.5")] = np.nan

        cleaned = handle_missing_values(df, target_col="PM2.5", max_gap_hours=6)

        # Rows that remain NaN in target must be dropped
        assert not cleaned["PM2.5"].isnull().any()
        assert len(cleaned) < len(df)


class TestFeatureEngineering:
    def test_create_features_output_structure(self, sample_raw_dataframe):
        """Verifies feature creation generates expected columns and strips raw metadata."""
        df = sample_raw_dataframe.copy()
        features_df = create_features(df, target_col="PM2.5")

        # Dropped columns
        for col in ["No", "year", "month", "day", "hour", "wd", "station"]:
            assert col not in features_df.columns

        # Lag features
        for lag in [1, 2, 3, 6, 12, 24]:
            assert f"PM2.5_lag_{lag}h" in features_df.columns

        # Rolling features
        for window in [6, 12, 24]:
            assert f"PM2.5_roll_mean_{window}h" in features_df.columns
            assert f"PM2.5_roll_std_{window}h" in features_df.columns

        # Cyclical time & wind
        for col in ["hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_week", "wind_u", "wind_v"]:
            assert col in features_df.columns

        # No NaN remaining
        assert not features_df.isnull().any().any()

    def test_no_target_leakage(self, sample_raw_dataframe):
        """Ensures that no feature directly encodes the target PM2.5 at time t."""
        df = sample_raw_dataframe.copy()
        features_df = create_features(df, target_col="PM2.5")

        # PM2.5_change_24h must NOT exist
        assert "PM2.5_change_24h" not in features_df.columns

        # Rolling mean must only use past data (shift 1 hour)
        past_pm25 = df["PM2.5"].shift(1)
        expected_roll6 = past_pm25.rolling(6, min_periods=6).mean().reindex(features_df.index)
        actual_roll6 = features_df["PM2.5_roll_mean_6h"]
        assert np.allclose(expected_roll6.to_numpy(), actual_roll6.to_numpy(), atol=1e-7)

    def test_cyclical_encoding_bounds(self, sample_raw_dataframe):
        """sin and cos encodings must stay strictly within [-1.0, 1.0]."""
        df = sample_raw_dataframe.copy()
        features_df = create_features(df, target_col="PM2.5")

        for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
            assert features_df[col].min() >= -1.0 - 1e-6
            assert features_df[col].max() <= 1.0 + 1e-6


class TestDataSplit:
    def test_time_series_split_chronological(self, sample_raw_dataframe):
        """Train, Val, Test sets must strictly preserve temporal ordering without shuffle."""
        df = sample_raw_dataframe.copy()
        features_df = create_features(df, target_col="PM2.5")

        X_train, X_val, X_test, y_train, y_val, y_test, scaler = time_series_split(
            features_df, target_col="PM2.5", train_ratio=0.7, val_ratio=0.15
        )

        # Chronological timestamps
        assert X_train.index.max() < X_val.index.min()
        assert X_val.index.max() < X_test.index.min()

        # Proportions
        n_total = len(features_df)
        assert len(X_train) == int(n_total * 0.70)
        assert len(X_val) == int(n_total * 0.85) - int(n_total * 0.70)

        # Scaler must have mean and std
        assert len(scaler.mean_) == X_train.shape[1]
        # Train features scaled to roughly zero mean
        assert np.allclose(X_train.mean(axis=0), 0, atol=0.2)
