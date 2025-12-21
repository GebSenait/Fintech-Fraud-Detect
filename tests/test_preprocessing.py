"""
Unit tests for preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.preprocessing import (
    clean_fraud_data,
    ip_to_integer,
    add_geolocation_features,
    engineer_time_features,
    engineer_velocity_features,
    engineer_frequency_features,
)


class TestCleanFraudData:
    """Test cases for clean_fraud_data function."""

    def test_clean_fraud_data_removes_duplicates(self):
        """Test that duplicates are removed."""
        df = pd.DataFrame(
            {
                "user_id": [1, 2, 2, 3],
                "amount": [100, 200, 200, 300],
                "class": [0, 1, 1, 0],
            }
        )

        df_clean = clean_fraud_data(df)
        assert len(df_clean) == 3
        assert df_clean["user_id"].nunique() == 3

    def test_clean_fraud_data_handles_missing_numeric(self):
        """Test that missing numeric values are filled with median."""
        df = pd.DataFrame(
            {
                "amount": [100, 200, None, 400, 500],
                "class": [0, 1, 0, 1, 0],
            }
        )

        df_clean = clean_fraud_data(df)
        assert df_clean["amount"].isnull().sum() == 0
        assert df_clean["amount"].median() == 300.0

    def test_clean_fraud_data_handles_missing_categorical(self):
        """Test that missing categorical values are filled with mode."""
        df = pd.DataFrame(
            {
                "device": ["mobile", "desktop", None, "mobile", "tablet"],
                "class": [0, 1, 0, 1, 0],
            }
        )

        df_clean = clean_fraud_data(df)
        assert df_clean["device"].isnull().sum() == 0
        assert (df_clean["device"] == "mobile").sum() >= 2

    def test_clean_fraud_data_no_changes_when_clean(self):
        """Test that clean data remains unchanged."""
        df = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "amount": [100, 200, 300],
                "class": [0, 1, 0],
            }
        )

        df_clean = clean_fraud_data(df)
        assert len(df_clean) == len(df)
        assert df_clean.equals(df)


class TestIPToInteger:
    """Test cases for ip_to_integer function."""

    def test_ip_to_integer_valid(self):
        """Test conversion of valid IP address."""
        result = ip_to_integer("192.168.1.1")
        expected = 192 * 256**3 + 168 * 256**2 + 1 * 256 + 1
        assert result == expected

    def test_ip_to_integer_invalid_format(self):
        """Test that invalid IP format returns None."""
        assert ip_to_integer("192.168.1") is None
        assert ip_to_integer("192.168.1.1.1") is None
        assert ip_to_integer("invalid") is None

    def test_ip_to_integer_none_input(self):
        """Test that None input returns None."""
        assert ip_to_integer(None) is None


class TestAddGeolocationFeatures:
    """Test cases for add_geolocation_features function."""

    def test_add_geolocation_features_numeric_ip(self):
        """Test geolocation features with numeric IP addresses."""
        df = pd.DataFrame(
            {
                "ip_address": [16777216.0, 33554432.0, 999999999.0],
                "class": [0, 1, 0],
            }
        )

        ip_country_df = pd.DataFrame(
            {
                "lower_bound_ip_address": [16777216.0, 33554432.0],
                "upper_bound_ip_address": [16777471.0, 33554687.0],
                "country": ["Australia", "China"],
            }
        )

        result = add_geolocation_features(df, ip_country_df)

        assert "country" in result.columns
        assert result["country"].iloc[0] == "Australia"
        assert result["country"].iloc[1] == "China"
        assert result["country"].iloc[2] == "Unknown"

    def test_add_geolocation_features_string_ip(self):
        """Test geolocation features with string IP addresses."""
        df = pd.DataFrame(
            {
                "ip_address": ["1.0.0.1", "2.0.0.1", "192.168.1.1"],
                "class": [0, 1, 0],
            }
        )

        ip_country_df = pd.DataFrame(
            {
                "lower_bound_ip_address": [16777216.0, 33554432.0],
                "upper_bound_ip_address": [16777471.0, 33554687.0],
                "country": ["Test1", "Test2"],
            }
        )

        result = add_geolocation_features(df, ip_country_df)
        assert "country" in result.columns
        assert len(result) == 3


class TestEngineerTimeFeatures:
    """Test cases for engineer_time_features function."""

    def test_engineer_time_features_basic(self):
        """Test basic time feature engineering."""
        base_time = datetime(2023, 1, 1, 10, 0, 0)
        df = pd.DataFrame(
            {
                "purchase_time": [
                    base_time,
                    base_time + timedelta(hours=1),
                    base_time + timedelta(days=1),
                ],
                "signup_time": [
                    base_time - timedelta(days=1),
                    base_time - timedelta(hours=1),
                    base_time - timedelta(days=2),
                ],
                "class": [0, 1, 0],
            }
        )

        result = engineer_time_features(df)

        assert "hour_of_day" in result.columns
        assert "day_of_week" in result.columns
        assert "is_weekend" in result.columns
        assert "is_night" in result.columns
        assert "time_since_signup" in result.columns
        assert "is_new_account" in result.columns

    def test_engineer_time_features_weekend_detection(self):
        """Test weekend detection."""
        # Sunday (weekend)
        sunday = datetime(2023, 1, 1, 10, 0, 0)  # Jan 1, 2023 is a Sunday
        # Monday (weekday)
        monday = datetime(2023, 1, 2, 10, 0, 0)  # Jan 2, 2023 is a Monday

        df = pd.DataFrame(
            {
                "purchase_time": [sunday, monday],
                "signup_time": [sunday - timedelta(days=1), monday - timedelta(days=1)],
                "class": [0, 1],
            }
        )

        result = engineer_time_features(df)
        assert result["is_weekend"].iloc[0] == 1  # Sunday is weekend
        assert result["is_weekend"].iloc[1] == 0  # Monday is weekday


class TestEngineerVelocityFeatures:
    """Test cases for engineer_velocity_features function."""

    def test_engineer_velocity_features_basic(self):
        """Test basic velocity feature engineering."""
        base_time = datetime(2023, 1, 1, 10, 0, 0)
        df = pd.DataFrame(
            {
                "user_id": [1, 1, 1, 2, 2],
                "purchase_time": [
                    base_time,
                    base_time + timedelta(minutes=5),
                    base_time + timedelta(minutes=10),
                    base_time,
                    base_time + timedelta(hours=1),
                ],
                "class": [0, 1, 0, 1, 0],
            }
        )

        result = engineer_velocity_features(df)

        assert "transactions_last_1h" in result.columns
        assert "transactions_last_24h" in result.columns
        assert result["transactions_last_1h"].dtype in [np.int64, int]

    def test_engineer_velocity_features_single_user(self):
        """Test velocity features for single user."""
        base_time = datetime(2023, 1, 1, 10, 0, 0)
        df = pd.DataFrame(
            {
                "user_id": [1, 1, 1],
                "purchase_time": [
                    base_time,
                    base_time + timedelta(minutes=30),
                    base_time + timedelta(hours=2),
                ],
                "class": [0, 1, 0],
            }
        )

        result = engineer_velocity_features(df)
        assert len(result) == 3
        # First transaction has 0 previous transactions
        assert result["transactions_last_1h"].iloc[0] == 0
        # Second transaction has 1 previous transaction within 1h
        assert result["transactions_last_1h"].iloc[1] >= 1


class TestEngineerFrequencyFeatures:
    """Test cases for engineer_frequency_features function."""

    def test_engineer_frequency_features_basic(self):
        """Test basic frequency feature engineering."""
        df = pd.DataFrame(
            {
                "user_id": [1, 1, 2, 2, 2],
                "device_id": ["d1", "d1", "d2", "d2", "d3"],
                "class": [0, 1, 0, 1, 0],
            }
        )

        result = engineer_frequency_features(df)

        assert "user_transaction_count" in result.columns
        assert "device_usage_count" in result.columns
        assert result["user_transaction_count"].iloc[0] == 2
        assert result["user_transaction_count"].iloc[2] == 3

    def test_engineer_frequency_features_device_sharing(self):
        """Test device sharing detection."""
        df = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "device_id": ["d1", "d1", "d2"],
                "class": [0, 1, 0],
            }
        )

        result = engineer_frequency_features(df)
        assert "is_shared_device" in result.columns
        assert result["device_usage_count"].iloc[0] == 2  # d1 used 2 times
