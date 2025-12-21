"""
Unit tests for data_loader module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, mock_open
import tempfile
import os

from src.data_loader import (
    load_fraud_data,
    load_creditcard_data,
    load_ip_country_mapping,
    validate_data_quality,
)


class TestLoadFraudData:
    """Test cases for load_fraud_data function."""

    def test_load_fraud_data_success(self):
        """Test successful loading of fraud data."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(
                "user_id,signup_time,purchase_time,ip_address,device_id,class\n"
                "1,2023-01-01 10:00:00,2023-01-01 11:00:00,192.168.1.1,device1,0\n"
                "2,2023-01-02 10:00:00,2023-01-02 11:00:00,192.168.1.2,device2,1\n"
            )
            temp_path = f.name

        try:
            df = load_fraud_data(temp_path)

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "class" in df.columns
            assert df["class"].dtype == int
            assert "signup_time" in df.columns
            assert "purchase_time" in df.columns
        finally:
            os.unlink(temp_path)

    def test_load_fraud_data_file_not_found(self):
        """Test FileNotFoundError when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_fraud_data("nonexistent_file.csv")

    def test_load_fraud_data_datetime_conversion(self):
        """Test that datetime columns are properly converted."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(
                "user_id,signup_time,purchase_time,class\n"
                "1,2023-01-01 10:00:00,2023-01-01 11:00:00,0\n"
            )
            temp_path = f.name

        try:
            df = load_fraud_data(temp_path)
            assert pd.api.types.is_datetime64_any_dtype(df["signup_time"])
            assert pd.api.types.is_datetime64_any_dtype(df["purchase_time"])
        finally:
            os.unlink(temp_path)


class TestLoadCreditcardData:
    """Test cases for load_creditcard_data function."""

    def test_load_creditcard_data_success(self):
        """Test successful loading of credit card data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("Time,V1,V2,Amount,Class\n")
            f.write("0,1.0,2.0,100.0,0\n")
            f.write("1,1.5,2.5,200.0,1\n")
            temp_path = f.name

        try:
            df = load_creditcard_data(temp_path)

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "Class" in df.columns
            assert df["Class"].dtype == int
        finally:
            os.unlink(temp_path)

    def test_load_creditcard_data_file_not_found(self):
        """Test FileNotFoundError when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_creditcard_data("nonexistent_file.csv")


class TestLoadIPCountryMapping:
    """Test cases for load_ip_country_mapping function."""

    def test_load_ip_country_mapping_success(self):
        """Test successful loading of IP country mapping."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(
                "lower_bound_ip_address,upper_bound_ip_address,country\n"
                "16777216.0,16777471.0,Australia\n"
                "16777472.0,16777727.0,China\n"
            )
            temp_path = f.name

        try:
            df = load_ip_country_mapping(temp_path)

            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2
            assert "lower_bound_ip_address" in df.columns
            assert "upper_bound_ip_address" in df.columns
            assert "country" in df.columns
            assert pd.api.types.is_numeric_dtype(df["lower_bound_ip_address"])
            assert pd.api.types.is_numeric_dtype(df["upper_bound_ip_address"])
        finally:
            os.unlink(temp_path)

    def test_load_ip_country_mapping_file_not_found(self):
        """Test FileNotFoundError when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_ip_country_mapping("nonexistent_file.csv")


class TestValidateDataQuality:
    """Test cases for validate_data_quality function."""

    def test_validate_data_quality_basic(self):
        """Test basic data quality validation."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 3, None, 5],
                "col2": ["a", "b", "c", "d", "e"],
                "col3": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )

        report = validate_data_quality(df, "test_dataset")

        assert isinstance(report, dict)
        assert report["dataset"] == "test_dataset"
        assert report["shape"] == (5, 3)
        assert "missing_values" in report
        assert "missing_percentage" in report
        assert "duplicates" in report
        assert "dtypes" in report

    def test_validate_data_quality_with_duplicates(self):
        """Test data quality validation with duplicates."""
        df = pd.DataFrame(
            {
                "col1": [1, 2, 2, 3],
                "col2": ["a", "b", "b", "c"],
            }
        )

        report = validate_data_quality(df, "test_dataset")
        assert report["duplicates"] == 1

    def test_validate_data_quality_empty_dataframe(self):
        """Test data quality validation with empty dataframe."""
        df = pd.DataFrame()

        report = validate_data_quality(df, "empty_dataset")
        assert report["shape"] == (0, 0)
        assert report["duplicates"] == 0
