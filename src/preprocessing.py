"""
Data preprocessing utilities for fraud detection.

This module provides functions for cleaning, transforming, and preparing
fraud detection datasets for modeling.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


def clean_fraud_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean e-commerce fraud data.

    Business Logic:
    - Remove duplicates (same transaction = data quality issue)
    - Handle missing values based on business rules
    - Correct data types for downstream processing

    Args:
        df: Raw fraud data DataFrame

    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()

    # Remove duplicates
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    duplicates_removed = initial_count - len(df_clean)

    if duplicates_removed > 0:
        print(
            f"Removed {duplicates_removed} duplicate records ({duplicates_removed/initial_count*100:.2f}%)"
        )

    # Handle missing values - business-driven decisions
    # For critical fraud indicators, we'll use mode/median imputation
    # For non-critical, we may drop or create "unknown" category

    # Numeric columns - use median (robust to outliers)
    # OPTIMIZED: Use vectorized operations instead of loops
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    numeric_cols = [
        col for col in numeric_cols if col != "class"
    ]  # Don't impute target

    if len(numeric_cols) > 0:
        # Get columns with missing values
        missing_numeric = df_clean[numeric_cols].isnull().any()
        if missing_numeric.any():
            # Fill all numeric columns at once using median
            medians = df_clean[numeric_cols].median()
            df_clean[numeric_cols] = df_clean[numeric_cols].fillna(medians)

    # Categorical columns - use mode or "unknown"
    # OPTIMIZED: Calculate mode once per column instead of twice
    categorical_cols = df_clean.select_dtypes(include=["object"]).columns
    if len(categorical_cols) > 0:
        # Get columns with missing values
        missing_categorical = df_clean[categorical_cols].isnull().any()
        if missing_categorical.any():
            for col in categorical_cols[missing_categorical]:
                mode_result = df_clean[col].mode()
                mode_value = mode_result[0] if len(mode_result) > 0 else "unknown"
                df_clean[col].fillna(mode_value, inplace=True)

    return df_clean


def ip_to_integer(ip_address: str) -> Optional[int]:
    """
    Convert IP address string to integer for range matching.

    Args:
        ip_address: IP address as string (e.g., "192.168.1.1")

    Returns:
        Integer representation of IP address
    """
    try:
        parts = ip_address.split(".")
        if len(parts) != 4:
            return None
        return (
            int(parts[0]) * 256**3
            + int(parts[1]) * 256**2
            + int(parts[2]) * 256
            + int(parts[3])
        )
    except (ValueError, AttributeError):
        return None


def add_geolocation_features(
    df: pd.DataFrame, ip_country_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add country information based on IP address ranges.

    Business Value:
    - High-risk countries are strong fraud indicators
    - Geolocation mismatches (user location vs transaction location) indicate fraud

    Args:
        df: Transaction DataFrame with ip_address column
        ip_country_df: DataFrame with IP ranges and countries

    Returns:
        DataFrame with added 'country' column
    """
    df_geo = df.copy()

    # OPTIMIZED: Vectorized IP conversion using string operations
    # Handle both string IPs (e.g., "1.0.0.1") and numeric IPs (already integers)
    if pd.api.types.is_numeric_dtype(df_geo["ip_address"]):
        # If already numeric, use directly as float64 (needed for merge_asof)
        df_geo["ip_integer"] = df_geo["ip_address"].astype(float)
    else:
        # Ensure ip_address is string type before using .str accessor
        df_geo["ip_address"] = df_geo["ip_address"].astype(str)
        # Convert IP addresses to integers using vectorized operations
        ip_parts = df_geo["ip_address"].str.split(".", expand=True).astype(float)
        df_geo["ip_integer"] = (
            ip_parts[0] * 256**3
            + ip_parts[1] * 256**2
            + ip_parts[2] * 256
            + ip_parts[3]
        ).astype(float)

    # Initialize country column
    df_geo["country"] = "Unknown"

    # OPTIMIZED: Use merge_asof for fast range matching (much faster than IntervalIndex)
    # Sort IP ranges by lower bound
    ip_country_sorted = ip_country_df.sort_values("lower_bound_ip_address").reset_index(
        drop=True
    )

    # Filter valid IPs
    valid_mask = df_geo["ip_integer"].notna()

    if valid_mask.sum() > 0:
        # Create a temporary dataframe with IPs to match (ensure float dtype for merge)
        ip_lookup = pd.DataFrame(
            {"ip_integer": df_geo.loc[valid_mask, "ip_integer"].astype(float).values}
        ).sort_values("ip_integer")

        # Create lookup dataframe with lower bounds (already float)
        ip_ranges = pd.DataFrame(
            {
                "lower_bound_ip_address": ip_country_sorted[
                    "lower_bound_ip_address"
                ].values,
                "upper_bound_ip_address": ip_country_sorted[
                    "upper_bound_ip_address"
                ].values,
                "country": ip_country_sorted["country"].values,
            }
        )

        # Use merge_asof to find the last range where lower_bound <= ip_integer
        # This is optimized for sorted data and much faster than IntervalIndex
        merged = pd.merge_asof(
            ip_lookup,
            ip_ranges,
            left_on="ip_integer",
            right_on="lower_bound_ip_address",
            direction="backward",
            allow_exact_matches=True,
        )

        # Filter to only keep matches where IP is within range
        in_range = (merged["ip_integer"] >= merged["lower_bound_ip_address"]) & (
            merged["ip_integer"] <= merged["upper_bound_ip_address"]
        )

        # Create mapping dictionary for fast assignment
        ip_to_country = dict(
            zip(merged.loc[in_range, "ip_integer"], merged.loc[in_range, "country"])
        )

        # Assign countries using map (very fast)
        df_geo.loc[valid_mask, "country"] = (
            df_geo.loc[valid_mask, "ip_integer"]
            .astype(float)
            .map(ip_to_country)
            .fillna("Unknown")
        )

    # Drop intermediate column
    df_geo.drop("ip_integer", axis=1, inplace=True)

    return df_geo


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer time-based features for fraud detection.

    Business Logic:
    - Fraud often occurs at unusual hours
    - Time since signup indicates account maturity (new accounts = higher risk)
    - Day of week patterns reveal fraud behavior

    Args:
        df: DataFrame with purchase_time and signup_time columns

    Returns:
        DataFrame with added time features
    """
    df_time = df.copy()

    if "purchase_time" in df_time.columns:
        df_time["hour_of_day"] = df_time["purchase_time"].dt.hour
        df_time["day_of_week"] = df_time["purchase_time"].dt.dayofweek
        df_time["day_of_month"] = df_time["purchase_time"].dt.day
        df_time["month"] = df_time["purchase_time"].dt.month
        df_time["is_weekend"] = (df_time["day_of_week"] >= 5).astype(int)
        df_time["is_night"] = (
            (df_time["hour_of_day"] >= 22) | (df_time["hour_of_day"] < 6)
        ).astype(int)

    if "signup_time" in df_time.columns and "purchase_time" in df_time.columns:
        df_time["time_since_signup"] = (
            df_time["purchase_time"] - df_time["signup_time"]
        ).dt.total_seconds() / 3600  # hours
        df_time["time_since_signup_days"] = df_time["time_since_signup"] / 24
        df_time["is_new_account"] = (df_time["time_since_signup"] < 24).astype(
            int
        )  # Less than 24 hours

    return df_time


def engineer_velocity_features(
    df: pd.DataFrame, time_window_hours: int = 24
) -> pd.DataFrame:
    """
    Engineer transaction velocity features.

    Business Logic:
    - High transaction frequency in short time = fraud indicator
    - Rapid successive transactions from same user/device = suspicious

    Args:
        df: Transaction DataFrame
        time_window_hours: Time window for velocity calculation

    Returns:
        DataFrame with velocity features
    """
    df_vel = df.copy()

    if "purchase_time" not in df_vel.columns:
        return df_vel

    # Sort by user and time
    df_vel = df_vel.sort_values(["user_id", "purchase_time"])

    # Transactions per user in time window
    df_vel["transactions_last_24h"] = 0
    df_vel["transactions_last_1h"] = 0

    for user_id in df_vel["user_id"].unique():
        user_mask = df_vel["user_id"] == user_id
        user_df = df_vel[user_mask].copy()

        for idx in user_df.index:
            current_time = user_df.loc[idx, "purchase_time"]

            # Count transactions in last 24 hours
            mask_24h = (
                user_df["purchase_time"] >= current_time - pd.Timedelta(hours=24)
            ) & (user_df["purchase_time"] < current_time)
            df_vel.loc[idx, "transactions_last_24h"] = mask_24h.sum()

            # Count transactions in last 1 hour
            mask_1h = (
                user_df["purchase_time"] >= current_time - pd.Timedelta(hours=1)
            ) & (user_df["purchase_time"] < current_time)
            df_vel.loc[idx, "transactions_last_1h"] = mask_1h.sum()

    # High velocity flags
    df_vel["high_velocity_24h"] = (df_vel["transactions_last_24h"] >= 5).astype(int)
    df_vel["high_velocity_1h"] = (df_vel["transactions_last_1h"] >= 3).astype(int)

    return df_vel


def engineer_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer transaction frequency features per user.

    Business Logic:
    - Users with many transactions may be legitimate power users or fraudsters
    - Need to combine with other features for context

    Args:
        df: Transaction DataFrame

    Returns:
        DataFrame with frequency features
    """
    df_freq = df.copy()

    # Total transactions per user
    user_counts = df_freq["user_id"].value_counts()
    df_freq["user_transaction_count"] = df_freq["user_id"].map(user_counts)

    # Device usage frequency
    if "device_id" in df_freq.columns:
        device_counts = df_freq["device_id"].value_counts()
        df_freq["device_usage_count"] = df_freq["device_id"].map(device_counts)
        df_freq["is_shared_device"] = (df_freq["device_usage_count"] > 10).astype(int)

    # Browser frequency
    if "browser" in df_freq.columns:
        browser_counts = df_freq["browser"].value_counts()
        df_freq["browser_usage_count"] = df_freq["browser"].map(browser_counts)

    return df_freq


def prepare_features_for_modeling(
    df: pd.DataFrame,
    target_col: str = "class",
    categorical_cols: Optional[List[str]] = None,
    numerical_cols: Optional[List[str]] = None,
    scaler_type: str = "standard",
) -> Tuple[pd.DataFrame, pd.Series, dict]:
    """
    Prepare features for machine learning modeling.

    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        categorical_cols: List of categorical column names (auto-detected if None)
        numerical_cols: List of numerical column names (auto-detected if None)
        scaler_type: 'standard' or 'minmax'

    Returns:
        Tuple of (feature_df, target_series, preprocessing_dict)
    """
    df_model = df.copy()

    # Separate target
    if target_col not in df_model.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    y = df_model[target_col].copy()
    X = df_model.drop(columns=[target_col])

    # Auto-detect column types if not provided
    if categorical_cols is None:
        categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    if numerical_cols is None:
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        # Remove any columns that should be categorical
        categorical_cols_set = set(categorical_cols)
        numerical_cols = [
            col for col in numerical_cols if col not in categorical_cols_set
        ]

    # One-hot encode categorical variables
    X_encoded = pd.get_dummies(
        X, columns=categorical_cols, prefix=categorical_cols, drop_first=True
    )

    # Scale numerical features
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler_type: {scaler_type}")

    X_encoded[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

    # Store preprocessing artifacts
    preprocessing_dict = {
        "scaler": scaler,
        "categorical_cols": categorical_cols,
        "numerical_cols": numerical_cols,
        "feature_names": X_encoded.columns.tolist(),
    }

    return X_encoded, y, preprocessing_dict


def handle_class_imbalance(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = "smote",
    sampling_strategy: float = 0.5,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Handle class imbalance in training data.

    Business Justification:
    - SMOTE: Creates synthetic minority samples, preserves all majority samples
      Best when we have sufficient data and want to minimize information loss
    - Undersampling: Reduces majority class, faster but loses information
      Best when dataset is very large and we need faster training

    Args:
        X_train: Training features
        y_train: Training target
        method: 'smote' or 'undersample'
        sampling_strategy: Target ratio of minority to majority (0.5 = 1:2 ratio)

    Returns:
        Resampled (X_train, y_train)
    """
    # Lazy import to avoid compatibility issues
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
    except ImportError as e:
        raise ImportError(
            f"imbalanced-learn is required for class imbalance handling. Install with: pip install imbalanced-learn. Error: {e}"
        )

    if method == "smote":
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(
            y_resampled
        )

    elif method == "undersample":
        undersampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy, random_state=42
        )
        X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
        return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(
            y_resampled
        )

    else:
        raise ValueError(f"Unknown method: {method}. Use 'smote' or 'undersample'")
