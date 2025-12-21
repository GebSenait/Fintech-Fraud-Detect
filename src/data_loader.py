"""
Data loading utilities for fraud detection datasets.

This module provides functions to load and validate fraud detection datasets
with proper error handling and data type conversion.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


def load_fraud_data(data_path: str = "data/raw/Fraud_Data.csv") -> pd.DataFrame:
    """
    Load e-commerce fraud transaction data.
    
    Args:
        data_path: Path to Fraud_Data.csv file
        
    Returns:
        DataFrame with properly typed columns
        
    Raises:
        FileNotFoundError: If data file doesn't exist
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Fraud data file not found at {data_path}")
    
    df = pd.read_csv(path)
    
    # Convert timestamp columns
    if 'signup_time' in df.columns:
        df['signup_time'] = pd.to_datetime(df['signup_time'], errors='coerce')
    if 'purchase_time' in df.columns:
        df['purchase_time'] = pd.to_datetime(df['purchase_time'], errors='coerce')
    
    # Ensure class column is integer
    if 'class' in df.columns:
        df['class'] = df['class'].astype(int)
    
    return df


def load_creditcard_data(data_path: str = "data/raw/creditcard.csv") -> pd.DataFrame:
    """
    Load bank credit card transaction data.
    
    Args:
        data_path: Path to creditcard.csv file
        
    Returns:
        DataFrame with properly typed columns
        
    Raises:
        FileNotFoundError: If data file doesn't exist
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Credit card data file not found at {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Ensure Class column is integer
    if 'Class' in df.columns:
        df['Class'] = df['Class'].astype(int)
    
    return df


def load_ip_country_mapping(data_path: str = "data/raw/IpAddress_to_Country.csv") -> pd.DataFrame:
    """
    Load IP address to country mapping data.
    
    Args:
        data_path: Path to IpAddress_to_Country.csv file
        
    Returns:
        DataFrame with IP ranges and countries
        
    Raises:
        FileNotFoundError: If data file doesn't exist
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"IP country mapping file not found at {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Convert IP addresses to numeric for range matching
    if 'lower_bound_ip_address' in df.columns:
        df['lower_bound_ip_address'] = pd.to_numeric(df['lower_bound_ip_address'], errors='coerce')
    if 'upper_bound_ip_address' in df.columns:
        df['upper_bound_ip_address'] = pd.to_numeric(df['upper_bound_ip_address'], errors='coerce')
    
    return df


def validate_data_quality(df: pd.DataFrame, dataset_name: str) -> dict:
    """
    Validate data quality and return summary statistics.
    
    Args:
        df: DataFrame to validate
        dataset_name: Name of the dataset for reporting
        
    Returns:
        Dictionary with quality metrics
    """
    quality_report = {
        'dataset': dataset_name,
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicates': df.duplicated().sum(),
        'dtypes': df.dtypes.to_dict()
    }
    
    return quality_report

