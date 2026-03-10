"""
data_processing.py
Core data loading, cleaning, and memory-optimization utilities.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    """Load the Heart Failure dataset from a CSV file."""
    df = pd.read_csv(filepath)
    return df


def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Systematically reduce DataFrame memory usage by down-casting
    numeric dtypes (float64 → float32, int64 → int32).

    Returns the optimized DataFrame and prints memory stats.
    """
    before = df.memory_usage(deep=True).sum() / 1024

    for col in df.columns:
        col_type = df[col].dtype

        if col_type == "float64":
            df[col] = df[col].astype("float32")
        elif col_type == "int64":
            # Use int32 if values fit; otherwise keep int64
            if df[col].min() >= np.iinfo(np.int32).min and df[col].max() <= np.iinfo(np.int32).max:
                df[col] = df[col].astype("int32")
        elif col_type == "object":
            # Convert low-cardinality string columns to category
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype("category")

    after = df.memory_usage(deep=True).sum() / 1024
    print(f"Memory optimized: {before:.1f} KB → {after:.1f} KB "
          f"(saved {before - after:.1f} KB, {(1 - after/before)*100:.1f}%)")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values:
    - Numeric columns → median imputation.
    - Categorical columns → mode imputation.
    """
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype in ["float64", "float32", "int64", "int32"]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    return df


def handle_outliers(df: pd.DataFrame, cols: list, method: str = "clip") -> pd.DataFrame:
    """
    Handle outliers in specified columns using IQR-based clipping or removal.

    Args:
        df: Input dataframe.
        cols: Columns to process.
        method: 'clip' (default) or 'remove'.
    """
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        if method == "clip":
            df[col] = df[col].clip(lower, upper)
        elif method == "remove":
            df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df


def get_feature_target(df: pd.DataFrame):
    """Split DataFrame into features X and target y."""
    target = "DEATH_EVENT"
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """Standard-scale continuous features. Returns scaled arrays and the scaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler
