"""
test_data_processing.py
Automated tests for data_processing.py utilities.
Run with: pytest tests/
"""

import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from data_processing import optimize_memory, handle_missing_values, handle_outliers, get_feature_target


# ── Fixtures ────────────────────────────────────────────────────────────────────
@pytest.fixture
def sample_df():
    """Minimal heart-failure-like DataFrame for testing."""
    return pd.DataFrame({
        "age":                      [60.0, 45.0, 75.0, 55.0, 80.0],
        "ejection_fraction":        [38,   55,   25,   40,   30],
        "serum_creatinine":         [1.1,  0.9,  2.5,  1.0,  3.0],
        "serum_sodium":             [137,  140,  125,  135,  130],
        "creatinine_phosphokinase": [250,  100,  7000, 300,  500],
        "platelets":                [265000.0, 200000.0, 300000.0, 250000.0, 400000.0],
        "time":                     [90,   120,  30,   60,   45],
        "anaemia":                  [0,    1,    0,    1,    0],
        "diabetes":                 [0,    0,    1,    0,    1],
        "high_blood_pressure":      [1,    0,    1,    0,    1],
        "sex":                      [1,    0,    1,    1,    0],
        "smoking":                  [0,    1,    0,    0,    1],
        "DEATH_EVENT":              [0,    0,    1,    0,    1],
    })


# ── optimize_memory tests ────────────────────────────────────────────────────────
def test_optimize_memory_reduces_size(sample_df):
    before = sample_df.memory_usage(deep=True).sum()
    optimized = optimize_memory(sample_df.copy())
    after = optimized.memory_usage(deep=True).sum()
    assert after <= before, "Memory should not increase after optimization."


def test_optimize_memory_float64_to_float32(sample_df):
    df = sample_df.copy()
    # Ensure float64 columns present
    df["age"] = df["age"].astype("float64")
    optimized = optimize_memory(df)
    assert optimized["age"].dtype == np.float32, "float64 should be cast to float32."


def test_optimize_memory_int64_to_int32(sample_df):
    df = sample_df.copy()
    df["time"] = df["time"].astype("int64")
    optimized = optimize_memory(df)
    assert optimized["time"].dtype == np.int32, "int64 should be cast to int32."


def test_optimize_memory_preserves_values(sample_df):
    original_ages = sample_df["age"].values.tolist()
    optimized = optimize_memory(sample_df.copy())
    assert [round(v, 1) for v in optimized["age"].values.tolist()] == \
           [round(v, 1) for v in original_ages], "Values should be preserved after optimization."


# ── handle_missing_values tests ─────────────────────────────────────────────────
def test_no_missing_after_imputation(sample_df):
    df = sample_df.copy()
    df.loc[0, "age"] = np.nan
    df.loc[2, "serum_creatinine"] = np.nan
    result = handle_missing_values(df)
    assert result.isnull().sum().sum() == 0, "No missing values should remain after imputation."


def test_imputation_uses_median(sample_df):
    df = sample_df.copy()
    original_median = df["ejection_fraction"].median()
    df.loc[0, "ejection_fraction"] = np.nan
    result = handle_missing_values(df)
    assert result.loc[0, "ejection_fraction"] == pytest.approx(original_median, abs=1), \
        "Numeric imputation should use the median."


# ── handle_outliers tests ────────────────────────────────────────────────────────
def test_outlier_clip_removes_extremes(sample_df):
    df = sample_df.copy()
    df.loc[2, "creatinine_phosphokinase"] = 999999  # extreme outlier
    result = handle_outliers(df, ["creatinine_phosphokinase"], method="clip")
    Q3 = df["creatinine_phosphokinase"].quantile(0.75)
    IQR = df["creatinine_phosphokinase"].quantile(0.75) - df["creatinine_phosphokinase"].quantile(0.25)
    upper = Q3 + 1.5 * IQR
    assert result["creatinine_phosphokinase"].max() <= upper, "Clipping should bound the maximum."


# ── get_feature_target tests ─────────────────────────────────────────────────────
def test_get_feature_target_splits_correctly(sample_df):
    X, y = get_feature_target(sample_df)
    assert "DEATH_EVENT" not in X.columns, "DEATH_EVENT should not be in features."
    assert len(y) == len(sample_df), "Target length must match DataFrame length."
    assert set(y.unique()).issubset({0, 1}), "Target must be binary."


def test_feature_count(sample_df):
    X, y = get_feature_target(sample_df)
    assert X.shape[1] == sample_df.shape[1] - 1, "Features should exclude only the target column."
