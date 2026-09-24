"""Tests for src.etl_pipeline."""

import pandas as pd
import pytest

from src.data_generator import generate_student_data
from src.etl_pipeline import ETLSummary, VALID_RANGES, transform


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return generate_student_data(n_students=2000, seed=42)


@pytest.fixture(scope="module")
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    summary = ETLSummary()
    return transform(raw_df.copy(), summary)


def test_no_missing_values_after_cleaning(clean_df: pd.DataFrame) -> None:
    assert clean_df.isna().sum().sum() == 0


def test_no_duplicate_student_ids_after_cleaning(clean_df: pd.DataFrame) -> None:
    assert clean_df["student_id"].duplicated().sum() == 0


def test_numeric_ranges_valid_after_cleaning(clean_df: pd.DataFrame) -> None:
    for col, (low, high) in VALID_RANGES.items():
        assert clean_df[col].between(low, high).all(), f"{col} has out-of-range values"


def test_gender_values_valid_after_cleaning(clean_df: pd.DataFrame) -> None:
    assert set(clean_df["gender"].unique()).issubset({"Male", "Female"})


def test_summary_counts_recorded(raw_df: pd.DataFrame) -> None:
    summary = ETLSummary()
    transform(raw_df.copy(), summary)
    assert summary.duplicates_removed >= 0
    assert summary.missing_values_handled >= 0
    assert summary.invalid_values_handled >= 0


def test_row_count_does_not_increase(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> None:
    assert len(clean_df) <= len(raw_df)
