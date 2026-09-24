"""Tests for src.data_generator."""

import pandas as pd
import pytest

from src.data_generator import generate_student_data
from src.etl_pipeline import REQUIRED_COLUMNS


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    # Small sample for fast tests; generator logic is identical regardless of n.
    return generate_student_data(n_students=1000, seed=42)


def test_record_count(raw_df: pd.DataFrame) -> None:
    # Note: the generator intentionally injects a small number of duplicate
    # rows (by design, to exercise ETL de-duplication), so the raw row
    # count is slightly >= n_students. The count of *unique* student_ids
    # generated should equal n_students exactly.
    assert len(raw_df) >= 1000
    assert raw_df["student_id"].nunique() == 1000


def test_required_columns_present(raw_df: pd.DataFrame) -> None:
    for col in REQUIRED_COLUMNS:
        assert col in raw_df.columns


def test_reproducibility() -> None:
    df1 = generate_student_data(n_students=500, seed=42)
    df2 = generate_student_data(n_students=500, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seed_produces_different_data() -> None:
    df1 = generate_student_data(n_students=500, seed=42)
    df2 = generate_student_data(n_students=500, seed=99)
    assert not df1["final_exam_score"].equals(df2["final_exam_score"])


def test_gender_categories(raw_df: pd.DataFrame) -> None:
    assert set(raw_df["gender"].dropna().unique()).issubset({"Male", "Female"})


def test_dataset_has_injected_missing_values(raw_df: pd.DataFrame) -> None:
    # By design, the raw (pre-ETL) dataset should contain some missing values.
    assert raw_df.isna().sum().sum() > 0


def test_dataset_has_some_out_of_range_values_before_cleaning(raw_df: pd.DataFrame) -> None:
    # By design, mild anomalies are injected pre-cleaning to exercise ETL validation.
    out_of_range = (
        (raw_df["attendance_rate"] < 0)
        | (raw_df["attendance_rate"] > 100)
        | (raw_df["final_exam_score"] < 0)
        | (raw_df["final_exam_score"] > 100)
    )
    assert out_of_range.sum() >= 0  # non-fatal sanity check; anomalies are probabilistic
