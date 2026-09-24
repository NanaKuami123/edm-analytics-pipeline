"""
etl_pipeline.py
================

Extract-Transform-Load pipeline for the synthetic student academic
dataset. Reads the raw CSV, validates and cleans it, and loads the
cleaned result into both a CSV file and a SQLite database.

Run directly:
    python -m src.etl_pipeline
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_academic_data.csv"
CLEAN_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "clean_student_academic_data.csv"
SQLITE_DB_PATH = PROJECT_ROOT / "data" / "processed" / "education_analytics.db"

REQUIRED_COLUMNS = [
    "student_id",
    "age",
    "gender",
    "attendance_rate",
    "study_hours_per_week",
    "assignment_score",
    "quiz_score",
    "midterm_score",
    "continuous_assessment",
    "previous_gpa",
    "final_exam_score",
]

# Documented validation ranges (inclusive) used for both cleaning and
# downstream tests.
VALID_RANGES = {
    "age": (18, 30),
    "attendance_rate": (0.0, 100.0),
    "study_hours_per_week": (0.0, 40.0),
    "assignment_score": (0.0, 100.0),
    "quiz_score": (0.0, 100.0),
    "midterm_score": (0.0, 100.0),
    "continuous_assessment": (0.0, 100.0),
    "previous_gpa": (0.0, 4.0),
    "final_exam_score": (0.0, 100.0),
}

@dataclass
class ETLSummary:
    """Tracks counts through the ETL run for reporting purposes."""

    raw_records: int = 0
    duplicates_removed: int = 0
    missing_values_handled: int = 0
    invalid_values_handled: int = 0
    final_records: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "raw_records": self.raw_records,
            "duplicates_removed": self.duplicates_removed,
            "missing_values_handled": self.missing_values_handled,
            "invalid_values_handled": self.invalid_values_handled,
            "final_records": self.final_records,
        }

    def print_report(self) -> None:
        print("\n" + "=" * 60)
        print("ETL PIPELINE — SUMMARY REPORT")
        print("=" * 60)
        print(f"Raw records read             : {self.raw_records:,}")
        print(f"Duplicate records removed    : {self.duplicates_removed:,}")
        print(f"Missing values handled       : {self.missing_values_handled:,}")
        print(f"Invalid/out-of-range handled : {self.invalid_values_handled:,}")
        print(f"Final clean records          : {self.final_records:,}")
        if self.notes:
            print("\nNotes:")
            for note in self.notes:
                print(f"  - {note}")
        print("=" * 60 + "\n")


def extract(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """EXTRACT: read the raw CSV dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Run `python -m src.data_generator` first."
        )
    df = pd.read_csv(path)
    logger.info("Extracted %d raw records from %s", len(df), path)
    return df


def _validate_columns(df: pd.DataFrame) -> None:
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")


def _remove_duplicates(df: pd.DataFrame, summary: ETLSummary) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset="student_id", keep="first").reset_index(drop=True)
    removed = before - len(df)
    summary.duplicates_removed += removed
    logger.info("Removed %d duplicate student_id records", removed)
    return df


def _handle_missing_values(df: pd.DataFrame, summary: ETLSummary) -> pd.DataFrame:
    """
    Missing-value strategy (documented):
      - Numeric columns: impute with the column median (robust to outliers).
      - Categorical columns (gender): impute with the column mode.
      - student_id: rows with a missing student_id are dropped (cannot
        be reliably identified/de-duplicated).
    """
    df = df.copy()

    missing_before = int(df.isna().sum().sum())

    # Drop rows with missing student_id (identity column).
    before = len(df)
    df = df.dropna(subset=["student_id"]).reset_index(drop=True)
    dropped_for_id = before - len(df)
    if dropped_for_id:
        summary.notes.append(f"Dropped {dropped_for_id} rows with missing student_id")

    numeric_cols = [c for c in REQUIRED_COLUMNS if c not in ("student_id", "gender")]
    for col in numeric_cols:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    if df["gender"].isna().any():
        mode_val = df["gender"].mode(dropna=True).iloc[0]
        df["gender"] = df["gender"].fillna(mode_val)

    missing_after = int(df.isna().sum().sum())
    summary.missing_values_handled += (missing_before - missing_after) + dropped_for_id
    logger.info("Handled %d missing values (median/mode imputation)", summary.missing_values_handled)
    return df


def _handle_invalid_values(df: pd.DataFrame, summary: ETLSummary) -> pd.DataFrame:
    """
    Invalid-value strategy (documented):
      - Any numeric value outside its documented valid range (VALID_RANGES)
        is clipped to the nearest valid boundary. This preserves the
        record (rather than dropping it) while removing impossible values,
        which is appropriate since anomalies were injected as corrupted
        versions of otherwise-plausible records.
      - Gender values outside {"Male", "Female"} are treated as invalid
        and replaced with the modal category.
    """
    df = df.copy()
    invalid_count = 0

    for col, (low, high) in VALID_RANGES.items():
        out_of_range = ~df[col].between(low, high)
        n_out = int(out_of_range.sum())
        if n_out:
            invalid_count += n_out
            df.loc[out_of_range, col] = df.loc[out_of_range, col].clip(lower=low, upper=high)

    valid_genders = {"Male", "Female"}
    bad_gender = ~df["gender"].isin(valid_genders)
    n_bad_gender = int(bad_gender.sum())
    if n_bad_gender:
        invalid_count += n_bad_gender
        mode_val = df["gender"].mode(dropna=True).iloc[0]
        df.loc[bad_gender, "gender"] = mode_val

    summary.invalid_values_handled += invalid_count
    logger.info("Handled %d invalid/out-of-range values (clipped to valid bounds)", invalid_count)
    return df


def _normalize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce consistent dtypes across all columns."""
    df = df.copy()
    df["student_id"] = df["student_id"].astype(str)
    df["gender"] = df["gender"].astype(str)
    df["age"] = df["age"].round().astype(int)

    for col in [c for c in REQUIRED_COLUMNS if c not in ("student_id", "gender", "age")]:
        df[col] = df[col].astype(float).round(2)

    return df


def transform(df: pd.DataFrame, summary: ETLSummary) -> pd.DataFrame:
    """TRANSFORM: validate, clean, and standardize the raw dataset."""
    _validate_columns(df)
    df = _remove_duplicates(df, summary)
    df = _handle_missing_values(df, summary)
    df = _handle_invalid_values(df, summary)
    df = _normalize_types(df)

    # NOTE: derived analytical columns (assessment_average, risk features,
    # etc.) are intentionally NOT added here — that is the responsibility
    # of src/feature_engineering.py, to keep ETL (validate/clean/load)
    # and feature engineering (derive analytical columns) cleanly separated.

    logger.info("Transformation complete: %d clean records", len(df))
    return df


def load(df: pd.DataFrame, csv_path: Path = CLEAN_CSV_PATH, db_path: Path = SQLITE_DB_PATH) -> None:
    """LOAD: persist the cleaned dataset to CSV and SQLite."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    logger.info("Clean dataset saved to: %s", csv_path)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        df.to_sql("student_performance", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_student_id ON student_performance(student_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_attendance ON student_performance(attendance_rate);")
        conn.commit()
    finally:
        conn.close()
    logger.info("Clean dataset loaded into SQLite table 'student_performance' at: %s", db_path)


def run_etl(raw_path: Path = RAW_DATA_PATH) -> tuple[pd.DataFrame, ETLSummary]:
    """Execute the full Extract -> Transform -> Load pipeline."""
    summary = ETLSummary()
    df_raw = extract(raw_path)
    summary.raw_records = len(df_raw)

    df_clean = transform(df_raw, summary)
    summary.final_records = len(df_clean)

    load(df_clean)
    summary.print_report()
    return df_clean, summary


def main() -> tuple[pd.DataFrame, ETLSummary]:
    return run_etl()


if __name__ == "__main__":
    main()
