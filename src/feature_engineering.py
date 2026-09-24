"""
feature_engineering.py
=======================

Derives educational-analytics features from the cleaned student dataset.

The module explicitly separates:
  - PRE-EXAM features: available before the final examination, safe to
    use for early academic-risk detection.
  - POST-EXAM features/analysis: may use `final_exam_score`, used only
    for retrospective evaluation, never for pre-exam risk scoring.

Run directly:
    python -m src.feature_engineering
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "clean_student_academic_data.csv"
FEATURED_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "featured_student_academic_data.csv"

# Columns that must NEVER be used when computing pre-exam risk features.
LEAKAGE_COLUMNS = {"final_exam_score"}


def add_assessment_average(df: pd.DataFrame) -> pd.DataFrame:
    """Average of the four pre-exam assessment components (0-100)."""
    df = df.copy()
    df["assessment_average"] = (
        df[["assignment_score", "quiz_score", "midterm_score", "continuous_assessment"]]
        .mean(axis=1)
        .round(2)
    )
    return df


def add_performance_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    performance_trend: compares an "early" assessment average
    (assignment + quiz) against a "later" pre-exam assessment average
    (midterm + continuous assessment).

    Positive values indicate improvement over the term; negative values
    indicate decline. This is entirely pre-exam information.
    """
    df = df.copy()
    early = df[["assignment_score", "quiz_score"]].mean(axis=1)
    later = df[["midterm_score", "continuous_assessment"]].mean(axis=1)
    df["performance_trend"] = (later - early).round(2)
    return df


def add_attendance_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    attendance_risk: 0-100 scale, higher = riskier.
    Simple linear inverse of attendance_rate.
    """
    df = df.copy()
    df["attendance_risk"] = (100 - df["attendance_rate"]).clip(0, 100).round(2)
    return df


def add_study_efficiency(df: pd.DataFrame) -> pd.DataFrame:
    """
    study_efficiency: assessment_average achieved per hour of weekly
    study, on a bounded 0-100 scale, as a rough proxy for how
    effectively study time converts into performance.

    Two guards keep this a well-behaved, bounded feature rather than an
    unstable ratio:
      - A floor of 1 hour is used in the denominator (instead of the raw
        value), so students reporting near-zero study hours don't produce
        an extreme, uninformative spike in the ratio.
      - The result is clipped to [0, 100], the same scale as the other
        pre-exam features, so it stays comparable and well-behaved for
        both visualization and modeling.
    """
    df = df.copy()
    if "assessment_average" not in df.columns:
        df = add_assessment_average(df)
    hours_floor = np.maximum(df["study_hours_per_week"].to_numpy(dtype=float), 1.0)
    average = df["assessment_average"].to_numpy(dtype=float)
    efficiency = np.clip(average / hours_floor, 0, 100)
    df["study_efficiency"] = pd.Series(efficiency, index=df.index).round(2)
    return df


def add_previous_performance_category(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize previous_gpa into ordinal bands."""
    df = df.copy()
    bins = [-0.01, 2.0, 3.0, 4.0]
    labels = ["Low", "Medium", "High"]
    df["previous_performance_category"] = pd.cut(df["previous_gpa"], bins=bins, labels=labels)
    return df


def add_assessment_performance_category(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize assessment_average into ordinal bands."""
    df = df.copy()
    if "assessment_average" not in df.columns:
        df = add_assessment_average(df)
    bins = [-0.01, 50, 70, 100]
    labels = ["Low", "Medium", "High"]
    df["assessment_performance_category"] = pd.cut(df["assessment_average"], bins=bins, labels=labels)
    return df


def add_academic_performance_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    academic_performance_index (API): a composite 0-100 pre-exam index
    blending assessment performance, attendance, and prior GPA. This is
    a descriptive index (NOT the risk score) used for EDA/summary
    purposes.

    API = 0.5 * assessment_average
        + 0.3 * attendance_rate
        + 0.2 * (previous_gpa / 4.0 * 100)
    """
    df = df.copy()
    if "assessment_average" not in df.columns:
        df = add_assessment_average(df)
    df["academic_performance_index"] = (
        0.5 * df["assessment_average"]
        + 0.3 * df["attendance_rate"]
        + 0.2 * (df["previous_gpa"] / 4.0 * 100)
    ).round(2)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full pre-exam feature-engineering pipeline, in order."""
    df = add_assessment_average(df)
    df = add_performance_trend(df)
    df = add_attendance_risk(df)
    df = add_study_efficiency(df)
    df = add_academic_performance_index(df)
    df = add_previous_performance_category(df)
    df = add_assessment_performance_category(df)
    logger.info("Feature engineering complete: %d columns total", df.shape[1])
    return df


def assert_no_leakage(feature_columns: list[str]) -> None:
    """
    Raise if any pre-exam feature set accidentally includes a
    post-exam/leakage column such as final_exam_score.
    """
    leaked = LEAKAGE_COLUMNS.intersection(feature_columns)
    if leaked:
        raise ValueError(
            f"Data leakage detected: pre-exam feature set includes forbidden column(s): {leaked}"
        )


def get_pre_exam_feature_columns() -> list[str]:
    """
    The canonical list of columns that are legitimate for PRE-EXAM risk
    detection and predictive modeling. Explicitly excludes
    `final_exam_score`.
    """
    cols = [
        "attendance_rate",
        "study_hours_per_week",
        "assignment_score",
        "quiz_score",
        "midterm_score",
        "continuous_assessment",
        "previous_gpa",
        "assessment_average",
        "performance_trend",
        "attendance_risk",
        "study_efficiency",
        "academic_performance_index",
    ]
    assert_no_leakage(cols)
    return cols


def save_featured_dataset(df: pd.DataFrame, path: Path = FEATURED_CSV_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Featured dataset saved to: %s", path)


def main() -> pd.DataFrame:
    df = pd.read_csv(CLEAN_CSV_PATH)
    df_featured = engineer_features(df)
    assert_no_leakage(get_pre_exam_feature_columns())
    save_featured_dataset(df_featured)
    print(f"Feature engineering complete. Shape: {df_featured.shape}")
    return df_featured


if __name__ == "__main__":
    main()
