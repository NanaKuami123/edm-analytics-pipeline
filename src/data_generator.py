"""
data_generator.py
==================

Generates a synthetic student-academic-performance dataset for the
Educational Data Mining (EDM) Analytics Pipeline.

IMPORTANT
---------
All data produced by this module is entirely SYNTHETIC. No real
students, institutions, or academic records are used or represented.
The data is generated from parametric statistical models with a fixed
random seed for full reproducibility.

Run directly:
    python -m src.data_generator
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
N_STUDENTS = 10_000

# Project root and output path, resolved relative to this file so the
# script works regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_academic_data.csv"


def _clip(series: pd.Series, low: float, high: float) -> pd.Series:
    """Clip a numeric series to a valid range."""
    return series.clip(lower=low, upper=high)


def generate_student_data(n_students: int = N_STUDENTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a synthetic dataset of student academic records.

    The generation process builds each variable from an underlying latent
    "academic aptitude/engagement" factor plus independent noise, which
    creates realistic (imperfect) correlations between attendance, study
    habits, prior performance, and outcomes -- without hard-coding a
    deterministic formula.

    Parameters
    ----------
    n_students : int
        Number of synthetic student records to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Raw synthetic dataset (may include missing values / mild
        anomalies, by design, for the ETL stage to clean).
    """
    rng = np.random.default_rng(seed)
    logger.info("Generating %d synthetic student records (seed=%d)...", n_students, seed)

    student_id = np.array([f"STU{100000 + i}" for i in range(n_students)])

    age = rng.integers(18, 31, size=n_students)  # 18-30 inclusive
    gender = rng.choice(["Male", "Female"], size=n_students, p=[0.5, 0.5])

    # Latent ability/engagement factor drives correlated outcomes.
    # Roughly standard-normal; individual variables are derived from it
    # plus their own independent noise so relationships stay imperfect.
    latent_ability = rng.normal(loc=0.0, scale=1.0, size=n_students)

    # Previous GPA (0-4.0): influenced by latent ability.
    previous_gpa = 2.5 + 0.5 * latent_ability + rng.normal(0, 0.35, n_students)
    previous_gpa = _clip(pd.Series(previous_gpa), 0.0, 4.0)

    # Attendance rate (0-100%): influenced by latent ability/engagement.
    attendance_rate = 78 + 12 * latent_ability + rng.normal(0, 8, n_students)
    attendance_rate = _clip(pd.Series(attendance_rate), 0.0, 100.0)

    # Study hours per week (0-40): influenced by latent ability, weaker link.
    study_hours_per_week = 15 + 6 * latent_ability + rng.normal(0, 5, n_students)
    study_hours_per_week = _clip(pd.Series(study_hours_per_week), 0.0, 40.0)

    # Assessment scores (0-100), each correlated with latent ability,
    # attendance, and study hours, with independent noise per assessment.
    def score_from_factors(base: float, ability_w: float, noise_sd: float) -> pd.Series:
        raw = (
            base
            + ability_w * latent_ability
            + 0.15 * (attendance_rate - 78) / 12
            + 0.10 * (study_hours_per_week - 15) / 6
            + rng.normal(0, noise_sd, n_students)
        )
        return _clip(pd.Series(raw), 0.0, 100.0)

    assignment_score = score_from_factors(base=72, ability_w=10, noise_sd=9)
    quiz_score = score_from_factors(base=70, ability_w=11, noise_sd=10)
    midterm_score = score_from_factors(base=68, ability_w=12, noise_sd=10)
    continuous_assessment = score_from_factors(base=71, ability_w=10, noise_sd=9)

    # Final exam score: correlated with everything above, with an added
    # "decline" penalty for students whose trend was already weakening
    # (midterm markedly below early assessments), simulating compounding
    # risk. Independent exam-day noise included.
    early_avg = (assignment_score + quiz_score) / 2
    trend_gap = midterm_score - early_avg  # negative => declining
    decline_penalty = np.where(trend_gap < -5, (trend_gap + 5) * 0.4, 0)

    final_exam_score = (
        0.35 * midterm_score
        + 0.20 * continuous_assessment
        + 0.15 * early_avg
        + 0.15 * (previous_gpa / 4.0 * 100)
        + 0.10 * attendance_rate
        + 0.05 * (study_hours_per_week / 40 * 100)
        + decline_penalty
        + rng.normal(0, 7, n_students)
    )
    final_exam_score = _clip(pd.Series(final_exam_score), 0.0, 100.0)

    df = pd.DataFrame(
        {
            "student_id": student_id,
            "age": age,
            "gender": gender,
            "attendance_rate": attendance_rate.round(2),
            "study_hours_per_week": study_hours_per_week.round(2),
            "assignment_score": assignment_score.round(2),
            "quiz_score": quiz_score.round(2),
            "midterm_score": midterm_score.round(2),
            "continuous_assessment": continuous_assessment.round(2),
            "previous_gpa": previous_gpa.round(2),
            "final_exam_score": final_exam_score.round(2),
        }
    )

    df = _inject_missing_values(df, rng, fraction=0.02)
    df = _inject_anomalies(df, rng, fraction=0.01)

    logger.info("Synthetic dataset generation complete: %d rows, %d columns", *df.shape)
    return df


def _inject_missing_values(df: pd.DataFrame, rng: np.random.Generator, fraction: float) -> pd.DataFrame:
    """Randomly null out a small fraction of values in numeric/categorical columns (not student_id)."""
    df = df.copy()
    eligible_cols = [c for c in df.columns if c != "student_id"]
    for col in eligible_cols:
        n_missing = int(len(df) * fraction * rng.uniform(0.5, 1.5))
        idx = rng.choice(df.index, size=min(n_missing, len(df) - 1), replace=False)
        df.loc[idx, col] = np.nan
    logger.info("Injected missing values into %d columns (~%.1f%% each)", len(eligible_cols), fraction * 100)
    return df


def _inject_anomalies(df: pd.DataFrame, rng: np.random.Generator, fraction: float) -> pd.DataFrame:
    """Inject a small number of out-of-range / implausible values to exercise ETL validation logic."""
    df = df.copy()
    n_anomalies = int(len(df) * fraction)
    anomaly_idx = rng.choice(df.index, size=n_anomalies, replace=False)

    numeric_score_cols = [
        "attendance_rate",
        "assignment_score",
        "quiz_score",
        "midterm_score",
        "continuous_assessment",
        "final_exam_score",
    ]

    for i in anomaly_idx:
        col = rng.choice(numeric_score_cols)
        anomaly_type = rng.choice(["negative", "over_100", "extreme_outlier"])
        if pd.isna(df.at[i, col]):
            continue
        if anomaly_type == "negative":
            df.at[i, col] = -rng.uniform(1, 15)
        elif anomaly_type == "over_100":
            df.at[i, col] = 100 + rng.uniform(1, 40)
        else:
            df.at[i, col] = df.at[i, col] * rng.uniform(3, 5)

    # A handful of duplicate rows (by student_id) to exercise dedup logic.
    n_dupes = max(1, int(len(df) * 0.003))
    dupe_rows = df.sample(n=n_dupes, random_state=int(rng.integers(0, 10_000)))
    df = pd.concat([df, dupe_rows], ignore_index=True)

    # A handful of implausible ages, to exercise range validation.
    n_bad_ages = max(1, int(len(df) * 0.001))
    bad_age_idx = rng.choice(df.index, size=n_bad_ages, replace=False)
    df.loc[bad_age_idx, "age"] = rng.integers(60, 90, size=n_bad_ages)

    logger.info(
        "Injected %d value anomalies, %d duplicate rows, %d implausible ages",
        n_anomalies, n_dupes, n_bad_ages,
    )
    return df


def save_raw_dataset(df: pd.DataFrame, path: Path = RAW_DATA_PATH) -> None:
    """Persist the raw synthetic dataset to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Raw dataset saved to: %s", path)


def print_summary(df: pd.DataFrame) -> None:
    """Print a human-readable summary of the generated dataset."""
    print("\n" + "=" * 60)
    print("SYNTHETIC STUDENT DATASET — GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total records generated : {len(df):,}")
    print(f"Total columns            : {df.shape[1]}")
    print(f"Missing values (total)   : {int(df.isna().sum().sum()):,}")
    print(f"Duplicate student_ids    : {int(df['student_id'].duplicated().sum()):,}")
    print("\nColumn dtypes:")
    print(df.dtypes.to_string())
    print("\nNumeric summary (head):")
    print(df.describe(include="number").T[["mean", "std", "min", "max"]].round(2))
    print("=" * 60 + "\n")


def main() -> pd.DataFrame:
    df = generate_student_data()
    print_summary(df)
    save_raw_dataset(df)
    return df


if __name__ == "__main__":
    main()
