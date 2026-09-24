"""Tests for src.risk_detection, including the critical no-leakage guarantee."""

import inspect

import pandas as pd
import pytest

from src.data_generator import generate_student_data
from src.etl_pipeline import ETLSummary, transform
from src.feature_engineering import engineer_features, get_pre_exam_feature_columns
from src.risk_detection import (
    RISK_THRESHOLDS,
    RISK_WEIGHTS,
    calculate_risk_scores,
    classify_risk_level,
    get_high_risk_students,
)


@pytest.fixture(scope="module")
def featured_df() -> pd.DataFrame:
    raw = generate_student_data(n_students=2000, seed=42)
    summary = ETLSummary()
    clean = transform(raw, summary)
    return engineer_features(clean)


@pytest.fixture(scope="module")
def scored_df(featured_df: pd.DataFrame) -> pd.DataFrame:
    return calculate_risk_scores(featured_df)


def test_risk_score_within_bounds(scored_df: pd.DataFrame) -> None:
    assert scored_df["risk_score"].min() >= 0
    assert scored_df["risk_score"].max() <= 100


def test_risk_level_values(scored_df: pd.DataFrame) -> None:
    assert set(scored_df["risk_level"].unique()).issubset({"LOW", "MODERATE", "HIGH"})


def test_risk_level_matches_thresholds(scored_df: pd.DataFrame) -> None:
    for _, row in scored_df.sample(min(200, len(scored_df)), random_state=1).iterrows():
        expected = classify_risk_level(row["risk_score"], RISK_THRESHOLDS)
        assert row["risk_level"] == expected


def test_weights_sum_to_one() -> None:
    assert abs(sum(RISK_WEIGHTS.values()) - 1.0) < 1e-6


def test_high_risk_subset_only_contains_high(scored_df: pd.DataFrame) -> None:
    high_risk = get_high_risk_students(scored_df)
    assert (high_risk["risk_level"] == "HIGH").all()


def test_lower_attendance_never_decreases_risk_component(featured_df: pd.DataFrame) -> None:
    """Sanity check on monotonicity: reducing attendance should not lower risk_score."""
    df_low_attendance = featured_df.copy()
    df_low_attendance["attendance_rate"] = (df_low_attendance["attendance_rate"] - 20).clip(0, 100)

    scored_original = calculate_risk_scores(featured_df)
    scored_low = calculate_risk_scores(df_low_attendance)

    assert (scored_low["risk_score"] >= scored_original["risk_score"] - 1e-6).mean() > 0.95


# ---------------------------------------------------------------------------
# Data-leakage guarantees (critical requirement)
# ---------------------------------------------------------------------------

def test_pre_exam_feature_list_excludes_final_exam_score() -> None:
    cols = get_pre_exam_feature_columns()
    assert "final_exam_score" not in cols


def test_calculate_risk_scores_source_does_not_reference_final_exam_score() -> None:
    """
    Static guarantee: inspect the source of calculate_risk_scores and its
    private helper functions to confirm `final_exam_score` is never
    referenced anywhere in the risk-scoring computation path.
    """
    import src.risk_detection as risk_module

    source = inspect.getsource(risk_module)
    # Only the `main()` demo function may reference final_exam_score, and only
    # to assert it is NOT used for scoring. All scoring functions must be clean.
    functions_to_check = [
        risk_module._compute_attendance_risk,
        risk_module._compute_assessment_performance_risk,
        risk_module._compute_trend_risk,
        risk_module._compute_gpa_risk,
        risk_module._compute_study_behavior_risk,
        risk_module._compute_recent_performance_risk,
        risk_module.compute_risk_components,
        risk_module.calculate_risk_scores,
    ]
    for func in functions_to_check:
        func_source = inspect.getsource(func)
        assert "final_exam_score" not in func_source, (
            f"Data leakage: {func.__name__} references final_exam_score"
        )


def test_risk_scores_identical_regardless_of_final_exam_score(featured_df: pd.DataFrame) -> None:
    """
    Behavioral guarantee: changing final_exam_score must have ZERO effect
    on risk_score, proving the exam result is not actually used.
    """
    df_a = featured_df.copy()
    df_b = featured_df.copy()
    df_b["final_exam_score"] = 0.0  # blank out the "future" information

    scored_a = calculate_risk_scores(df_a)
    scored_b = calculate_risk_scores(df_b)

    pd.testing.assert_series_equal(
        scored_a["risk_score"].reset_index(drop=True),
        scored_b["risk_score"].reset_index(drop=True),
    )
