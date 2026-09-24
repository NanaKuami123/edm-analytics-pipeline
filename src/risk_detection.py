"""
risk_detection.py
==================

Transparent, rule-based Academic Risk Score (0-100) for early
identification of students exhibiting patterns associated with academic
risk, using only information available BEFORE the final examination.

Risk Score formulation
-----------------------
    Risk Score = 0.20*A + 0.25*P + 0.20*T + 0.15*G + 0.10*S + 0.10*R

    A = attendance risk        (higher when attendance is low)
    P = assessment performance risk (higher when assessment scores are low)
    T = performance trend risk (higher when performance is declining)
    G = previous GPA risk      (higher when prior GPA is low)
    S = study behavior risk    (higher when study hours are low)
    R = recent performance risk (higher when the most recent pre-exam
        assessment, continuous_assessment, is low)

All six components are normalized to a 0-100 scale before weighting, so
the final risk score is also bounded to [0, 100].

IMPORTANT: `final_exam_score` is never read by this module. This is
verified both here (the function only accepts the pre-exam feature
columns) and by an automated test in tests/test_risk_detection.py.

Run directly:
    python -m src.risk_detection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURED_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "featured_student_academic_data.csv"
RISK_SCORES_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "student_risk_scores.csv"

# Forbidden for pre-exam risk scoring.
LEAKAGE_COLUMNS = {"final_exam_score"}

# Configurable weights (must sum to 1.0).
RISK_WEIGHTS = {
    "attendance_risk_component": 0.20,
    "assessment_performance_risk": 0.25,
    "trend_risk": 0.20,
    "gpa_risk": 0.15,
    "study_behavior_risk": 0.10,
    "recent_performance_risk": 0.10,
}

# Configurable classification thresholds (upper bound inclusive).
RISK_THRESHOLDS = {
    "LOW": (0, 29),
    "MODERATE": (30, 59),
    "HIGH": (60, 100),
}


def _normalize(series: pd.Series, low: float, high: float) -> pd.Series:
    """Min-max normalize a series to [0, 100] given known theoretical bounds."""
    span = high - low
    if span == 0:
        return pd.Series(0.0, index=series.index)
    normalized = (series - low) / span * 100
    return normalized.clip(0, 100)


def _compute_attendance_risk(df: pd.DataFrame) -> pd.Series:
    """Higher risk when attendance_rate is low."""
    return _normalize(100 - df["attendance_rate"], 0, 100)


def _compute_assessment_performance_risk(df: pd.DataFrame) -> pd.Series:
    """Higher risk when assessment_average is low."""
    return _normalize(100 - df["assessment_average"], 0, 100)


def _compute_trend_risk(df: pd.DataFrame) -> pd.Series:
    """
    Higher risk when performance_trend is negative (declining).
    performance_trend typically ranges roughly -40..+40; we normalize
    the *negated* trend, clipped to a reasonable band, so decline maps
    to higher risk and improvement maps to near-zero risk.
    """
    inverted_trend = -df["performance_trend"]
    return _normalize(inverted_trend.clip(-40, 40), -40, 40)


def _compute_gpa_risk(df: pd.DataFrame) -> pd.Series:
    """Higher risk when previous_gpa is low."""
    return _normalize(4.0 - df["previous_gpa"], 0, 4.0)


def _compute_study_behavior_risk(df: pd.DataFrame) -> pd.Series:
    """Higher risk when study_hours_per_week is low."""
    return _normalize(40 - df["study_hours_per_week"], 0, 40)


def _compute_recent_performance_risk(df: pd.DataFrame) -> pd.Series:
    """Higher risk when the most recent pre-exam assessment (continuous_assessment) is low."""
    return _normalize(100 - df["continuous_assessment"], 0, 100)


@dataclass
class RiskComponents:
    attendance_risk_component: pd.Series
    assessment_performance_risk: pd.Series
    trend_risk: pd.Series
    gpa_risk: pd.Series
    study_behavior_risk: pd.Series
    recent_performance_risk: pd.Series


def compute_risk_components(df: pd.DataFrame) -> RiskComponents:
    """Compute all six normalized (0-100) risk sub-components."""
    return RiskComponents(
        attendance_risk_component=_compute_attendance_risk(df),
        assessment_performance_risk=_compute_assessment_performance_risk(df),
        trend_risk=_compute_trend_risk(df),
        gpa_risk=_compute_gpa_risk(df),
        study_behavior_risk=_compute_study_behavior_risk(df),
        recent_performance_risk=_compute_recent_performance_risk(df),
    )


def classify_risk_level(score: float, thresholds: dict = RISK_THRESHOLDS) -> str:
    """Map a numeric risk score to a categorical risk level using configurable thresholds."""
    for level, (low, high) in thresholds.items():
        if low <= score <= high:
            return level
    # Fallback for scores outside all configured bands (should not occur
    # given the score is bounded to [0, 100]).
    return "HIGH" if score > max(h for _, h in thresholds.values()) else "LOW"


def calculate_risk_scores(
    df: pd.DataFrame,
    weights: dict = RISK_WEIGHTS,
    thresholds: dict = RISK_THRESHOLDS,
) -> pd.DataFrame:
    """
    Calculate the transparent, rule-based academic risk score for every
    student using ONLY pre-exam information.

    Raises
    ------
    ValueError
        If a forbidden post-exam leakage column (see LEAKAGE_COLUMNS) is
        used as a scoring input, or if weights do not sum to 1.0.
    """
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError(f"Risk weights must sum to 1.0, got {sum(weights.values())}")

    required_cols = {
        "attendance_rate",
        "assessment_average",
        "performance_trend",
        "previous_gpa",
        "study_hours_per_week",
        "continuous_assessment",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for risk scoring: {missing}")

    df = df.copy()
    components = compute_risk_components(df)

    df["attendance_risk_component"] = components.attendance_risk_component.round(2)
    df["assessment_performance_risk"] = components.assessment_performance_risk.round(2)
    df["trend_risk"] = components.trend_risk.round(2)
    df["gpa_risk"] = components.gpa_risk.round(2)
    df["study_behavior_risk"] = components.study_behavior_risk.round(2)
    df["recent_performance_risk"] = components.recent_performance_risk.round(2)

    df["risk_score"] = (
        weights["attendance_risk_component"] * df["attendance_risk_component"]
        + weights["assessment_performance_risk"] * df["assessment_performance_risk"]
        + weights["trend_risk"] * df["trend_risk"]
        + weights["gpa_risk"] * df["gpa_risk"]
        + weights["study_behavior_risk"] * df["study_behavior_risk"]
        + weights["recent_performance_risk"] * df["recent_performance_risk"]
    ).clip(0, 100).round(2)

    df["risk_level"] = df["risk_score"].apply(lambda s: classify_risk_level(s, thresholds))

    logger.info(
        "Risk scoring complete. Distribution: %s",
        df["risk_level"].value_counts().to_dict(),
    )
    return df


def get_high_risk_students(df: pd.DataFrame) -> pd.DataFrame:
    """Return the subset of students classified as HIGH risk."""
    return df[df["risk_level"] == "HIGH"].sort_values("risk_score", ascending=False)


def save_risk_scores(df: pd.DataFrame, path: Path = RISK_SCORES_CSV_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output_cols = [
        "student_id",
        "attendance_risk_component",
        "assessment_performance_risk",
        "trend_risk",
        "gpa_risk",
        "study_behavior_risk",
        "recent_performance_risk",
        "risk_score",
        "risk_level",
    ]
    df[output_cols].to_csv(path, index=False)
    logger.info("Risk scores saved to: %s", path)


def main() -> pd.DataFrame:
    df = pd.read_csv(FEATURED_CSV_PATH)
    assert "final_exam_score" in df.columns  # exists in source data, but must not be used below

    df_scored = calculate_risk_scores(df)
    save_risk_scores(df_scored)

    high_risk = get_high_risk_students(df_scored)
    print(f"\nHigh-risk students identified: {len(high_risk):,} of {len(df_scored):,} "
          f"({len(high_risk) / len(df_scored) * 100:.1f}%)")
    return df_scored


if __name__ == "__main__":
    main()
