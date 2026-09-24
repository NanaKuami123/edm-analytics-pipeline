"""
analytics.py
============

Exploratory data analysis, statistical analysis, a predictive-modeling
extension, and a summary analytical report for the EDM pipeline.

Run directly:
    python -m src.analytics
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.feature_engineering import get_pre_exam_feature_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURED_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "featured_student_academic_data.csv"
RISK_SCORES_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "student_risk_scores.csv"
REPORTS_DIR = PROJECT_ROOT / "data" / "reports"

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Data loading helper
# ---------------------------------------------------------------------------

def load_full_dataset() -> pd.DataFrame:
    """Load the featured dataset merged with computed risk scores."""
    df_features = pd.read_csv(FEATURED_CSV_PATH)
    df_risk = pd.read_csv(RISK_SCORES_CSV_PATH)
    df = df_features.merge(df_risk, on="student_id", how="inner")
    return df


# ---------------------------------------------------------------------------
# Exploratory Data Analysis (charts)
# ---------------------------------------------------------------------------

def _savefig(fig: plt.Figure, filename: str) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved chart: %s", path)


def plot_attendance_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["attendance_rate"], bins=30, kde=True, ax=ax, color="#4C72B0")
    ax.set_title("Distribution of Student Attendance Rate")
    ax.set_xlabel("Attendance Rate (%)")
    ax.set_ylabel("Number of Students")
    _savefig(fig, "attendance_distribution.png")


def plot_assessment_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    cols = ["assignment_score", "quiz_score", "midterm_score", "continuous_assessment"]
    melted = df[cols].melt(var_name="Assessment", value_name="Score")
    sns.boxplot(data=melted, x="Assessment", y="Score", hue="Assessment", ax=ax, palette="Set2", legend=False)
    ax.set_title("Distribution of Pre-Exam Assessment Scores")
    ax.set_xlabel("Assessment Type")
    ax.set_ylabel("Score (0-100)")
    _savefig(fig, "assessment_distribution.png")


def plot_performance_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["final_exam_score"], bins=30, kde=True, ax=ax, color="#DD8452")
    ax.set_title("Distribution of Final Examination Scores")
    ax.set_xlabel("Final Exam Score")
    ax.set_ylabel("Number of Students")
    _savefig(fig, "performance_distribution.png")


def plot_attendance_vs_performance(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="attendance_rate", y="final_exam_score", alpha=0.25, s=15, ax=ax)
    sns.regplot(data=df, x="attendance_rate", y="final_exam_score", scatter=False, ax=ax, color="red")
    ax.set_title("Attendance Rate vs. Final Exam Score")
    ax.set_xlabel("Attendance Rate (%)")
    ax.set_ylabel("Final Exam Score")
    _savefig(fig, "attendance_vs_performance.png")


def plot_study_hours_vs_performance(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="study_hours_per_week", y="final_exam_score", alpha=0.25, s=15, ax=ax)
    sns.regplot(data=df, x="study_hours_per_week", y="final_exam_score", scatter=False, ax=ax, color="red")
    ax.set_title("Weekly Study Hours vs. Final Exam Score")
    ax.set_xlabel("Study Hours per Week")
    ax.set_ylabel("Final Exam Score")
    _savefig(fig, "study_hours_vs_performance.png")


def plot_gpa_vs_performance(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="previous_gpa", y="final_exam_score", alpha=0.25, s=15, ax=ax)
    sns.regplot(data=df, x="previous_gpa", y="final_exam_score", scatter=False, ax=ax, color="red")
    ax.set_title("Previous GPA vs. Final Exam Score")
    ax.set_xlabel("Previous GPA (0-4.0)")
    ax.set_ylabel("Final Exam Score")
    _savefig(fig, "gpa_vs_performance.png")


def plot_risk_level_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    order = ["LOW", "MODERATE", "HIGH"]
    counts = df["risk_level"].value_counts().reindex(order)
    palette = {"LOW": "#55A868", "MODERATE": "#DD8452", "HIGH": "#C44E52"}
    sns.barplot(
        x=counts.index, y=counts.values, hue=counts.index, ax=ax,
        palette=[palette[k] for k in order], legend=False,
    )
    ax.set_title("Distribution of Academic Risk Levels")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Number of Students")
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, f"{v:,}", ha="center")
    _savefig(fig, "risk_level_distribution.png")


def plot_performance_trend(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["performance_trend"], bins=30, kde=True, ax=ax, color="#8172B2")
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Distribution of Pre-Exam Performance Trend\n(later assessments − earlier assessments)")
    ax.set_xlabel("Performance Trend")
    ax.set_ylabel("Number of Students")
    _savefig(fig, "performance_trend.png")


def plot_high_risk_analysis(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.boxplot(
        data=df, x="risk_level", y="attendance_rate", order=["LOW", "MODERATE", "HIGH"],
        hue="risk_level", hue_order=["LOW", "MODERATE", "HIGH"],
        ax=axes[0], palette="Set2", legend=False,
    )
    axes[0].set_title("Attendance Rate by Risk Level")
    axes[0].set_xlabel("Risk Level")
    axes[0].set_ylabel("Attendance Rate (%)")

    sns.boxplot(
        data=df, x="risk_level", y="final_exam_score", order=["LOW", "MODERATE", "HIGH"],
        hue="risk_level", hue_order=["LOW", "MODERATE", "HIGH"],
        ax=axes[1], palette="Set2", legend=False,
    )
    axes[1].set_title("Final Exam Score by Risk Level (retrospective)")
    axes[1].set_xlabel("Risk Level")
    axes[1].set_ylabel("Final Exam Score")

    fig.suptitle("Characteristics of Students by Academic Risk Level")
    _savefig(fig, "high_risk_analysis.png")


def run_eda(df: pd.DataFrame) -> None:
    """Generate and save all EDA charts."""
    plot_attendance_distribution(df)
    plot_assessment_distribution(df)
    plot_performance_distribution(df)
    plot_attendance_vs_performance(df)
    plot_study_hours_vs_performance(df)
    plot_gpa_vs_performance(df)
    plot_risk_level_distribution(df)
    plot_performance_trend(df)
    plot_high_risk_analysis(df)
    logger.info("EDA complete: 9 charts saved to %s", REPORTS_DIR)


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def compute_statistical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute descriptive statistics and Pearson correlations with
    final_exam_score for key pre-exam variables. Saved for retrospective
    (post-exam) analysis only.
    """
    numeric_cols = [
        "attendance_rate",
        "study_hours_per_week",
        "assignment_score",
        "quiz_score",
        "midterm_score",
        "continuous_assessment",
        "previous_gpa",
        "final_exam_score",
        "risk_score",
    ]

    rows = []
    for col in numeric_cols:
        series = df[col].dropna()
        rows.append(
            {
                "variable": col,
                "mean": round(series.mean(), 3),
                "median": round(series.median(), 3),
                "std_dev": round(series.std(), 3),
                "min": round(series.min(), 3),
                "max": round(series.max(), 3),
            }
        )
    desc_df = pd.DataFrame(rows)

    correlation_targets = [
        "attendance_rate",
        "study_hours_per_week",
        "previous_gpa",
        "continuous_assessment",
        "assessment_average",
        "risk_score",
    ]
    corr_rows = []
    for col in correlation_targets:
        r, p = stats.pearsonr(df[col].dropna(), df.loc[df[col].dropna().index, "final_exam_score"])
        corr_rows.append(
            {
                "variable": col,
                "pearson_r_with_final_exam_score": round(r, 4),
                "p_value": round(p, 6),
            }
        )
    corr_df = pd.DataFrame(corr_rows)

    summary = desc_df.merge(corr_df, on="variable", how="left")
    return summary


def save_statistical_summary(summary: pd.DataFrame, path: Path = None) -> None:
    path = path or (REPORTS_DIR / "statistical_summary.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    logger.info("Statistical summary saved to: %s", path)


# ---------------------------------------------------------------------------
# Machine learning extension (pre-exam prediction of final_exam_score)
# ---------------------------------------------------------------------------

def train_and_evaluate_models(df: pd.DataFrame, random_state: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Train and evaluate Linear Regression and Random Forest models that
    predict final_exam_score from legitimate PRE-EXAM features only.

    NOTE: This model is trained and evaluated exclusively on synthetic
    data. It has not been validated for use in any real educational
    institution and must not be deployed to make decisions about real
    students.
    """
    feature_cols = get_pre_exam_feature_columns()
    X = df[feature_cols]
    y = df["final_exam_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200, max_depth=8, random_state=random_state, n_jobs=-1
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = r2_score(y_test, preds)

        results.append(
            {
                "model": name,
                "mae": round(mae, 3),
                "rmse": round(rmse, 3),
                "r2_score": round(r2, 4),
                "n_train": len(X_train),
                "n_test": len(X_test),
            }
        )
        logger.info("%s -> MAE=%.3f RMSE=%.3f R2=%.4f", name, mae, rmse, r2)

    return pd.DataFrame(results)


def save_model_performance(results: pd.DataFrame, path: Path = None) -> None:
    path = path or (REPORTS_DIR / "model_performance.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)
    logger.info("Model performance results saved to: %s", path)


# ---------------------------------------------------------------------------
# Analytical report
# ---------------------------------------------------------------------------

def generate_report_text(df: pd.DataFrame, model_results: pd.DataFrame) -> str:
    """Build a concise, data-driven analytical report (plain text)."""
    total = len(df)
    avg_attendance = df["attendance_rate"].mean()
    avg_assessment = df["assessment_average"].mean()
    avg_final = df["final_exam_score"].mean()

    risk_counts = df["risk_level"].value_counts().reindex(["LOW", "MODERATE", "HIGH"]).fillna(0).astype(int)

    r_attendance, _ = stats.pearsonr(df["attendance_rate"], df["final_exam_score"])
    r_assessment, _ = stats.pearsonr(df["assessment_average"], df["final_exam_score"])

    high_risk_df = df[df["risk_level"] == "HIGH"]
    low_risk_df = df[df["risk_level"] == "LOW"]

    best_model_row = model_results.sort_values("r2_score", ascending=False).iloc[0]

    lines = [
        "=" * 70,
        "EDM ANALYTICS PIPELINE — SUMMARY REPORT (SYNTHETIC DATA)",
        "=" * 70,
        "",
        f"Total students analyzed        : {total:,}",
        f"Average attendance rate        : {avg_attendance:.2f}%",
        f"Average assessment score       : {avg_assessment:.2f} / 100",
        f"Average final exam score       : {avg_final:.2f} / 100",
        "",
        "Risk level distribution:",
        f"  LOW RISK       : {risk_counts['LOW']:,} students ({risk_counts['LOW'] / total * 100:.1f}%)",
        f"  MODERATE RISK  : {risk_counts['MODERATE']:,} students ({risk_counts['MODERATE'] / total * 100:.1f}%)",
        f"  HIGH RISK      : {risk_counts['HIGH']:,} students ({risk_counts['HIGH'] / total * 100:.1f}%)",
        "",
        "Observations (descriptive, correlational — not causal):",
        f"  - Attendance rate is correlated with final exam score "
        f"(Pearson r = {r_attendance:.3f}). Students with higher attendance "
        f"tend to show higher final exam scores in this synthetic dataset; "
        f"this reflects the correlational structure built into the data "
        f"generator, not a causal claim about any real population.",
        f"  - Pre-exam assessment average is correlated with final exam score "
        f"(Pearson r = {r_assessment:.3f}), consistent with assessments "
        f"functioning as leading indicators of exam performance.",
        f"  - Students classified as HIGH RISK show a mean attendance rate of "
        f"{high_risk_df['attendance_rate'].mean():.1f}% and mean final exam "
        f"score of {high_risk_df['final_exam_score'].mean():.1f}, compared to "
        f"{low_risk_df['attendance_rate'].mean():.1f}% attendance and "
        f"{low_risk_df['final_exam_score'].mean():.1f} final exam score among "
        f"LOW RISK students — consistent with the risk score's intended "
        f"purpose of flagging students exhibiting patterns associated with "
        f"academic risk.",
        f"  - The best-performing predictive model on held-out synthetic data "
        f"was {best_model_row['model']} (R² = {best_model_row['r2_score']:.3f}, "
        f"MAE = {best_model_row['mae']:.2f}). This model is trained and "
        f"evaluated on SYNTHETIC data only and is not validated for "
        f"deployment in any real educational institution.",
        "",
        "All figures above are computed directly from the generated dataset;",
        "no conclusions in this report are fabricated or assumed.",
        "=" * 70,
    ]
    return "\n".join(lines)


def save_report(text: str, path: Path = None) -> None:
    path = path or (REPORTS_DIR / "analytical_report.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    logger.info("Analytical report saved to: %s", path)


def main() -> None:
    df = load_full_dataset()

    run_eda(df)

    summary = compute_statistical_summary(df)
    save_statistical_summary(summary)

    model_results = train_and_evaluate_models(df)
    save_model_performance(model_results)

    report_text = generate_report_text(df, model_results)
    print("\n" + report_text + "\n")
    save_report(report_text)


if __name__ == "__main__":
    main()
