"""
main.py
=======

Orchestrates the complete EDM Analytics Pipeline end-to-end:

    1. Synthetic data generation
    2. ETL (extract, transform, load)
    3. Feature engineering
    4. Academic risk detection (pre-exam only)
    5. EDA, statistical analysis, and ML model evaluation
    6. Analytical report generation

Usage:
    python -m src.main
"""

from __future__ import annotations

import logging
import time

from src import analytics, data_generator, etl_pipeline, feature_engineering, risk_detection

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    start = time.time()

    print("\n########## STEP 1/6: SYNTHETIC DATA GENERATION ##########")
    data_generator.main()

    print("\n########## STEP 2/6: ETL PIPELINE ##########")
    df_clean, _ = etl_pipeline.run_etl()

    print("\n########## STEP 3/6: FEATURE ENGINEERING ##########")
    df_featured = feature_engineering.engineer_features(df_clean)
    feature_engineering.assert_no_leakage(feature_engineering.get_pre_exam_feature_columns())
    feature_engineering.save_featured_dataset(df_featured)

    print("\n########## STEP 4/6: ACADEMIC RISK DETECTION ##########")
    df_scored = risk_detection.calculate_risk_scores(df_featured)
    risk_detection.save_risk_scores(df_scored)
    high_risk = risk_detection.get_high_risk_students(df_scored)
    print(f"High-risk students identified: {len(high_risk):,} of {len(df_scored):,} "
          f"({len(high_risk) / len(df_scored) * 100:.1f}%)")

    print("\n########## STEP 5/6: EDA + STATISTICAL ANALYSIS + ML EVALUATION ##########")
    analytics.main()

    print("\n########## STEP 6/6: PIPELINE COMPLETE ##########")
    elapsed = time.time() - start
    print(f"Total pipeline runtime: {elapsed:.1f} seconds")
    print("\nOutputs written to:")
    print("  data/raw/student_academic_data.csv")
    print("  data/processed/clean_student_academic_data.csv")
    print("  data/processed/education_analytics.db")
    print("  data/processed/featured_student_academic_data.csv")
    print("  data/processed/student_risk_scores.csv")
    print("  data/reports/*.png (9 charts)")
    print("  data/reports/statistical_summary.csv")
    print("  data/reports/model_performance.csv")
    print("  data/reports/analytical_report.txt")


if __name__ == "__main__":
    run_pipeline()
