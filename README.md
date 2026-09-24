# Educational Data Mining (EDM) Analytics Pipeline for Early Identification of At-Risk Students

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen)
![Data](https://img.shields.io/badge/data-synthetic-orange)

An end-to-end Educational Data Mining pipeline for analyzing synthetic
student data, detecting academic risk patterns, and evaluating predictive
models.

> ⚠️ **All student records in this repository are synthetically generated.**
> No real students, schools, or institutions are represented, referenced,
> or identifiable anywhere in this project.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Research Problem](#research-problem)
- [Objectives](#objectives)
- [Key Questions](#key-questions)
- [Architecture](#architecture)
- [Technology Stack](#technology-stack)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Example Outputs](#example-outputs)
- [Limitations](#limitations)
- [Ethical Considerations](#ethical-considerations)
- [Future Improvements](#future-improvements)
- [Reproducibility](#reproducibility)
- [Author](#author)

---

## Project Overview

This project is a portfolio-grade, reproducible data science pipeline that
simulates a realistic Educational Data Mining (EDM) workflow: generating a
synthetic dataset of 10,000 students, cleaning and engineering features
from it, computing a transparent rule-based academic risk score, and
producing exploratory, statistical, and machine-learning analyses — all
runnable end-to-end with a single command.

It is designed to demonstrate practical skills across data engineering,
ETL design, EDA, feature engineering, statistical analysis, machine
learning fundamentals, data visualization, software engineering, and
testing — in the specific domain of learning analytics.

## Research Problem

Educational institutions increasingly collect data on attendance,
coursework, and assessment performance. A common analytical goal is to
identify students exhibiting patterns associated with academic risk
**early enough** — before final examinations — that support can be
offered. This project explores how such a pipeline can be built
responsibly: using only information legitimately available before a final
exam, with a transparent (not black-box) scoring methodology.

## Objectives

1. Generate a realistic, correlated, but clearly synthetic student dataset.
2. Build a reproducible ETL pipeline that validates and cleans the data.
3. Engineer meaningful, leakage-free, pre-exam academic features.
4. Compute a transparent, explainable academic risk score.
5. Explore the data statistically and visually.
6. Build and evaluate baseline predictive models for final exam
   performance, using only pre-exam information.
7. Package everything as a tested, documented, reproducible pipeline.

## Key Questions

- Which pre-exam indicators (attendance, study habits, prior assessments,
  prior GPA) are most correlated with final exam performance?
- Can a simple, transparent rule-based score meaningfully separate
  students who go on to perform well from those who do not — **without**
  ever looking at the final exam result itself?
- How well can a basic regression model predict final exam performance
  from legitimate pre-exam features alone?

## Architecture

```
Synthetic Data Generation
        ↓
Raw Dataset (data/raw/)
        ↓
ETL Pipeline  (validate → dedupe → clean → load to CSV + SQLite)
        ↓
Feature Engineering  (pre-exam only, leakage-checked)
        ↓
Academic Risk Scoring  (transparent, rule-based, 0–100)
        ↓
At-Risk Student Identification
        ↓
Statistical Analysis + Exploratory Data Analysis
        ↓
Machine Learning Extension  (Linear Regression, Random Forest)
        ↓
Reports & Visualizations  (data/reports/)
```

Every stage is a plain Python module in `src/`, orchestrated by
`src/main.py`, so each step can also be run and inspected independently.

## Technology Stack

| Purpose | Library |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine learning | `scikit-learn` |
| Statistics | `scipy` |
| Storage | `sqlite3` (standard library) |
| Testing | `pytest` |
| Notebook | Jupyter (`notebooks/exploratory_analysis.ipynb`) |

Python 3.11+ is required.

## Dataset Description

`src/data_generator.py` produces **10,000 synthetic student records** with
a fixed random seed (`42`) for full reproducibility.

| Column | Description | Range |
|---|---|---|
| `student_id` | Synthetic identifier | `STU100000`–`STU109999` |
| `age` | Student age | 18–30 |
| `gender` | Male / Female | — |
| `attendance_rate` | Class attendance | 0–100% |
| `study_hours_per_week` | Self-reported weekly study time | 0–40 hrs |
| `assignment_score` | Assignment performance | 0–100 |
| `quiz_score` | Quiz performance | 0–100 |
| `midterm_score` | Midterm exam performance | 0–100 |
| `continuous_assessment` | Ongoing coursework assessment | 0–100 |
| `previous_gpa` | Prior-term GPA | 0.0–4.0 |
| `final_exam_score` | Final examination result | 0–100 |

Variables are **not** independently random: each is derived from a shared
latent "ability/engagement" factor plus independent noise, producing
realistic — but imperfect — correlations (e.g., higher attendance tends to
associate with better outcomes). The raw dataset also contains a small,
deliberate proportion of missing values, invalid/out-of-range values, and
duplicate records, so the ETL stage has meaningful cleaning work to do. See
[`docs/methodology.md`](docs/methodology.md) for full generation details.

## Methodology

Full methodology — including the exact risk-score formula, normalization
rules, and statistical/ML methods — is documented in
[`docs/methodology.md`](docs/methodology.md). Summary:

### ETL Process
Extract the raw CSV → validate schema → remove duplicates → impute missing
values (median/mode) → clip out-of-range values to documented valid bounds
→ normalize dtypes → load to `data/processed/clean_student_academic_data.csv`
and a SQLite database (`education_analytics.db`, table
`student_performance`, indexed on `student_id` and `attendance_rate`).

### Feature Engineering
Computes `assessment_average`, `performance_trend`, `attendance_risk`,
`study_efficiency`, `academic_performance_index`, and category bins for
prior GPA and assessment performance — **all using only pre-exam data**.
`final_exam_score` is explicitly excluded from the pre-exam feature set,
enforced by an automated leakage check (`assert_no_leakage`) that runs on
every pipeline execution and is covered by dedicated tests.

### Risk Scoring Methodology

```
Risk Score = 0.20·A + 0.25·P + 0.20·T + 0.15·G + 0.10·S + 0.10·R
```

| Symbol | Component | Weight |
|---|---|---|
| A | Attendance risk | 20% |
| P | Assessment performance risk | 25% |
| T | Performance trend risk | 20% |
| G | Previous GPA risk | 15% |
| S | Study behavior risk | 10% |
| R | Recent performance risk | 10% |

Each component is normalized to 0–100 before weighting, so `risk_score` is
always in [0, 100]. Students are classified as:

| Level | Range |
|---|---|
| LOW RISK | 0–29 |
| MODERATE RISK | 30–59 |
| HIGH RISK | 60–100 |

Thresholds and weights are configurable in `src/risk_detection.py`
(`RISK_WEIGHTS`, `RISK_THRESHOLDS`), not hard-coded throughout the codebase.

### Machine Learning Methodology
Linear Regression and Random Forest regressors are trained on an 80/20
split (`random_state=42`) using **only** the approved pre-exam feature set
to predict `final_exam_score`, evaluated with MAE, RMSE, and R².

## Results

*(Figures below are from a full run of `python -m src.main` on the 10,000
synthetic records generated with the default seed; re-running reproduces
them exactly.)*

- **10,000** synthetic students analyzed.
- Average attendance: **77.8%** · Average assessment score: **70.4/100** ·
  Average final exam score: **66.2/100**.
- Risk distribution: **2,543 LOW** (25.4%) · **7,337 MODERATE** (73.4%) ·
  **120 HIGH** (1.2%).
- Attendance vs. final exam score: Pearson **r = 0.647**. Assessment
  average vs. final exam score: **r = 0.792**. Previous GPA vs. final exam
  score: **r = 0.671**. (All correlational, not causal.)
- HIGH-risk students averaged **45.6%** attendance and a **35.0** final
  exam score, versus **91.2%** attendance and **81.1** for LOW-risk
  students — consistent with the risk score's intended purpose.
- Best model: **Random Forest Regressor**, R² = **0.750**, MAE = **5.76**,
  RMSE = **7.40** on held-out synthetic data (Linear Regression: R² =
  0.746, MAE = 5.80).

Full outputs are written to `data/reports/` on every run, including 9
charts, `statistical_summary.csv`, `model_performance.csv`, and
`analytical_report.txt`.

## Project Structure

```
edm-analytics-pipeline/
│
├── data/
│   ├── raw/                  # Raw synthetic dataset
│   ├── processed/            # Cleaned/featured CSVs + SQLite DB
│   └── reports/              # Charts, statistical summary, ML results, report
│
├── src/
│   ├── data_generator.py     # Synthetic data generation
│   ├── etl_pipeline.py       # Extract, transform, load
│   ├── feature_engineering.py# Pre-exam feature engineering + leakage guard
│   ├── risk_detection.py     # Transparent rule-based risk scoring
│   ├── analytics.py          # EDA, statistics, ML, reporting
│   └── main.py                # End-to-end orchestration
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── tests/
│   ├── test_data_generator.py
│   ├── test_etl_pipeline.py
│   └── test_risk_detection.py
│
├── docs/
│   └── methodology.md
│
├── requirements.txt
├── requirements-dev.txt
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/<your-username>/edm-analytics-pipeline.git
cd edm-analytics-pipeline

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt          # to run the pipeline
pip install -r requirements-dev.txt      # to also run tests / notebook
```

## Usage

Run the complete pipeline end-to-end:

```bash
python -m src.main
```

This executes, in order: data generation → ETL → feature engineering →
risk detection → EDA/statistics/ML evaluation → report generation, and
prints a summary of every output file written.

Individual stages can also be run independently, e.g.:

```bash
python -m src.data_generator
python -m src.etl_pipeline
python -m src.feature_engineering
python -m src.risk_detection
python -m src.analytics
```

Or explore interactively:

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

## Testing

```bash
pytest
```

The test suite (22 tests) covers:
- Dataset generation: record count, required columns, reproducibility,
  gender categories, presence of injected missing/anomalous values.
- ETL cleaning: no missing values or duplicates after cleaning, all
  numeric ranges valid, summary counts recorded correctly.
- Risk detection: score bounds [0, 100], correct classification against
  configurable thresholds, weights sum to 1.0, monotonicity with respect
  to attendance.
- **Data-leakage guarantees**: `final_exam_score` is excluded from the
  pre-exam feature list; the risk-scoring functions' source code never
  references it; and — most importantly — a behavioral test proves
  `risk_score` is numerically **identical** whether `final_exam_score` is
  left untouched or zeroed out.

## Example Outputs

The generated data, charts, and reports are **committed to this repository**
under `data/`, so you can browse them directly on GitHub without cloning
or running anything. Re-running `python -m src.main` regenerates them
byte-for-byte identically (fixed random seed) if you want to confirm that
locally.

After running `python -m src.main`, `data/reports/` contains:

- `attendance_distribution.png`, `assessment_distribution.png`,
  `performance_distribution.png`
- `attendance_vs_performance.png`, `study_hours_vs_performance.png`,
  `gpa_vs_performance.png`
- `risk_level_distribution.png`, `performance_trend.png`,
  `high_risk_analysis.png`
- `statistical_summary.csv`, `model_performance.csv`,
  `analytical_report.txt`

## Limitations

- The dataset is **entirely synthetic**, generated from a simplified
  single-latent-factor statistical model. It does not capture the full
  complexity of real student populations (e.g., socioeconomic context,
  learning differences, external life events).
- The risk score is a **rule-based heuristic**, not a validated clinical
  or institutional predictive tool.
- The regression models are trained and evaluated only on synthetic data
  and have not been validated against any real institution's data.
- Correlations reported throughout this project reflect the generative
  structure of the synthetic data and must not be interpreted as causal
  claims about real students.

## Ethical Considerations

- Academic data about real students is sensitive. This project uses
  **only synthetic data**; no real student identities, records, or
  institutions are included anywhere in this repository.
- Risk scores produced by this pipeline are intended to **support human
  review** — by advisors, instructors, or student-support staff — not to
  automatically determine any educational outcome (grading, admission,
  discipline, etc.).
- Any model, including the ones here, can encode or amplify bias present
  in its training data. This project does not claim its models are free
  of bias, and users adapting this code to real data should conduct their
  own fairness and bias evaluation.
- This project does not, and cannot, determine whether any student *will*
  fail. It can, at most, identify students exhibiting **patterns
  associated with** academic risk — a materially different and much
  weaker claim.
- Real-world deployment of any similar system would require: informed
  consent and data governance processes, privacy protections (e.g.,
  FERPA/GDPR compliance as applicable), rigorous fairness/bias validation
  on real data, and institutional review/approval before use in any
  decision affecting real students.

## Future Improvements

- Add a hold-out temporal validation split (e.g., simulate multiple
  academic terms) to test risk-score stability over time.
- Explore fairness metrics across the `gender` field, even though it is
  currently unused in scoring/modeling.
- Add SHAP-based explainability for the Random Forest model.
- Add a lightweight dashboard (e.g., Streamlit) on top of the existing
  SQLite database for interactive exploration.
- Expand the synthetic generator with additional realistic variables
  (e.g., course subject, instructor, term).

## Reproducibility

All randomness (data generation, missing-value/anomaly injection, and the
ML train/test split) is controlled by a single seed
(`RANDOM_SEED = 42`, defined in each relevant module). Re-running
`python -m src.main` on an unmodified checkout reproduces all output files
byte-for-byte identically.

## Author

Built as a graduate-school portfolio project demonstrating applied skills
in data engineering, educational data mining, statistical analysis, and
machine learning fundamentals. This is an independent, reproducible
data-science project and is **not** published academic research.
