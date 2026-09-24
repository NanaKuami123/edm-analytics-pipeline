# Methodology

This document describes the methodology used throughout the EDM Analytics
Pipeline, from synthetic data generation through statistical analysis and
predictive modeling.

> **All data referenced in this document is synthetic.** No real students,
> institutions, or academic records are used anywhere in this project.

---

## 1. Synthetic Data Generation

### 1.1 Approach

Rather than sampling each variable independently, the generator
(`src/data_generator.py`) builds every student record from a shared latent
variable, `latent_ability`, drawn from a standard normal distribution:

```
latent_ability ~ Normal(0, 1)
```

Each observable variable (attendance, study hours, assessment scores,
previous GPA, final exam score) is then generated as a weighted function of
`latent_ability` plus independent Gaussian noise specific to that variable.
This produces **realistic, imperfect correlations** between variables — for
example, attendance and final exam score are positively correlated, but far
from perfectly so — without hand-coding a deterministic formula that would
make the dataset trivially predictable.

### 1.2 Assumptions

- A single latent "engagement/ability" factor is a simplification of the
  many real factors (motivation, health, socioeconomic context, teaching
  quality, etc.) that influence real student outcomes. It is used here only
  to produce data with plausible statistical structure for demonstration
  purposes.
- Noise terms are Gaussian for simplicity; real educational data is rarely
  perfectly normal.
- All variables are generated independently of any protected characteristic
  other than `gender`, which is included only as a demographic field and is
  **not** used as an input to risk scoring or predictive modeling anywhere
  in this pipeline.

### 1.3 Data Quality Injection

To give the ETL stage meaningful work, the generator deliberately injects:

- **Missing values**: ~2% of values per column (excluding `student_id`),
  via `numpy`'s random generator.
- **Anomalies**: ~1% of records receive an out-of-range value (negative,
  >100, or an extreme outlier multiplier) in a randomly chosen score
  column.
- **Duplicate records**: ~0.3% of rows are duplicated (by `student_id`).
- **Implausible ages**: a small number of records receive ages outside the
  documented 18–30 range.

All injected issues are documented so the ETL cleaning rules can be
verified against them directly.

---

## 2. ETL Methodology

Implemented in `src/etl_pipeline.py`, following a classic Extract →
Transform → Load structure.

**Extract**: read the raw CSV as-is.

**Transform**:
1. Validate that all required columns are present.
2. Remove duplicate records (by `student_id`, keeping the first occurrence).
3. Handle missing values:
   - Numeric columns are imputed with the **column median** (robust to the
     injected outliers).
   - `gender` is imputed with the **column mode**.
   - Rows with a missing `student_id` are dropped, since identity cannot be
     reliably recovered or de-duplicated.
4. Handle invalid values: any value outside its documented valid range
   (see `VALID_RANGES` in `etl_pipeline.py`) is **clipped** to the nearest
   valid boundary. Clipping (rather than dropping) is used because the
   underlying record is otherwise plausible — only a single corrupted field
   needs correction.
5. Normalize dtypes (e.g., `age` as `int`, scores as rounded `float`).

**Load**: the cleaned dataset is written to
`data/processed/clean_student_academic_data.csv` and loaded into a SQLite
database (`data/processed/education_analytics.db`) as the
`student_performance` table, with indexes on `student_id` and
`attendance_rate`.

---

## 3. Feature Engineering

Implemented in `src/feature_engineering.py`. Every feature listed below is
computed **without** using `final_exam_score`, so it is safe for pre-exam
risk detection.

| Feature | Definition |
|---|---|
| `assessment_average` | Mean of `assignment_score`, `quiz_score`, `midterm_score`, `continuous_assessment` |
| `performance_trend` | `(midterm_score + continuous_assessment)/2 − (assignment_score + quiz_score)/2`; negative = declining |
| `attendance_risk` | `100 − attendance_rate`, clipped to [0, 100] |
| `study_efficiency` | `assessment_average / max(study_hours_per_week, 1)`, clipped to [0, 100] |
| `academic_performance_index` | `0.5·assessment_average + 0.3·attendance_rate + 0.2·(previous_gpa/4·100)` — a descriptive composite, distinct from the risk score |
| `previous_performance_category` | `previous_gpa` binned into Low / Medium / High |
| `assessment_performance_category` | `assessment_average` binned into Low / Medium / High |

### Leakage Prevention

`get_pre_exam_feature_columns()` returns the canonical list of columns
approved for pre-exam risk scoring and predictive modeling. The function
`assert_no_leakage()` raises a `ValueError` if `final_exam_score` (or any
other column in `LEAKAGE_COLUMNS`) ever appears in that list. This check
runs automatically in `src/main.py` and is covered by dedicated tests in
`tests/test_risk_detection.py`, including a **behavioral** test that
verifies the risk score is numerically identical whether `final_exam_score`
is left as-is or zeroed out.

---

## 4. Academic Risk Scoring

Implemented in `src/risk_detection.py`.

### 4.1 Formulation

```
Risk Score = 0.20·A + 0.25·P + 0.20·T + 0.15·G + 0.10·S + 0.10·R
```

where each component is independently normalized to the **0–100** scale
before weighting:

| Symbol | Component | Normalization |
|---|---|---|
| `A` | Attendance risk | `100 − attendance_rate`, scaled to [0, 100] |
| `P` | Assessment performance risk | `100 − assessment_average`, scaled to [0, 100] |
| `T` | Performance trend risk | `−performance_trend`, clipped to [−40, 40] and rescaled to [0, 100] |
| `G` | Previous GPA risk | `4.0 − previous_gpa`, scaled to [0, 100] |
| `S` | Study behavior risk | `40 − study_hours_per_week`, scaled to [0, 100] |
| `R` | Recent performance risk | `100 − continuous_assessment`, scaled to [0, 100] |

Because every component is bounded to [0, 100] and the weights sum to 1.0,
the final `risk_score` is mathematically guaranteed to fall in [0, 100].

### 4.2 Classification

| Risk Level | Score Range |
|---|---|
| LOW | 0–29 |
| MODERATE | 30–59 |
| HIGH | 60–100 |

Thresholds are defined once, in `RISK_THRESHOLDS`, and consumed everywhere
else in the codebase — they are not hard-coded elsewhere.

### 4.3 Interpretation

The risk score is a **transparent, rule-based heuristic**, not a
probability of failure or a prediction. It is intended to flag students
exhibiting *patterns associated with* academic risk, to support — not
replace — human review (advisors, instructors). See
[Ethical Considerations](../README.md#ethical-considerations) in the
README for further discussion.

---

## 5. Statistical Methods

Implemented in `src/analytics.py`. For each key pre-exam variable, the
pipeline computes:

- Mean, median, standard deviation, minimum, and maximum.
- Pearson correlation coefficient (and two-sided p-value) against
  `final_exam_score`, via `scipy.stats.pearsonr`.

**Correlation is not causation.** All statistical observations in the
generated report are phrased descriptively (e.g., "is correlated with"),
and the report explicitly notes that any apparent relationships reflect the
correlational structure deliberately built into the synthetic data
generator — not a causal claim about any real population.

---

## 6. Machine Learning Methodology

Implemented in `train_and_evaluate_models()` in `src/analytics.py`.

- **Features**: exactly the pre-exam feature set returned by
  `get_pre_exam_feature_columns()`. `final_exam_score` is never included.
- **Target**: `final_exam_score`.
- **Split**: 80/20 train/test split, `random_state=42` for reproducibility.
- **Models compared**:
  - `LinearRegression` (scikit-learn)
  - `RandomForestRegressor` (200 trees, max depth 8, `random_state=42`)
- **Evaluation metrics**: MAE, RMSE, R² on the held-out test set.

### Limitations

- The model is trained and evaluated **exclusively on synthetic data**
  generated by a simplified latent-factor process. It has not been
  validated against any real student population.
- It must **not** be used, as-is, to make decisions about real students.
- Performance on synthetic data (bounded by the noise injected during
  generation) should not be interpreted as an estimate of real-world
  predictive accuracy.

---

## 7. Evaluation Summary

Running the full pipeline (`python -m src.main`) reproduces:

- `data/reports/statistical_summary.csv` — descriptive statistics and
  correlations.
- `data/reports/model_performance.csv` — MAE / RMSE / R² for each model.
- `data/reports/analytical_report.txt` — a plain-text summary report
  generated directly from the computed data (no fabricated figures).

## 8. Reproducibility

All randomness in this project (data generation, missing-value/anomaly
injection, and the train/test split) is seeded with `RANDOM_SEED = 42`, so
re-running `python -m src.main` on the same code produces identical output
files.
