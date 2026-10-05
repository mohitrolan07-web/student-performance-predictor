"""
generate_data.py
-----------------
Generates a realistic synthetic dataset of student academic performance.

Why synthetic data?
In a real project you'd use a dataset you've collected or downloaded (e.g.
a school's exam records). Here we SIMULATE one with realistic relationships
baked in, so the whole pipeline (EDA -> model training -> evaluation) can be
demonstrated end-to-end without depending on an external file. The
relationships are intentionally not perfectly linear (there's an interaction
between study hours and sleep, plus noise), so tree-based models have a
genuine edge over plain linear regression -- which makes for a more
interesting model comparison later.

Run this first:  python generate_data.py
It writes data/student_performance.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 600  # number of students

study_hours = np.clip(np.random.normal(4.5, 2.0, N), 0.5, 10)
attendance_pct = np.clip(np.random.normal(80, 12, N), 40, 100)
previous_score = np.clip(np.random.normal(65, 15, N), 20, 100)
sleep_hours = np.clip(np.random.normal(6.5, 1.3, N), 3, 10)
extracurricular_hours = np.clip(np.random.exponential(1.5, N), 0, 6)
parental_support = np.random.choice([1, 2, 3, 4, 5], size=N, p=[0.05, 0.15, 0.35, 0.3, 0.15])

# Underlying "true" relationship (nonlinear-ish on purpose):
# - study hours help, but with diminishing returns after ~7 hrs
# - very low sleep hurts even if study hours are high (interaction)
# - attendance and previous score matter a lot
# - extracurricular has a small positive effect up to a point, then a mild penalty
# - parental support has a modest positive effect
diminishing_study = 10 * np.log1p(study_hours)
sleep_penalty = np.where(sleep_hours < 5, (5 - sleep_hours) * 3, 0)
extracurricular_effect = np.where(extracurricular_hours < 3, extracurricular_hours * 1.2, 3.6 - (extracurricular_hours - 3) * 0.8)

final_score = (
    diminishing_study
    + 0.25 * attendance_pct
    + 0.35 * previous_score
    + 1.1 * (sleep_hours - 6)
    - sleep_penalty
    + extracurricular_effect
    + 1.3 * parental_support
    - 0.15 * study_hours * np.where(sleep_hours < 5, 1, 0) * 2  # tired overstudying backfires a bit
    + np.random.normal(0, 5, N)  # noise
)
final_score = np.clip(final_score, 0, 100).round(1)

df = pd.DataFrame({
    "study_hours_per_day": study_hours.round(2),
    "attendance_percentage": attendance_pct.round(1),
    "previous_score": previous_score.round(1),
    "sleep_hours": sleep_hours.round(2),
    "extracurricular_hours": extracurricular_hours.round(2),
    "parental_support_level": parental_support,
    "final_exam_score": final_score,
})

df.to_csv("data/student_performance.csv", index=False)
print(f"Wrote data/student_performance.csv with {len(df)} rows")
print(df.describe().round(2))
