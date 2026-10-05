# Student Performance Predictor

A complete data analysis + machine learning project that predicts a
student's final exam score from study habits and background factors, using
regression models trained and compared in scikit-learn.

---

## 1. Problem Statement

Given a few measurable factors about a student — study hours, attendance,
previous performance, sleep, extracurricular involvement, and parental
support — can we predict their final exam score? This is a classic
**supervised regression** problem: the target (final exam score) is a
continuous number, not a category.

## 2. Dataset

`data/student_performance.csv` — 600 synthetic student records, generated
by `generate_data.py` with realistic, *intentionally not perfectly linear*
relationships (diminishing returns on study hours, a sleep-deprivation
penalty, etc.), plus random noise — so the dataset behaves similarly to
real-world data, without needing an external download.

| Column | Meaning | Range |
|---|---|---|
| `study_hours_per_day` | Average daily study time | 0.5–10 hrs |
| `attendance_percentage` | Class attendance | 40–100% |
| `previous_score` | Prior exam score | 20–100 |
| `sleep_hours` | Average nightly sleep | 3–10 hrs |
| `extracurricular_hours` | Daily extracurricular time | 0–6 hrs |
| `parental_support_level` | Ordinal rating | 1 (low) – 5 (high) |
| `final_exam_score` | **Target** — score to predict | 0–100 |

> To use your own real data instead, just replace `data/student_performance.csv`
> with a CSV that has the same column names, and re-run `train_models.py`.

## 3. Methodology

```
generate_data.py  →  data/student_performance.csv
        ↓
train_models.py:
   1. Load data
   2. EDA: correlation heatmap, feature distributions
   3. Train/test split (80/20)
   4. Scale features (StandardScaler)
   5. Train 5 models: Linear Regression, Ridge, Decision Tree,
      Random Forest, Gradient Boosting
   6. Evaluate each on R², MAE, RMSE
   7. Pick the best model by R², save it + charts
```

## 4. Results

On this dataset, **Linear Regression** came out on top:

| Model | R² | MAE | RMSE |
|---|---|---|---|
| **Linear Regression** | **0.686** | **3.88** | **4.85** |
| Ridge Regression | 0.685 | 3.88 | 4.85 |
| Random Forest | 0.548 | 4.64 | 5.81 |
| Gradient Boosting | 0.548 | 4.63 | 5.81 |
| Decision Tree | 0.345 | 5.65 | 6.99 |

**R²** (coefficient of determination) measures how much of the variation
in exam scores the model explains — 0.686 means the model explains about
69% of the variation; the rest is noise/unmeasured factors (which we
deliberately injected). **MAE** (mean absolute error) means predictions are
off by about 3.9 points on average, out of 100.

**Why did linear regression win over Random Forest/Gradient Boosting?**
This is a genuinely useful talking point for a viva: the underlying
relationship, while not perfectly linear, is still *close enough* to
linear that a simple model generalizes better than complex tree ensembles
on a modest 600-row dataset. Tree models tend to need more data to show
their advantage, and are more prone to overfitting on small datasets —
exactly what the Decision Tree's weak score (0.345) demonstrates. This is
a real, common phenomenon in applied ML, not a mistake in the code.

See `charts/` for all visualizations:
- `correlation_heatmap.png` — which features matter most (previous_score: 0.64, study_hours: 0.37, attendance: 0.31)
- `feature_distributions.png` — sanity check on input data
- `actual_vs_predicted.png` — how close predictions are to reality
- `model_comparison.png` — R² bar chart across all 5 models
- `feature_importance.png` — standardized coefficients (which factors push the score up/down most)

## 5. Project Structure

```
student_performance_project/
├── generate_data.py      # Creates the synthetic dataset
├── train_models.py        # EDA + trains & compares 5 models + saves charts
├── predict.py              # CLI tool: predict a score for one student
├── data/
│   └── student_performance.csv
├── charts/                 # All generated visualizations (PNG)
├── model/
│   ├── best_model.pkl       # Saved trained model + scaler
│   └── model_comparison.csv
└── requirements.txt
```

## 6. Setup & Running

```bash
pip install -r requirements.txt

python generate_data.py     # Step 1: create the dataset
python train_models.py       # Step 2: EDA, train, compare, save best model

# Step 3: predict for a new student
python predict.py --study-hours 6 --attendance 90 --previous-score 75 \
    --sleep-hours 7.5 --extracurricular 1 --parental-support 4
```

## 7. Limitations & Future Scope

- **Synthetic data**: results are illustrative of the *method*, not a claim
  about real students. Swapping in a real dataset (e.g. a school's actual
  records, or a public dataset like the UCI Student Performance dataset)
  is a natural next step — the pipeline is built to handle that with no
  code changes, only a new CSV with matching columns.
- **More features**: a real dataset would likely include things like
  tutoring hours, screen time, or subject difficulty, which could improve
  accuracy further.
- **Classification variant**: the same pipeline could be adapted to predict
  a *pass/fail* or letter grade (classification) instead of a raw score,
  using `LogisticRegression` / `RandomForestClassifier` with only minor
  changes to `train_models.py`.
- **Cross-validation**: the current evaluation uses a single train/test
  split; k-fold cross-validation would give a more robust estimate of each
  model's true performance.
