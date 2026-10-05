"""
predict.py
-----------
Loads the trained model (model/best_model.pkl) and predicts a final exam
score for a single student, given via command-line arguments.

Example:
    python predict.py --study-hours 5 --attendance 85 --previous-score 70 \
        --sleep-hours 7 --extracurricular 1.5 --parental-support 4
"""

import argparse
import joblib
import pandas as pd

parser = argparse.ArgumentParser(description="Predict a student's final exam score")
parser.add_argument("--study-hours", type=float, required=True, help="Average study hours per day")
parser.add_argument("--attendance", type=float, required=True, help="Attendance percentage (0-100)")
parser.add_argument("--previous-score", type=float, required=True, help="Previous exam score (0-100)")
parser.add_argument("--sleep-hours", type=float, required=True, help="Average sleep hours per night")
parser.add_argument("--extracurricular", type=float, required=True, help="Extracurricular hours per day")
parser.add_argument("--parental-support", type=int, required=True, choices=[1, 2, 3, 4, 5],
                     help="Parental support level (1=low, 5=high)")
args = parser.parse_args()

bundle = joblib.load("model/best_model.pkl")
model, scaler, features, model_name = bundle["model"], bundle["scaler"], bundle["features"], bundle["model_name"]

row = pd.DataFrame([[args.study_hours, args.attendance, args.previous_score,
                      args.sleep_hours, args.extracurricular, args.parental_support]],
                    columns=features)
row_scaled = scaler.transform(row)
prediction = model.predict(row_scaled)[0]
prediction = max(0, min(100, prediction))

print(f"Model used: {model_name}")
print(f"Predicted final exam score: {prediction:.1f} / 100")
