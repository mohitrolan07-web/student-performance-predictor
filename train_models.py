"""
train_models.py
-----------------
Full pipeline: load data -> EDA -> train multiple regression models ->
compare them -> save charts, comparison table, and the best model.

Run after generate_data.py:  python train_models.py

Outputs:
  charts/correlation_heatmap.png
  charts/feature_distributions.png
  charts/actual_vs_predicted.png
  charts/feature_importance.png
  charts/model_comparison.png
  model/best_model.pkl
  model/model_comparison.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("data/student_performance.csv")
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

FEATURES = ["study_hours_per_day", "attendance_percentage", "previous_score",
            "sleep_hours", "extracurricular_hours", "parental_support_level"]
TARGET = "final_exam_score"

# ---------------------------------------------------------------------------
# 2. Exploratory Data Analysis (EDA)
# ---------------------------------------------------------------------------
# Correlation heatmap -- shows which features relate most strongly to the
# target, and whether any features are redundant with each other.
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("charts/correlation_heatmap.png")
plt.close()

# Feature distributions -- sanity-check that values look realistic
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, col in zip(axes.flat, FEATURES):
    sns.histplot(df[col], kde=True, ax=ax, color="#2F54EB")
    ax.set_title(col)
plt.tight_layout()
plt.savefig("charts/feature_distributions.png")
plt.close()

# ---------------------------------------------------------------------------
# 3. Train / test split + scaling
# ---------------------------------------------------------------------------
X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 4. Train multiple models and compare them
# ---------------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42),
}

results = []
predictions = {}
fitted_models = {}

for name, model in models.items():
    # Linear/Ridge benefit from scaled features; tree models don't need it
    # but aren't hurt by it either, so we scale consistently for simplicity.
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    results.append({"Model": name, "R2": round(r2, 4), "MAE": round(mae, 3), "RMSE": round(rmse, 3)})
    predictions[name] = preds
    fitted_models[name] = model

results_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
print("\nModel comparison (test set):")
print(results_df.to_string(index=False))
results_df.to_csv("model/model_comparison.csv", index=False)

best_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_name]
print(f"\nBest model: {best_name} (R2 = {results_df.iloc[0]['R2']})")

# ---------------------------------------------------------------------------
# 5. Charts for the best model
# ---------------------------------------------------------------------------
# Actual vs Predicted scatter -- points on the diagonal = perfect predictions
plt.figure(figsize=(6, 6))
plt.scatter(y_test, predictions[best_name], alpha=0.6, color="#2F54EB", edgecolor="white")
lims = [min(y_test.min(), predictions[best_name].min()), max(y_test.max(), predictions[best_name].max())]
plt.plot(lims, lims, "--", color="#E08A2B", linewidth=2, label="Perfect prediction")
plt.xlabel("Actual Final Exam Score")
plt.ylabel("Predicted Final Exam Score")
plt.title(f"Actual vs Predicted — {best_name}")
plt.legend()
plt.tight_layout()
plt.savefig("charts/actual_vs_predicted.png")
plt.close()

# Model comparison bar chart (R2 scores)
plt.figure(figsize=(8, 5))
colors = ["#16A34A" if m == best_name else "#8CA0FF" for m in results_df["Model"]]
plt.barh(results_df["Model"], results_df["R2"], color=colors)
plt.xlabel("R² Score (higher is better)")
plt.title("Model Comparison")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("charts/model_comparison.png")
plt.close()

# Feature importance (only meaningful for tree-based models)
if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh", color="#7C5CFA")
    plt.title(f"Feature Importance — {best_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("charts/feature_importance.png")
    plt.close()
elif hasattr(best_model, "coef_"):
    # For linear models, show standardized coefficients instead
    coefs = pd.Series(best_model.coef_, index=FEATURES).sort_values()
    plt.figure(figsize=(8, 5))
    coefs.plot(kind="barh", color="#7C5CFA")
    plt.title(f"Standardized Coefficients — {best_name}")
    plt.xlabel("Coefficient (effect on predicted score)")
    plt.tight_layout()
    plt.savefig("charts/feature_importance.png")
    plt.close()

# ---------------------------------------------------------------------------
# 6. Save the best model + the scaler (needed to transform new inputs later)
# ---------------------------------------------------------------------------
joblib.dump({"model": best_model, "scaler": scaler, "features": FEATURES, "model_name": best_name},
            "model/best_model.pkl")
print("\nSaved best model to model/best_model.pkl")
print("Saved charts to charts/")
