
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
ASSET_DIR = BASE_DIR / "assets"

ASSET_DIR.mkdir(exist_ok=True)

# Load test data
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

# Load tuned models
rf_model = joblib.load(MODEL_DIR / "random_forest_tuned.joblib")
xgb_model = joblib.load(MODEL_DIR / "xgboost_tuned.joblib")

# Generate predictions
rf_predictions = rf_model.predict(X_test)
xgb_predictions = xgb_model.predict(X_test)

# Create prediction analysis dataframe
results = X_test.copy()

results["actual_price"] = y_test.values
results["rf_prediction"] = rf_predictions
results["xgb_prediction"] = xgb_predictions

results["rf_error"] = (
    results["actual_price"] - results["rf_prediction"]
)

results["rf_absolute_error"] = (
    results["rf_error"].abs()
)

results["rf_percentage_error"] = (
    results["rf_absolute_error"] /
    results["actual_price"] * 100
)

print("=" * 65)
print("       AUTOVALUE AI - PREDICTION ERROR ANALYSIS")
print("=" * 65)

print("\n1. MODEL COMPARISON")
print("-" * 50)

print(
    "Random Forest MAE:",
    round(mean_absolute_error(y_test, rf_predictions), 2)
)

print(
    "XGBoost MAE:",
    round(mean_absolute_error(y_test, xgb_predictions), 2)
)

print(
    "Random Forest R2:",
    round(r2_score(y_test, rf_predictions), 4)
)

print(
    "XGBoost R2:",
    round(r2_score(y_test, xgb_predictions), 4)
)

# Price segment analysis
results["price_segment"] = pd.cut(
    results["actual_price"],
    bins=[0, 300000, 600000, 1000000, float("inf")],
    labels=[
        "Budget",
        "Mid-range",
        "Premium",
        "Luxury"
    ]
)

print("\n2. RANDOM FOREST ERROR BY PRICE SEGMENT")
print("-" * 50)

segment_analysis = results.groupby(
    "price_segment",
    observed=False
).agg(
    records=("actual_price", "count"),
    mean_actual_price=("actual_price", "mean"),
    mean_absolute_error=("rf_absolute_error", "mean")
)

print(segment_analysis.round(2))

# Plot actual vs predicted
plt.figure(figsize=(9, 7))

sns.scatterplot(
    x=results["actual_price"],
    y=results["rf_prediction"],
    alpha=0.5
)

max_value = max(
    results["actual_price"].max(),
    results["rf_prediction"].max()
)

plt.plot(
    [0, max_value],
    [0, max_value],
    linestyle="--"
)

plt.xlabel("Actual Listed Price (INR)")
plt.ylabel("Predicted Listed Price (INR)")
plt.title("Random Forest: Actual vs Predicted Prices")
plt.tight_layout()

plt.savefig(
    ASSET_DIR / "actual_vs_predicted.png",
    dpi=300
)

plt.close()

# Plot residual distribution
plt.figure(figsize=(9, 6))

sns.histplot(
    results["rf_error"],
    bins=40,
    kde=True
)

plt.axvline(0, linestyle="--")
plt.xlabel("Prediction Error (Actual - Predicted)")
plt.ylabel("Number of Cars")
plt.title("Random Forest Prediction Error Distribution")
plt.tight_layout()

plt.savefig(
    ASSET_DIR / "prediction_error_distribution.png",
    dpi=300
)

plt.close()

# Save detailed predictions
results.to_csv(
    DATA_DIR / "prediction_error_analysis.csv",
    index=False
)

print("\nCharts saved:")
print(ASSET_DIR / "actual_vs_predicted.png")
print(ASSET_DIR / "prediction_error_distribution.png")

print("\nDetailed results saved:")
print(DATA_DIR / "prediction_error_analysis.csv")

print("\n" + "=" * 65)
print("       ERROR ANALYSIS COMPLETED")
print("=" * 65)
