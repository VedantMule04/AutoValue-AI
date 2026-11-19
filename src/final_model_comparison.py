
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"
ASSET_DIR = BASE_DIR / "assets"

ASSET_DIR.mkdir(exist_ok=True)

# Load test data
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

# Models available for comparison
model_files = {
    "Random Forest (Initial)": "random_forest_pipeline.joblib",
    "Random Forest (Tuned)": "random_forest_tuned.joblib",
    "XGBoost (Initial)": "xgboost_pipeline.joblib",
    "XGBoost (Tuned)": "xgboost_tuned.joblib",
    "Gradient Boosting": "gradient_boosting_pipeline.joblib",
    "SVR": "svr_pipeline.joblib"
}

results = []

print("=" * 70)
print("       AUTOVALUE AI - FINAL MODEL COMPARISON")
print("=" * 70)

for model_name, filename in model_files.items():

    model_path = MODEL_DIR / filename

    print(f"\nEvaluating: {model_name}")

    model = joblib.load(model_path)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    rmse = mean_squared_error(
        y_test,
        predictions
    ) ** 0.5

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

# Create comparison table
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="MAE",
    ascending=True
)

print("\n" + "=" * 70)
print("FINAL COMPARISON RESULTS")
print("=" * 70)

print(results_df.to_string(
    index=False,
    formatters={
        "MAE": "{:,.2f}".format,
        "RMSE": "{:,.2f}".format,
        "R2": "{:.4f}".format
    }
))

# Save results
results_df.to_csv(
    DATA_DIR / "final_model_comparison.csv",
    index=False
)

# Plot MAE comparison
plt.figure(figsize=(11, 6))

plt.barh(
    results_df["Model"],
    results_df["MAE"]
)

plt.xlabel("Mean Absolute Error (INR)")
plt.ylabel("Machine Learning Model")
plt.title("AutoValue AI - Model Comparison by MAE")
plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    ASSET_DIR / "model_comparison_mae.png",
    dpi=300
)

plt.close()

# Plot R2 comparison
plt.figure(figsize=(11, 6))

plt.barh(
    results_df["Model"],
    results_df["R2"]
)

plt.xlabel("R2 Score")
plt.ylabel("Machine Learning Model")
plt.title("AutoValue AI - Model Comparison by R2")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    ASSET_DIR / "model_comparison_r2.png",
    dpi=300
)

plt.close()

print("\nFiles saved successfully:")
print(DATA_DIR / "final_model_comparison.csv")
print(ASSET_DIR / "model_comparison_mae.png")
print(ASSET_DIR / "model_comparison_r2.png")

print("\nMODEL COMPARISON COMPLETED")
