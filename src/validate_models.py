
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

# Load training and testing datasets
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

models = {
    "Random Forest": "random_forest_pipeline.joblib",
    "XGBoost": "xgboost_pipeline.joblib",
    "Gradient Boosting": "gradient_boosting_pipeline.joblib",
    "SVR": "svr_pipeline.joblib"
}

results = []

print("=" * 70)
print("       AUTOVALUE AI - MODEL VALIDATION")
print("=" * 70)

for model_name, filename in models.items():

    print(f"\nEvaluating {model_name}...")
    model_path = MODEL_DIR / filename

    model = joblib.load(model_path)

    # Training performance
    train_pred = model.predict(X_train)

    train_mae = mean_absolute_error(y_train, train_pred)
    train_rmse = mean_squared_error(y_train, train_pred) ** 0.5
    train_r2 = r2_score(y_train, train_pred)

    # Test performance
    test_pred = model.predict(X_test)

    test_mae = mean_absolute_error(y_test, test_pred)
    test_rmse = mean_squared_error(y_test, test_pred) ** 0.5
    test_r2 = r2_score(y_test, test_pred)

    # Three-fold cross-validation on training data only
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=3,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2"
        },
        n_jobs=1
    )

    cv_mae = -cv_results["test_mae"].mean()
    cv_rmse = -cv_results["test_rmse"].mean()
    cv_r2 = cv_results["test_r2"].mean()

    results.append({
        "Model": model_name,
        "Train MAE": train_mae,
        "Test MAE": test_mae,
        "Train RMSE": train_rmse,
        "Test RMSE": test_rmse,
        "Train R2": train_r2,
        "Test R2": test_r2,
        "CV MAE": cv_mae,
        "CV RMSE": cv_rmse,
        "CV R2": cv_r2
    })

    print(f"{model_name} validation completed.")

# Create comparison table
results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("       FINAL MODEL VALIDATION RESULTS")
print("=" * 70)

print(results_df.round(4).to_string(index=False))

# Save results
output_path = DATA_DIR / "model_validation_results.csv"
results_df.to_csv(output_path, index=False)

print("\nResults saved at:")
print(output_path)

print("\n" + "=" * 70)
print("       VALIDATION COMPLETED")
print("=" * 70)
