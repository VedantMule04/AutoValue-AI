
import pandas as pd
import joblib

from pathlib import Path
from xgboost import XGBRegressor

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from preprocessing import create_preprocessor


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

# Load data
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

print("=" * 65)
print("       AUTOVALUE AI - XGBOOST TUNING")
print("=" * 65)

# Create pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", create_preprocessor(scale_numeric=False)),
        ("model", XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=2
        ))
    ]
)

# Search space
parameter_grid = {
    "model__n_estimators": [100, 150, 200],
    "model__max_depth": [3, 4, 5, 6],
    "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
    "model__subsample": [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0]
}

# Randomized search
search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=parameter_grid,
    n_iter=8,
    scoring="neg_mean_absolute_error",
    cv=3,
    random_state=42,
    n_jobs=1,
    verbose=1
)

print("\nStarting XGBoost hyperparameter tuning...")
search.fit(X_train, y_train)

print("\nTUNING COMPLETED")

print("\nBEST PARAMETERS")
print("-" * 50)
print(search.best_params_)

print("\nBEST CROSS-VALIDATION MAE")
print("-" * 50)
print(f"Rs. {-search.best_score_:,.2f}")

# Evaluate on untouched test data
best_model = search.best_estimator_
predictions = best_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nTUNED XGBOOST TEST PERFORMANCE")
print("-" * 50)
print(f"MAE: Rs. {mae:,.2f}")
print(f"RMSE: Rs. {rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")

# Save tuned model
model_path = MODEL_DIR / "xgboost_tuned.joblib"
joblib.dump(best_model, model_path)

print("\nTuned model saved at:")
print(model_path)

print("\n" + "=" * 65)
print("       XGBOOST TUNING COMPLETED")
print("=" * 65)
