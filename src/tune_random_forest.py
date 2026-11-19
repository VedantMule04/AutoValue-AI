
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
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
print("       AUTOVALUE AI - RANDOM FOREST TUNING")
print("=" * 65)

# Pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", create_preprocessor(scale_numeric=False)),
        ("model", RandomForestRegressor(
            random_state=42,
            n_jobs=2
        ))
    ]
)

# Hyperparameter search space
parameter_grid = {
    "model__n_estimators": [100, 150, 200],
    "model__max_depth": [10, 15, 20, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": [0.7, 1.0]
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

print("\nStarting hyperparameter tuning...")
search.fit(X_train, y_train)

print("\nTUNING COMPLETED")

print("\nBEST PARAMETERS")
print("-" * 50)
print(search.best_params_)

print("\nBEST CROSS-VALIDATION MAE")
print("-" * 50)
print(f"Rs. {-search.best_score_:,.2f}")

# Evaluate the selected model on the untouched test set
best_model = search.best_estimator_
predictions = best_model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = mean_squared_error(y_test, predictions) ** 0.5
r2 = r2_score(y_test, predictions)

print("\nTUNED MODEL TEST PERFORMANCE")
print("-" * 50)
print(f"MAE: Rs. {mae:,.2f}")
print(f"RMSE: Rs. {rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")

# Save tuned pipeline
model_path = MODEL_DIR / "random_forest_tuned.joblib"
joblib.dump(best_model, model_path)

print("\nTuned model saved at:")
print(model_path)

print("\n" + "=" * 65)
print("       RANDOM FOREST TUNING COMPLETED")
print("=" * 65)
