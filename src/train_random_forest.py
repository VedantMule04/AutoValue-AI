
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from preprocessing import create_preprocessor


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# Load training and testing data
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

print("=" * 65)
print("       AUTOVALUE AI - RANDOM FOREST REGRESSION")
print("=" * 65)

# Create preprocessing pipeline
preprocessor = create_preprocessor(
    scale_numeric=False
)

# Define Random Forest model
random_forest = RandomForestRegressor(
    n_estimators=120,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=2
)

# Combine preprocessing and model
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", random_forest)
    ]
)

# Train model
print("\nTraining Random Forest model...")
model_pipeline.fit(X_train, y_train)

print("Training completed.")

# Predict on untouched test data
y_pred = model_pipeline.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-" * 50)
print(f"MAE:  Rs. {mae:,.2f}")
print(f"RMSE: Rs. {rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")

# Save trained pipeline
model_path = MODEL_DIR / "random_forest_pipeline.joblib"

joblib.dump(model_pipeline, model_path)

print("\nModel saved at:")
print(model_path)

print("\n" + "=" * 65)
print("       RANDOM FOREST TRAINING COMPLETED")
print("=" * 65)
