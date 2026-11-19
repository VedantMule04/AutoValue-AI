
import pandas as pd
import joblib

from pathlib import Path
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from preprocessing import create_preprocessor


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# Load data
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()
y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze()

print("=" * 65)
print("       AUTOVALUE AI - SUPPORT VECTOR REGRESSION")
print("=" * 65)

# Preprocessing with numerical scaling
preprocessor = create_preprocessor(
    scale_numeric=True
)

# Define SVR model
svr_model = SVR(
    kernel="rbf",
    C=100000,
    epsilon=10000,
    gamma="scale"
)

# Build complete pipeline
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", svr_model)
    ]
)

# Train model
print("\nTraining SVR model...")
model_pipeline.fit(X_train, y_train)

print("Training completed.")

# Predictions
y_pred = model_pipeline.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nMODEL PERFORMANCE")
print("-" * 50)
print(f"MAE: Rs. {mae:,.2f}")
print(f"RMSE: Rs. {rmse:,.2f}")
print(f"R2 Score: {r2:.4f}")

# Save model
model_path = MODEL_DIR / "svr_pipeline.joblib"
joblib.dump(model_pipeline, model_path)

print("\nModel saved at:")
print(model_path)

print("\n" + "=" * 65)
print("       SVR TRAINING COMPLETED")
print("=" * 65)
