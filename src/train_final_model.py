
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from preprocessing import create_preprocessor

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

print("=" * 65)
print("       AUTOVALUE AI - FINAL MODEL TRAINING")
print("=" * 65)

# Load training data only
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze()

print("\nTraining records:", len(X_train))
print("Input features:", len(X_train.columns))

# Tuned Random Forest parameters
final_rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features=0.7,
    random_state=42,
    n_jobs=2
)

# Complete preprocessing + model pipeline
final_pipeline = Pipeline(steps=[
    ("preprocessor", create_preprocessor()),
    ("model", final_rf)
])

print("\nTraining final Random Forest model...")

final_pipeline.fit(X_train, y_train)

# Save complete pipeline
model_path = MODEL_DIR / "autovalue_final_model.joblib"

joblib.dump(final_pipeline, model_path)

print("\nFinal model saved successfully:")
print(model_path)

print("\nFINAL MODEL TRAINING COMPLETED")
