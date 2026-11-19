
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"
ASSET_DIR = BASE_DIR / "assets"

ASSET_DIR.mkdir(exist_ok=True)

print("Loading trained model...")

model = joblib.load(
    MODEL_DIR / "random_forest_tuned.joblib"
)

X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")

preprocessor = model.named_steps["preprocessor"]
random_forest = model.named_steps["model"]

print("Transforming data...")

# Use a small sample to control memory usage
X_sample = X_test.sample(
    n=min(150, len(X_test)),
    random_state=42
)

X_transformed = preprocessor.transform(X_sample)

feature_names = preprocessor.get_feature_names_out()

# Convert sparse matrix to dense if required
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

X_shap = pd.DataFrame(
    X_transformed,
    columns=feature_names
)

print("Creating SHAP explainer...")

explainer = shap.TreeExplainer(random_forest)

shap_values = explainer.shap_values(X_shap)

print("Generating SHAP summary plot...")

plt.figure()

shap.summary_plot(
    shap_values,
    X_shap,
    max_display=15,
    show=False
)

plt.tight_layout()

plt.savefig(
    ASSET_DIR / "shap_summary.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close()

print("SHAP summary chart saved.")

# Mean absolute SHAP values
mean_shap = np.abs(shap_values).mean(axis=0)

shap_importance = pd.DataFrame({
    "feature": feature_names,
    "mean_absolute_shap": mean_shap
}).sort_values(
    by="mean_absolute_shap",
    ascending=False
)

shap_importance.to_csv(
    DATA_DIR / "shap_feature_importance.csv",
    index=False
)

print("\nTOP 15 SHAP FEATURES")
print("-" * 50)

print(shap_importance.head(15).to_string(index=False))

print("\nSHAP analysis completed successfully.")