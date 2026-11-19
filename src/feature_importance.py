
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"
ASSET_DIR = BASE_DIR / "assets"

ASSET_DIR.mkdir(exist_ok=True)

# Load tuned Random Forest model
model = joblib.load(MODEL_DIR / "random_forest_tuned.joblib")

# Load training data
X_train = pd.read_csv(DATA_DIR / "X_train.csv")

# Extract fitted preprocessing pipeline
preprocessor = model.named_steps["preprocessor"]
random_forest = model.named_steps["model"]

# Get feature names after preprocessing
feature_names = preprocessor.get_feature_names_out()

# Get Random Forest feature importance
importances = random_forest.feature_importances_

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

print("=" * 65)
print("       AUTOVALUE AI - FEATURE IMPORTANCE")
print("=" * 65)

print("\nTOP 20 IMPORTANT FEATURES")
print("-" * 50)

print(importance_df.head(20).to_string(index=False))

# Save complete feature importance
importance_df.to_csv(
    DATA_DIR / "feature_importance.csv",
    index=False
)

# Plot top 15 features
top_features = importance_df.head(15).sort_values(
    by="importance",
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.title("Top 15 Features Influencing Used Car Price")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.tight_layout()

plt.savefig(
    ASSET_DIR / "feature_importance.png",
    dpi=300
)

plt.close()

print("\nChart saved:")
print(ASSET_DIR / "feature_importance.png")

print("\nDetailed results saved:")
print(DATA_DIR / "feature_importance.csv")

print("\nFEATURE IMPORTANCE ANALYSIS COMPLETED")
