
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "data" / "processed" / "modeling_dataset_final.csv"
output_dir = BASE_DIR / "data" / "processed"

df = pd.read_csv(input_path)

print("=" * 65)
print("       AUTOVALUE AI - TRAIN TEST SPLIT")
print("=" * 65)

# Separate input features and target
X = df.drop(columns=["selling_price"])
y = df["selling_price"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Save the four separate files
X_train.to_csv(output_dir / "X_train.csv", index=False)
X_test.to_csv(output_dir / "X_test.csv", index=False)

y_train.to_csv(output_dir / "y_train.csv", index=False)
y_test.to_csv(output_dir / "y_test.csv", index=False)

print("\n1. DATASET SPLIT")
print("-" * 50)
print("Original records:", len(df))
print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\n2. TARGET DISTRIBUTION")
print("-" * 50)
print("Training target mean:", round(y_train.mean(), 2))
print("Testing target mean:", round(y_test.mean(), 2))

print("\n3. FEATURE SHAPES")
print("-" * 50)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

print("\n4. SAVED FILES")
print("-" * 50)
print("X_train.csv")
print("X_test.csv")
print("y_train.csv")
print("y_test.csv")

print("\n" + "=" * 65)
print("       TRAIN TEST SPLIT COMPLETED")
print("=" * 65)
