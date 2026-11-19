
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "data" / "processed" / "historical_prepared.csv"
output_path = BASE_DIR / "data" / "processed" / "modeling_dataset.csv"

df = pd.read_csv(input_path)

# Extract brand from the car name
df["brand"] = df["name"].str.split().str[0]

# Select features for machine learning
selected_columns = [
    "selling_price",
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner",
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats",
    "brand"
]

model_df = df[selected_columns].copy()

# Remove rows with missing target values
model_df = model_df.dropna(subset=["selling_price"])

# Save separate modelling dataset
model_df.to_csv(output_path, index=False)

print("=" * 65)
print("       AUTOVALUE AI - MODELLING DATASET")
print("=" * 65)

print("\nDataset shape:", model_df.shape)

print("\nSelected features:")
for column in model_df.columns:
    print("-", column)

print("\nMissing values:")
print(model_df.isnull().sum())

print("\nTarget variable statistics:")
print(model_df["selling_price"].describe())

print("\nSaved to:")
print(output_path)

print("\n" + "=" * 65)
print("       MODELLING DATASET CREATED")
print("=" * 65)
