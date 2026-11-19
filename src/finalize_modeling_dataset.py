
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "data" / "processed" / "historical_prepared.csv"
output_path = BASE_DIR / "data" / "processed" / "modeling_dataset_final.csv"

df = pd.read_csv(input_path)

print("=" * 65)
print("       AUTOVALUE AI - FINAL MODELLING DATASET")
print("=" * 65)

original_rows = len(df)

# Correct brand extraction
def extract_brand(name):
    name = str(name).strip()

    if name.startswith("Land Rover "):
        return "Land Rover"

    if name.startswith("Ashok Leyland "):
        return "Ashok Leyland"

    return name.split()[0]

df["brand"] = df["name"].apply(extract_brand)

# Select model features
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

# Remove exact duplicate modelling rows
before_duplicates = len(model_df)
model_df = model_df.drop_duplicates().copy()
removed_duplicates = before_duplicates - len(model_df)

# Verify dataset
print("\n1. DATASET SUMMARY")
print("-" * 50)
print("Original prepared records:", original_rows)
print("Records before duplicate removal:", before_duplicates)
print("Duplicate rows removed:", removed_duplicates)
print("Final records:", len(model_df))
print("Final columns:", len(model_df.columns))

print("\n2. CORRECTED BRAND LABELS")
print("-" * 50)
print(model_df["brand"].value_counts().to_string())

print("\n3. REMAINING DUPLICATES")
print("-" * 50)
print(model_df.duplicated().sum())

print("\n4. MISSING VALUES")
print("-" * 50)
print(model_df.isnull().sum())

# Save a separate final dataset
model_df.to_csv(output_path, index=False)

print("\nSaved to:")
print(output_path)

print("\n" + "=" * 65)
print("       FINAL DATASET CREATED")
print("=" * 65)
