
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
input_path = BASE_DIR / "data" / "processed" / "modeling_dataset.csv"

df = pd.read_csv(input_path)

print("=" * 65)
print("       AUTOVALUE AI - MODEL DATA VALIDATION")
print("=" * 65)

print("\n1. DATA TYPES")
print("-" * 50)
print(df.dtypes)

print("\n2. DUPLICATE RECORDS")
print("-" * 50)
print("Exact duplicate rows:", df.duplicated().sum())

print("\n3. UNIQUE CATEGORICAL VALUES")
print("-" * 50)

categorical_columns = [
    "fuel", "seller_type", "transmission", "owner", "brand"
]

for column in categorical_columns:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False).to_string())

print("\n4. NUMERICAL FEATURE RANGES")
print("-" * 50)

numeric_columns = [
    "year", "km_driven", "mileage_value",
    "engine_cc", "max_power_bhp", "torque_nm", "seats"
]

print(df[numeric_columns].describe().round(2))

print("\n5. TARGET VARIABLE CHECK")
print("-" * 50)
print("Zero or negative prices:", (df["selling_price"] <= 0).sum())

print("\n6. MISSING VALUE CHECK")
print("-" * 50)
print(df.isnull().sum())

print("\n" + "=" * 65)
print("       VALIDATION COMPLETED")
print("=" * 65)
