
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "data" / "processed" / "historical_numeric.csv"

df = pd.read_csv(file_path)

print("=" * 65)
print("       AUTOVALUE AI - MISSING VALUE ANALYSIS")
print("=" * 65)

# 1. Missing value count and percentage
print("\n1. MISSING VALUE SUMMARY")
print("-" * 50)

missing_count = df.isnull().sum()
missing_percentage = (missing_count / len(df)) * 100

missing_summary = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing Percentage": missing_percentage.round(2)
})

print(
    missing_summary[missing_summary["Missing Count"] > 0]
    .sort_values("Missing Count", ascending=False)
    .to_string()
)

# 2. Records with multiple missing numerical features
numeric_features = [
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm"
]

missing_per_record = df[numeric_features].isnull().sum(axis=1)

print("\n2. MISSING VALUES PER RECORD")
print("-" * 50)

print("No missing numerical features:", (missing_per_record == 0).sum())
print("One missing feature:", (missing_per_record == 1).sum())
print("Two missing features:", (missing_per_record == 2).sum())
print("Three missing features:", (missing_per_record == 3).sum())
print("Four missing features:", (missing_per_record == 4).sum())

# 3. Mileage unit distribution
print("\n3. MILEAGE UNIT DISTRIBUTION")
print("-" * 50)

print(df["mileage_unit"].value_counts(dropna=False))

# 4. Numerical feature statistics
print("\n4. NUMERICAL FEATURE STATISTICS")
print("-" * 50)

print(
    df[numeric_features + ["selling_price", "km_driven", "year"]]
    .describe()
    .round(2)
    .to_string()
)

# 5. Check for impossible or suspicious values
print("\n5. BASIC VALUE VALIDATION")
print("-" * 50)

checks = {
    "Selling price <= 0": (df["selling_price"] <= 0).sum(),
    "Kilometres driven < 0": (df["km_driven"] < 0).sum(),
    "Engine capacity <= 0": (df["engine_cc"] <= 0).sum(),
    "Maximum power <= 0": (df["max_power_bhp"] <= 0).sum(),
    "Torque <= 0": (df["torque_nm"] <= 0).sum(),
    "Mileage <= 0": (df["mileage_value"] <= 0).sum()
}

for check, count in checks.items():
    print(f"{check}: {count}")

print("\n" + "=" * 65)
print("       MISSING VALUE ANALYSIS COMPLETED")
print("=" * 65)
