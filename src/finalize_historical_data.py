
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

input_path = (
    BASE_DIR / "data" / "processed" / "historical_numeric.csv"
)

output_path = (
    BASE_DIR / "data" / "processed" / "historical_prepared.csv"
)

df = pd.read_csv(input_path)

print("=" * 65)
print("       AUTOVALUE AI - FINAL HISTORICAL DATA PREPARATION")
print("=" * 65)

# 1. Flag suspicious torque records
zen_mask = (
    df["name"].str.contains("Maruti Zen D", case=False, na=False)
    & (df["torque_nm"] == 789)
)

df["torque_quality_flag"] = "Valid or unverified"

df.loc[zen_mask, "torque_quality_flag"] = (
    "Suspicious source value"
)

df.loc[zen_mask, "torque_nm"] = float("nan")

# 2. Flag extreme kilometres driven
df["km_quality_flag"] = "Retained"

extreme_km_mask = df["km_driven"] > 1000000

df.loc[extreme_km_mask, "km_quality_flag"] = (
    "Extreme value - excluded from primary training"
)

# 3. Prepare primary modelling dataset
prepared_df = df.loc[~extreme_km_mask].copy()

# 4. Save prepared dataset
prepared_df.to_csv(output_path, index=False)

# 5. Display summary
print("\nDATA PREPARATION SUMMARY")
print("-" * 50)

print("Original cleaned records:", len(df))
print("Extreme KM records flagged:", extreme_km_mask.sum())
print("Records retained:", len(prepared_df))
print("Suspicious torque values flagged:", zen_mask.sum())

print("\nMISSING VALUES IN IMPORTANT FEATURES")
print("-" * 50)

features = [
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm"
]

print(prepared_df[features].isnull().sum())

print("\nQUALITY FLAG SUMMARY")
print("-" * 50)

print("\nTorque:")
print(df["torque_quality_flag"].value_counts())

print("\nKilometres:")
print(df["km_quality_flag"].value_counts())

print("\nFINAL DATASET SHAPE:", prepared_df.shape)

print("\nSaved to:")
print(output_path)

print("\n" + "=" * 65)
print("       HISTORICAL DATA PREPARATION COMPLETED")
print("=" * 65)
