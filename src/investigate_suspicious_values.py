
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "data" / "processed" / "historical_numeric.csv"

df = pd.read_csv(file_path)

print("=" * 70)
print("       AUTOVALUE AI - SUSPICIOUS VALUE INVESTIGATION")
print("=" * 70)

# 1. Zero or negative mileage
print("\n1. ZERO OR NEGATIVE MILEAGE")
print("-" * 55)

bad_mileage = df[df["mileage_value"] <= 0]

print("Count:", len(bad_mileage))

if not bad_mileage.empty:
    print(
        bad_mileage[
            ["name", "year", "fuel", "mileage", "mileage_value"]
        ].to_string(index=False)
    )

# 2. Zero or negative maximum power
print("\n2. ZERO OR NEGATIVE MAX POWER")
print("-" * 55)

bad_power = df[df["max_power_bhp"] <= 0]

print("Count:", len(bad_power))

if not bad_power.empty:
    print(
        bad_power[
            ["name", "year", "max_power", "max_power_bhp"]
        ].to_string(index=False)
    )

# 3. Highest torque values
print("\n3. TOP 15 HIGHEST TORQUE VALUES")
print("-" * 55)

print(
    df.nlargest(15, "torque_nm")[
        ["name", "year", "torque", "torque_nm",
         "torque_conversion_note"]
    ].to_string(index=False)
)

# 4. Highest kilometres driven
print("\n4. TOP 15 HIGHEST KM DRIVEN")
print("-" * 55)

print(
    df.nlargest(15, "km_driven")[
        ["name", "year", "km_driven", "selling_price"]
    ].to_string(index=False)
)

# 5. Mileage unit distribution
print("\n5. MILEAGE BY UNIT")
print("-" * 55)

print(
    df.groupby("mileage_unit", dropna=False)["mileage_value"]
    .agg(["count", "min", "median", "max"])
    .round(2)
    .to_string()
)

print("\n" + "=" * 70)
print("       INVESTIGATION COMPLETED")
print("=" * 70)
