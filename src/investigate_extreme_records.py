
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "data" / "processed" / "historical_numeric.csv"

df = pd.read_csv(file_path)

print("=" * 65)
print("       AUTOVALUE AI - EXTREME RECORD INVESTIGATION")
print("=" * 65)

# Investigate torque values above 500 Nm
print("\n1. TORQUE VALUES ABOVE 500 Nm")
print("-" * 50)

high_torque = df[df["torque_nm"] > 500]

print("Count:", len(high_torque))

print(
    high_torque[
        ["name", "year", "fuel", "torque", "torque_nm"]
    ].to_string(index=False)
)

# Investigate vehicles driven more than 300,000 km
print("\n2. VEHICLES DRIVEN ABOVE 300,000 KM")
print("-" * 50)

high_km = df[df["km_driven"] > 300000]

print("Count:", len(high_km))

print(
    high_km[
        ["name", "year", "km_driven", "selling_price"]
    ].sort_values("km_driven", ascending=False)
    .to_string(index=False)
)

# Check the suspicious Maruti Zen records
print("\n3. MARUTI ZEN D RECORDS")
print("-" * 50)

zen = df[df["name"].str.contains("Maruti Zen D", case=False, na=False)]

print(
    zen[
        ["name", "year", "selling_price", "torque", "torque_nm"]
    ].to_string(index=False)
)

print("\n" + "=" * 65)
print("       INVESTIGATION COMPLETED")
print("=" * 65)
