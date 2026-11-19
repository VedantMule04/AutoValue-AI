
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "data" / "processed" / "historical_numeric.csv"

df = pd.read_csv(file_path)

checks = {
    "max_power": "max_power_bhp",
    "torque": "torque_nm"
}

print("=" * 65)
print("       AUTOVALUE AI - FAILED CONVERSION INVESTIGATION")
print("=" * 65)

for original, converted in checks.items():

    failed = df[
        df[original].notna() &
        df[converted].isna()
    ]

    print(f"\n{'-' * 50}")
    print("ORIGINAL COLUMN:", original)
    print("FAILED CONVERSIONS:", len(failed))
    print("-" * 50)

    if len(failed) > 0:
        print(failed[original].value_counts().head(30).to_string())
    else:
        print("No failed conversions.")

print("\n" + "=" * 65)
print("       INVESTIGATION COMPLETED")
print("=" * 65)

