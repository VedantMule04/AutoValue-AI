
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

input_path = BASE_DIR / "data" / "processed" / "historical_prepared.csv"

df = pd.read_csv(input_path)

print("=" * 65)
print("       AUTOVALUE AI - MISSING VALUE ANALYSIS")
print("=" * 65)

missing_count = df.isnull().sum()
missing_percentage = (missing_count / len(df)) * 100

summary = pd.DataFrame({
    "Missing_Count": missing_count,
    "Missing_Percentage": missing_percentage.round(2)
})

summary = summary[summary["Missing_Count"] > 0]
summary = summary.sort_values(
    by="Missing_Percentage",
    ascending=False
)

print("\nMISSING VALUE SUMMARY")
print("-" * 50)
print(summary)

print("\nTOTAL MISSING CELLS:", df.isnull().sum().sum())
print("TOTAL ROWS:", len(df))

print("\n" + "=" * 65)
print("       ANALYSIS COMPLETED")
print("=" * 65)
