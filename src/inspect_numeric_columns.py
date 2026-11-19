
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "data" / "processed" / "historical_cleaned.csv"

df = pd.read_csv(file_path)

columns = ["mileage", "engine", "max_power", "torque"]

print("=" * 65)
print("       AUTOVALUE AI - NUMERIC COLUMN INSPECTION")
print("=" * 65)

for column in columns:
    print(f"\n{'-' * 50}")
    print(f"COLUMN: {column.upper()}")
    print("-" * 50)

    print("Data type:", df[column].dtype)
    print("Missing values:", df[column].isnull().sum())

    print("\nSample values:")
    print(df[column].dropna().head(10).to_list())

    print("\nMost frequent formats:")
    print(df[column].value_counts().head(10).to_string())

print("\n" + "=" * 65)
print("       INSPECTION COMPLETED")
print("=" * 65)
