
import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Input: original dataset
input_path = BASE_DIR / "data" / "raw" / "Car details v3.csv"

# Output: cleaned dataset
output_path = BASE_DIR / "data" / "processed" / "historical_cleaned.csv"

# Load original data
df = pd.read_csv(input_path)

print("=" * 60)
print("       AUTOVALUE AI - HISTORICAL DATA CLEANING")
print("=" * 60)

original_rows = len(df)

# Count exact duplicates
duplicate_count = df.duplicated().sum()

# Remove exact duplicate rows
cleaned_df = df.drop_duplicates().copy()

# Save cleaned dataset
cleaned_df.to_csv(output_path, index=False)

# Display cleaning summary
print("\nCLEANING SUMMARY")
print("-" * 40)
print("Original rows:", original_rows)
print("Exact duplicates removed:", duplicate_count)
print("Rows after cleaning:", len(cleaned_df))
print("Columns retained:", len(cleaned_df.columns))

print("\nMISSING VALUES AFTER DUPLICATE REMOVAL")
print("-" * 40)
print(cleaned_df.isnull().sum())

print("\nFINAL DATASET SHAPE:", cleaned_df.shape)
print("\nSaved cleaned dataset to:")
print(output_path)

print("\n" + "=" * 60)
print("       CLEANING COMPLETED SUCCESSFULLY")
print("=" * 60)
