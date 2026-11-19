
import pandas as pd
from pathlib import Path

# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset directory
DATA_DIR = BASE_DIR / "data" / "raw"

# All four datasets
dataset_files = [
    "car data.csv",
    "CAR DETAILS FROM CAR DEKHO.csv",
    "Car details v3.csv",
    "car details v4.csv"
]

print("=" * 70)
print("          AUTOVALUE AI - DATASET COMPARISON")
print("=" * 70)

for filename in dataset_files:

    file_path = DATA_DIR / filename

    print(f"\n\nDATASET: {filename}")
    print("-" * 70)

    try:
        df = pd.read_csv(file_path)

        # Dataset dimensions
        print(f"Total rows: {df.shape[0]}")
        print(f"Total columns: {df.shape[1]}")

        # Column names
        print("\nColumn names:")
        print(df.columns.tolist())

        # Data types
        print("\nData types:")
        print(df.dtypes)

        # Missing values
        print("\nMissing values:")
        print(df.isnull().sum())

        # Duplicate records
        print(f"\nDuplicate rows: {df.duplicated().sum()}")

        # First five records
        print("\nFirst five records:")
        print(df.head())

        # File loading confirmation
        print("\nDataset loaded successfully!")

    except Exception as error:
        print(f"Error reading dataset: {error}")

print("\n\n" + "=" * 70)
print("             COMPARISON COMPLETED")
print("=" * 70)
