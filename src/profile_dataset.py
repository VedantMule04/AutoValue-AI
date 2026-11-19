
import pandas as pd
from pathlib import Path

# Locate the project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load our selected dataset
file_path = BASE_DIR / "data" / "raw" / "Car details v3.csv"

df = pd.read_csv(file_path)

print("=" * 65)
print("       AUTOVALUE AI - SELECTED DATASET PROFILE")
print("=" * 65)

# 1. Dataset dimensions
print("\n1. DATASET SIZE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 2. Column information
print("\n2. COLUMN INFORMATION")
print(df.info())

# 3. Statistical summary
print("\n3. NUMERICAL STATISTICS")
print(df.describe())

# 4. Unique values in categorical columns
print("\n4. CATEGORICAL FEATURE ANALYSIS")

categorical_columns = [
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

for column in categorical_columns:
    print(f"\n{column.upper()}")
    print(df[column].value_counts(dropna=False))

# 5. Target variable analysis
print("\n5. SELLING PRICE ANALYSIS")
print("Minimum price:", df["selling_price"].min())
print("Maximum price:", df["selling_price"].max())
print("Average price:", df["selling_price"].mean())
print("Median price:", df["selling_price"].median())

# 6. Manufacturing year analysis
print("\n6. YEAR ANALYSIS")
print("Oldest manufacturing year:", df["year"].min())
print("Latest manufacturing year:", df["year"].max())
print(df["year"].value_counts().sort_index())

# 7. Missing values
print("\n7. MISSING VALUES")
print(df.isnull().sum())

# 8. Duplicate analysis
print("\n8. DUPLICATE ANALYSIS")
print("Exact duplicate rows:", df.duplicated().sum())

print("\nSample duplicate records:")
print(df[df.duplicated(keep=False)].head(10))

print("\n" + "=" * 65)
print("             PROFILE COMPLETED")
print("=" * 65)
