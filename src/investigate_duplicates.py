
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "data" / "raw" / "Car details v3.csv"

df = pd.read_csv(file_path)

print("=" * 65)
print("       AUTOVALUE AI - DUPLICATE INVESTIGATION")
print("=" * 65)

# Total records
print("\n1. TOTAL RECORDS:", len(df))

# Exact duplicate records
print("\n2. EXACT DUPLICATES")
print("Duplicate count:", df.duplicated().sum())

# Records involved in duplicate groups
duplicate_rows = df[df.duplicated(keep=False)]

print("\n3. RECORDS INVOLVED IN DUPLICATE GROUPS")
print("Total rows:", len(duplicate_rows))

# Number of duplicate groups
print("\n4. NUMBER OF DUPLICATE GROUPS")
print(
    duplicate_rows.drop_duplicates().shape[0]
)

# Show repeated vehicles
print("\n5. SAMPLE REPEATED RECORDS")

print(
    duplicate_rows[
        ["name", "year", "selling_price", "km_driven"]
    ].head(20).to_string(index=False)
)

# Repeated car names
print("\n6. MOST FREQUENT CAR NAMES")

print(
    df["name"].value_counts().head(15)
)

# Duplicate percentage
percentage = (
    df.duplicated().sum() / len(df)
) * 100

print("\n7. DUPLICATE PERCENTAGE")
print(f"{percentage:.2f}%")

print("\n" + "=" * 65)
print("       INVESTIGATION COMPLETED")
print("=" * 65)
