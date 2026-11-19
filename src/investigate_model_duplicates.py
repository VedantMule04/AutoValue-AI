
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

model_path = BASE_DIR / "data" / "processed" / "modeling_dataset.csv"
historical_path = BASE_DIR / "data" / "processed" / "historical_prepared.csv"

model_df = pd.read_csv(model_path)
historical_df = pd.read_csv(historical_path)

print("=" * 65)
print("       AUTOVALUE AI - DUPLICATE INVESTIGATION")
print("=" * 65)

# 1. Inspect duplicate modelling records
duplicate_mask = model_df.duplicated(keep=False)
duplicates = model_df[duplicate_mask].copy()

print("\n1. DUPLICATE SUMMARY")
print("-" * 50)
print("Rows involved in duplicate groups:", len(duplicates))
print("Extra duplicate rows:", model_df.duplicated().sum())

print("\n2. DUPLICATE MODELLING RECORDS")
print("-" * 50)

if len(duplicates) > 0:
    print(
        duplicates.sort_values(
            by=["brand", "selling_price"]
        ).to_string(index=False)
    )
else:
    print("No duplicate records found.")

# 2. Inspect original names for rare brands
print("\n3. RARE BRAND INVESTIGATION")
print("-" * 50)

brand_counts = model_df["brand"].value_counts()
rare_brands = brand_counts[brand_counts <= 5].index

for brand in rare_brands:
    print(f"\nBrand label: {brand}")
    print("Record count:", brand_counts[brand])

    names = historical_df.loc[
        historical_df["name"].str.split().str[0] == brand,
        "name"
    ].drop_duplicates()

    print("Original car names:")
    print(names.head(15).to_list())

# 3. Check multi-word brand names
print("\n4. POSSIBLE MULTI-WORD BRAND NAMES")
print("-" * 50)

for name in historical_df["name"].dropna().unique():
    first_word = name.split()[0]

    if first_word in ["Land", "Ashok"]:
        print(name)

print("\n" + "=" * 65)
print("       INVESTIGATION COMPLETED")
print("=" * 65)
