
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "data" / "raw" / "recent_used_listings.parquet"

df = pd.read_parquet(file_path)

print("=" * 65)
print("       AUTOVALUE AI - RECENT MARKET ANALYSIS")
print("=" * 65)

# 1. Data source distribution
print("\n1. DATA SOURCES")
print(df["source"].value_counts(dropna=False))

# 2. City distribution
print("\n2. CITY DISTRIBUTION")
print(df["city_name"].value_counts())

# 3. Fuel type distribution
print("\n3. FUEL TYPE DISTRIBUTION")
print(df["fuel_type"].value_counts())

# 4. Transmission distribution
print("\n4. TRANSMISSION DISTRIBUTION")
print(df["transmission"].value_counts())

# 5. Manufacturing year distribution
print("\n5. MANUFACTURING YEAR DISTRIBUTION")
print(df["model_year"].value_counts().sort_index())

# 6. Price statistics
print("\n6. ASKING PRICE STATISTICS")
print(df["price_inr"].describe())

# 7. Top 10 expensive listings
print("\n7. TOP 10 EXPENSIVE CARS")
print(
    df.nlargest(10, "price_inr")[
        ["make", "model", "model_year", "price_inr", "city_name"]
    ].to_string(index=False)
)

# 8. Duplicate listing IDs
print("\n8. DUPLICATE CHECK")
print("Duplicate SKU IDs:", df["sku_id"].duplicated().sum())
print("Duplicate listing URLs:", df["listing_url"].duplicated().sum())

# 9. Date distribution
print("\n9. PRICE DATE DISTRIBUTION")
print(df["price_date"].value_counts())

# 10. Missing value percentages by source
print("\n10. MISSING VALUES BY SOURCE")

for source, group in df.groupby("source"):
    print(f"\nSource: {source}")
    print(
        (group.isnull().mean() * 100)
        .round(2)
        .sort_values(ascending=False)
    )

print("\n" + "=" * 65)
print("       RECENT MARKET ANALYSIS COMPLETED")
print("=" * 65)
