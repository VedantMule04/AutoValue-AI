
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

input_path = (
    BASE_DIR / "data" / "processed" / "historical_prepared.csv"
)

output_dir = BASE_DIR / "assets"
output_dir.mkdir(exist_ok=True)

# Load prepared dataset
df = pd.read_csv(input_path)

sns.set_theme(style="whitegrid")

print("=" * 65)
print("       AUTOVALUE AI - EXPLORATORY DATA ANALYSIS")
print("=" * 65)

# 1. Dataset overview
print("\n1. DATASET OVERVIEW")
print("-" * 45)

print("Total records:", len(df))
print("Total columns:", len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())

# 2. Selling price statistics
print("\n2. SELLING PRICE STATISTICS")
print("-" * 45)

print(df["selling_price"].describe().round(2))

# 3. Most common car brands
print("\n3. MOST COMMON CAR BRANDS")
print("-" * 45)

df["brand"] = df["name"].str.split().str[0]

print(df["brand"].value_counts().head(10))

# 4. Fuel type distribution
print("\n4. FUEL TYPE DISTRIBUTION")
print("-" * 45)

print(df["fuel"].value_counts())

# 5. Transmission distribution
print("\n5. TRANSMISSION DISTRIBUTION")
print("-" * 45)

print(df["transmission"].value_counts())


# --------------------------------------------------
# VISUALIZATION 1: SELLING PRICE DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="selling_price",
    bins=40,
    kde=True
)

plt.title("Distribution of Used Car Selling Prices")
plt.xlabel("Selling Price (INR)")
plt.ylabel("Number of Cars")
plt.tight_layout()

plt.savefig(output_dir / "selling_price_distribution.png", dpi=300)
plt.close()


# --------------------------------------------------
# VISUALIZATION 2: CAR YEAR DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="year",
    bins=25,
    kde=True
)

plt.title("Distribution of Car Manufacturing Years")
plt.xlabel("Manufacturing Year")
plt.ylabel("Number of Cars")
plt.tight_layout()

plt.savefig(output_dir / "car_year_distribution.png", dpi=300)
plt.close()


# --------------------------------------------------
# VISUALIZATION 3: FUEL TYPE DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(9, 5))

sns.countplot(
    data=df,
    x="fuel",
    order=df["fuel"].value_counts().index
)

plt.title("Used Cars by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Number of Cars")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(output_dir / "fuel_type_distribution.png", dpi=300)
plt.close()


# --------------------------------------------------
# VISUALIZATION 4: TRANSMISSION DISTRIBUTION
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="transmission",
    order=df["transmission"].value_counts().index
)

plt.title("Used Cars by Transmission Type")
plt.xlabel("Transmission")
plt.ylabel("Number of Cars")
plt.tight_layout()

plt.savefig(output_dir / "transmission_distribution.png", dpi=300)
plt.close()


# --------------------------------------------------
# COMPLETION
# --------------------------------------------------

print("\nEDA CHARTS SAVED")
print("-" * 45)

for file in [
    "selling_price_distribution.png",
    "car_year_distribution.png",
    "fuel_type_distribution.png",
    "transmission_distribution.png"
]:
    print(output_dir / file)

print("\n" + "=" * 65)
print("       EDA OVERVIEW COMPLETED")
print("=" * 65)
