
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

input_path = (
    BASE_DIR / "data" / "processed" / "historical_prepared.csv"
)

output_dir = BASE_DIR / "assets"
output_dir.mkdir(exist_ok=True)

df = pd.read_csv(input_path)

sns.set_theme(style="whitegrid")

print("=" * 65)
print("       AUTOVALUE AI - PRICE RELATIONSHIP ANALYSIS")
print("=" * 65)

# 1. Price by manufacturing year
print("\n1. MEDIAN PRICE BY MANUFACTURING YEAR")
print("-" * 50)

year_price = df.groupby("year")["selling_price"].median()

print(year_price.tail(15).to_string())


# 2. Price by fuel type
print("\n2. MEDIAN PRICE BY FUEL TYPE")
print("-" * 50)

fuel_price = (
    df.groupby("fuel")["selling_price"]
    .agg(["count", "median", "mean"])
    .round(2)
)

print(fuel_price.to_string())


# 3. Price by transmission
print("\n3. MEDIAN PRICE BY TRANSMISSION")
print("-" * 50)

transmission_price = (
    df.groupby("transmission")["selling_price"]
    .agg(["count", "median", "mean"])
    .round(2)
)

print(transmission_price.to_string())


# 4. Price by owner type
print("\n4. MEDIAN PRICE BY OWNER TYPE")
print("-" * 50)

owner_price = (
    df.groupby("owner")["selling_price"]
    .agg(["count", "median", "mean"])
    .round(2)
)

print(owner_price.to_string())


# --------------------------------------------------
# CHART 1: PRICE VS MANUFACTURING YEAR
# --------------------------------------------------

plt.figure(figsize=(10, 5))

sns.regplot(
    data=df,
    x="year",
    y="selling_price",
    scatter_kws={"alpha": 0.25, "s": 18},
    line_kws={"linewidth": 2},
    lowess=False,
    order=2
)

plt.yscale("log")
plt.title("Relationship Between Manufacturing Year and Selling Price")
plt.xlabel("Manufacturing Year")
plt.ylabel("Selling Price (INR, log scale)")
plt.tight_layout()

plt.savefig(
    output_dir / "price_vs_year.png",
    dpi=300
)
plt.close()


# --------------------------------------------------
# CHART 2: PRICE VS KILOMETRES DRIVEN
# --------------------------------------------------

plt.figure(figsize=(10, 5))

sns.scatterplot(
    data=df,
    x="km_driven",
    y="selling_price",
    alpha=0.35,
    s=25
)

plt.yscale("log")
plt.title("Relationship Between Kilometres Driven and Selling Price")
plt.xlabel("Kilometres Driven")
plt.ylabel("Selling Price (INR, log scale)")
plt.tight_layout()

plt.savefig(
    output_dir / "price_vs_km_driven.png",
    dpi=300
)
plt.close()


# --------------------------------------------------
# CHART 3: PRICE VS ENGINE CAPACITY
# --------------------------------------------------

plt.figure(figsize=(10, 5))

sns.scatterplot(
    data=df,
    x="engine_cc",
    y="selling_price",
    alpha=0.35,
    s=25
)

plt.yscale("log")
plt.title("Relationship Between Engine Capacity and Selling Price")
plt.xlabel("Engine Capacity (CC)")
plt.ylabel("Selling Price (INR, log scale)")
plt.tight_layout()

plt.savefig(
    output_dir / "price_vs_engine.png",
    dpi=300
)
plt.close()


# --------------------------------------------------
# CHART 4: PRICE BY FUEL TYPE
# --------------------------------------------------

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="fuel",
    y="selling_price"
)

plt.yscale("log")
plt.title("Selling Price Distribution by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Selling Price (INR, log scale)")
plt.tight_layout()

plt.savefig(
    output_dir / "price_by_fuel.png",
    dpi=300
)
plt.close()


# --------------------------------------------------
# CHART 5: PRICE BY TRANSMISSION
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="transmission",
    y="selling_price"
)

plt.yscale("log")
plt.title("Selling Price Distribution by Transmission")
plt.xlabel("Transmission Type")
plt.ylabel("Selling Price (INR, log scale)")
plt.tight_layout()

plt.savefig(
    output_dir / "price_by_transmission.png",
    dpi=300
)
plt.close()


# --------------------------------------------------
# CHART 6: PRICE BY OWNER TYPE
# --------------------------------------------------

plt.figure(figsize=(10, 5))

sns.boxplot(
    data=df,
    x="owner",
    y="selling_price"
)

plt.yscale("log")
plt.title("Selling Price Distribution by Ownership")
plt.xlabel("Previous Owner Category")
plt.ylabel("Selling Price (INR, log scale)")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    output_dir / "price_by_owner.png",
    dpi=300
)
plt.close()


# --------------------------------------------------
# COMPLETION
# --------------------------------------------------

print("\nCHARTS SAVED SUCCESSFULLY")
print("-" * 50)

chart_names = [
    "price_vs_year.png",
    "price_vs_km_driven.png",
    "price_vs_engine.png",
    "price_by_fuel.png",
    "price_by_transmission.png",
    "price_by_owner.png"
]

for chart in chart_names:
    print(output_dir / chart)

print("\n" + "=" * 65)
print("       PRICE RELATIONSHIP ANALYSIS COMPLETED")
print("=" * 65)
