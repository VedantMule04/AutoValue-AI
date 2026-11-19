
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
print("       AUTOVALUE AI - CORRELATION ANALYSIS")
print("=" * 65)

# Numerical features
numeric_columns = [
    "selling_price",
    "year",
    "km_driven",
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm",
    "seats"
]

numeric_df = df[numeric_columns]

# 1. Pearson correlation
pearson_corr = numeric_df.corr(method="pearson")

print("\n1. PEARSON CORRELATION WITH SELLING PRICE")
print("-" * 50)

print(
    pearson_corr["selling_price"]
    .sort_values(ascending=False)
    .round(3)
)

# 2. Spearman correlation
spearman_corr = numeric_df.corr(method="spearman")

print("\n2. SPEARMAN CORRELATION WITH SELLING PRICE")
print("-" * 50)

print(
    spearman_corr["selling_price"]
    .sort_values(ascending=False)
    .round(3)
)

# 3. Pearson correlation heatmap
plt.figure(figsize=(11, 8))

sns.heatmap(
    pearson_corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    center=0
)

plt.title("Pearson Correlation Heatmap - AutoValue AI")
plt.tight_layout()

plt.savefig(
    output_dir / "pearson_correlation_heatmap.png",
    dpi=300
)

plt.close()

# 4. Spearman correlation heatmap
plt.figure(figsize=(11, 8))

sns.heatmap(
    spearman_corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5,
    center=0
)

plt.title("Spearman Correlation Heatmap - AutoValue AI")
plt.tight_layout()

plt.savefig(
    output_dir / "spearman_correlation_heatmap.png",
    dpi=300
)

plt.close()

print("\nCORRELATION CHARTS SAVED")
print("-" * 50)

print(output_dir / "pearson_correlation_heatmap.png")
print(output_dir / "spearman_correlation_heatmap.png")

print("\n" + "=" * 65)
print("       CORRELATION ANALYSIS COMPLETED")
print("=" * 65)
