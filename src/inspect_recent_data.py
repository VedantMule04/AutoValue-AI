
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

file_path = BASE_DIR / "data" / "raw" / "recent_used_listings.parquet"

df = pd.read_parquet(file_path)

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nDATA TYPES")
print(df.dtypes)

print("\nFIRST 10 RECORDS")
print(df.head(10).to_string())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDATE-RELATED COLUMNS")
for col in df.columns:
    if any(word in col.lower() for word in ["date", "time", "crawl", "scrape"]):
        print(col, df[col].head(10).tolist())

print("\nDATASET SUMMARY")
print(df.describe(include="all").to_string())

print("\nINSPECTION COMPLETED")
