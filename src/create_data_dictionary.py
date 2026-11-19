
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

datasets = {
    "Historical CarDekho": (
        BASE_DIR / "data" / "raw" / "Car details v3.csv"
    ),
    "Recent Market Listings": (
        BASE_DIR / "data" / "raw" / "recent_used_listings.parquet"
    )
}

for name, path in datasets.items():

    print("\n" + "=" * 70)
    print(name.upper())
    print("=" * 70)

    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    dictionary = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Non-Missing Values": df.notna().sum().values,
        "Missing Values": df.isna().sum().values,
        "Unique Values": df.nunique().values
    })

    print(dictionary.to_string(index=False))

    output_name = (
        "historical_data_dictionary.csv"
        if "Historical" in name
        else "recent_data_dictionary.csv"
    )

    output_path = BASE_DIR / "data" / "processed" / output_name

    dictionary.to_csv(output_path, index=False)

    print("\nSaved:", output_path)

print("\nDATA DICTIONARIES CREATED SUCCESSFULLY!")
