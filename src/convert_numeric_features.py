
import pandas as pd
import re
from pathlib import Path

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

input_path = (
    BASE_DIR / "data" / "processed" / "historical_cleaned.csv"
)

output_path = (
    BASE_DIR / "data" / "processed" / "historical_numeric.csv"
)

df = pd.read_csv(input_path)


# --------------------------------------------------
# 2. EXTRACT NUMERICAL VALUES
# --------------------------------------------------

def extract_number(value):
    if pd.isna(value):
        return float("nan")

    match = re.search(r"[-+]?\d*\.?\d+", str(value))

    if match:
        return float(match.group())

    return float("nan")


# --------------------------------------------------
# 3. CONVERT MILEAGE
# --------------------------------------------------

df["mileage_value"] = df["mileage"].apply(extract_number)

df["mileage_unit"] = (
    df["mileage"]
    .str.extract(r"(kmpl|km/kg)", flags=re.IGNORECASE)[0]
    .str.lower()
)


# --------------------------------------------------
# 4. CONVERT ENGINE CAPACITY
# --------------------------------------------------

df["engine_cc"] = df["engine"].apply(extract_number)


# --------------------------------------------------
# 5. CONVERT MAXIMUM POWER
# --------------------------------------------------

df["max_power_bhp"] = df["max_power"].apply(extract_number)


# --------------------------------------------------
# 6. CONVERT TORQUE INTO Nm
# --------------------------------------------------

def convert_torque(value):

    if pd.isna(value):
        return float("nan"), "Missing original value"

    text = str(value).lower().strip()

    number = extract_number(text)

    if pd.isna(number):
        return float("nan"), "No numerical value"

    # Check Nm first because some records contain
    # both Nm and kgm in the same string.

    if "nm" in text:
        return number, "Already in Nm"

    # Convert kgm into Nm.

    if "kgm" in text:

        # Flag ambiguous formats for manual review.
        if "(kgm@" in text and number > 50:
            return float("nan"), "Ambiguous unit - requires review"

        return number * 9.80665, "Converted from kgm to Nm"

    # Special formats where the unit is omitted.
    # Interpret as Nm based on dataset conventions.

    if "@" in text or "/" in text or "(" in text:
        return number, "Assumed Nm - unit omitted"

    return float("nan"), "Unrecognized torque format"


# Apply torque conversion

torque_results = df["torque"].apply(convert_torque)

df["torque_nm"] = torque_results.apply(
    lambda result: result[0]
)

df["torque_conversion_note"] = torque_results.apply(
    lambda result: result[1]
)


# --------------------------------------------------
# 7. HANDLE INVALID ZERO OR NEGATIVE VALUES
# --------------------------------------------------

df.loc[
    df["mileage_value"] <= 0,
    "mileage_value"
] = float("nan")

df.loc[
    df["max_power_bhp"] <= 0,
    "max_power_bhp"
] = float("nan")

df.loc[
    df["torque_nm"] <= 0,
    "torque_nm"
] = float("nan")


# --------------------------------------------------
# 8. DISPLAY CONVERSION RESULTS
# --------------------------------------------------

print("=" * 65)
print("       AUTOVALUE AI - CORRECTED NUMERIC CONVERSION")
print("=" * 65)

print("\nUNUSUAL TORQUE CONVERSIONS")
print("-" * 50)

unusual = df[
    df["torque_conversion_note"] == "Assumed Nm - unit omitted"
]

print(
    unusual[
        ["torque", "torque_nm", "torque_conversion_note"]
    ]
    .drop_duplicates()
    .to_string(index=False)
)


# --------------------------------------------------
# 9. DISPLAY MISSING VALUES
# --------------------------------------------------

print("\nCONVERSION SUMMARY")
print("-" * 50)

columns = [
    "mileage_value",
    "engine_cc",
    "max_power_bhp",
    "torque_nm"
]

print("\nMissing values:")
print(df[columns].isnull().sum())


# --------------------------------------------------
# 10. TORQUE CONVERSION BREAKDOWN
# --------------------------------------------------

print("\nTORQUE CONVERSION NOTES")
print("-" * 50)

print(df["torque_conversion_note"].value_counts())


# --------------------------------------------------
# 11. SAVE FINAL NUMERICAL DATASET
# --------------------------------------------------

df.to_csv(output_path, index=False)

print("\nFINAL DATASET SHAPE:", df.shape)

print("\nSaved dataset to:")
print(output_path)

print("\n" + "=" * 65)
print("       CORRECTED CONVERSION COMPLETED")
print("=" * 65)
