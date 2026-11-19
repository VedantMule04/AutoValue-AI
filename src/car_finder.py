
import pandas as pd


def filter_car_listings(
    data,
    min_budget,
    max_budget,
    selected_make="All Companies",
    max_km=1000000,
    selected_fuel="All Fuel Types",
    selected_transmission="All Transmission Types",
    selected_city="All Cities",
    sort_by="Price: Low to High",
):
    """Filter collected used-car listings based on user preferences."""

    df = data.copy()

    # Convert important columns to numeric values
    for column in ["price_inr", "km", "model_year"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Standardise text fields and handle missing values
    text_columns = [
        "make",
        "model",
        "fuel_type",
        "transmission",
        "city_name",
    ]

    for column in text_columns:
        df[column] = df[column].fillna("Unknown").astype(str).str.strip()
        df[column] = df[column].replace("", "Unknown")

    # Remove records without essential information
    df = df.dropna(subset=["price_inr", "km", "make", "model"])

    # Apply budget and kilometre filters
    df = df[
        df["price_inr"].between(min_budget, max_budget)
        & (df["km"] <= max_km)
    ]

    # Apply optional filters
    if selected_make != "All Companies":
        df = df[df["make"] == selected_make]

    if selected_fuel != "All Fuel Types":
        df = df[df["fuel_type"] == selected_fuel]

    if selected_transmission != "All Transmission Types":
        df = df[df["transmission"] == selected_transmission]

    if selected_city != "All Cities":
        df = df[df["city_name"] == selected_city]

    # Sort the matching listings
    if sort_by == "Price: Low to High":
        df = df.sort_values("price_inr", ascending=True)

    elif sort_by == "Price: High to Low":
        df = df.sort_values("price_inr", ascending=False)

    elif sort_by == "Newest First":
        df = df.sort_values("model_year", ascending=False)

    elif sort_by == "Lowest KM":
        df = df.sort_values("km", ascending=True)

    return df.reset_index(drop=True)
