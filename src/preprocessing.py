import pandas as pd
import numpy as np

def preprocess_data(weather_path, prices_path, ndvi_path, output_path):
    weather = pd.read_csv(weather_path)
    prices = pd.read_csv(prices_path)
    ndvi = pd.read_csv(ndvi_path)

    # missing values
    weather["rainfall"].fillna(weather["rainfall"].mean(), inplace=True)
    weather["temperature"].fillna(weather["temperature"].mean(), inplace=True)

    # clean city names
    for df in [weather, prices]:
        df['city'] = df['city'].str.lower()
        df['city'] = df['city'].str.strip()
        df['city'] = df['city'].replace({'Anuradapura': 'Anuradhapura'})

    # standardize date formats
    for df in [weather, prices, ndvi]:
        df['date'] = pd.to_datetime(df['date'], format='mixed').dt.strftime('%Y-%m-%d')
    
    print("--- Debugging Info ---")
    print(f"Unique dates in weather data: {weather['date'].unique()}")
    print(f"Unique dates in price data: {prices['date'].unique()}")
    print("\n")
    print(f"Unique cities in weather data: {weather['city'].unique()}")
    print(f"Unique cities in price data: {prices['city'].unique()}")
    print("----------------------\n")

    merged = pd.merge(weather, prices, on=["date", "city"], how="right")
    if merged.empty:
        print("Warning: The merge operation resulted in an empty DataFrame. No output file will be created.")
    
    merged = pd.merge(merged, ndvi, on=["date"], how="left")
    if merged.empty:
        print("Warning: The merge operation resulted in an empty DataFrame. No output file will be created.")
    
    # handle missing values
    merged['price'] = pd.to_numeric(merged['price'], errors='coerce')
    merged['price'] = merged['price'].ffill()
    merged['ndvi'] = merged['ndvi'].fillna(merged['ndvi'].mean())

    merged["month"] = pd.to_datetime(merged["date"]).dt.month

    if 'price' in merged.columns and 'ndvi' in merged.columns:
        merged["crop_yield"] = round(merged["price"] * 0.1 + merged["ndvi"] * 0.5, 5)
    else:
        print("Warning: 'price' or 'ndvi' column missing after merge. Cannot calculate yield.")


    merged.to_csv(output_path, index=False)
    print(f"Successfully created {output_path}")
    return merged

if __name__ == "__main__":
    preprocess_data(
        "data/raw/weather_current.csv",
        "data/raw/market_prices.csv",
        "data/raw/ndvi.csv",
        "data/processed/merged_data.csv"
    )