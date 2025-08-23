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

def preprocess_price(price_path, output_path):
    prices_data = pd.read_csv(price_path)

    # Filter rows where 'commodity' contains "rice"
    riceprice_data = prices_data[prices_data['commodity'].str.contains('rice', case=False, na=False)].copy()

    # Handle data types and cleaning
    riceprice_data['date'] = pd.to_datetime(riceprice_data['date'], errors='coerce')

    numeric_columns = ['latitude', 'longitude', 'price', 'usdprice']
    for col in numeric_columns:
        riceprice_data[col] = pd.to_numeric(riceprice_data[col], errors='coerce')
    riceprice_data = riceprice_data.dropna(subset=['date', 'price'])

    # Handle missing values in other columns
    # Categorical columns
    categorical_cols = ['admin1', 'admin2', 'market', 'category', 'commodity', 'unit', 'priceflag', 'pricetype', 'currency']
    riceprice_data[categorical_cols] = riceprice_data[categorical_cols].fillna('Unknown')

    # numeric columns
    riceprice_data['market_id'] = riceprice_data['market_id'].fillna('Unknown')
    riceprice_data['commodity_id'] = riceprice_data['commodity_id'].fillna('Unknown')

    riceprice_data['longitude'] = riceprice_data['longitude'].fillna(riceprice_data['longitude'].median())
    riceprice_data['latitude'] = riceprice_data['latitude'].fillna(riceprice_data['latitude'].median())

    riceprice_data = riceprice_data.drop_duplicates()

    riceprice_data['year'] = riceprice_data['date'].dt.year
    riceprice_data['month'] = riceprice_data['date'].dt.month
    riceprice_data['season'] = riceprice_data['month'].apply(
        lambda m: 'Maha' if m in [9,10, 11, 12, 1, 2, 3] else ('Yala' if m in [4,5, 6, 7, 8] else 'Unknown') 
    )

    agg_prices = riceprice_data.groupby(['year', 'season', 'commodity']).agg({
        'price': 'mean',
        'usdprice': 'mean'
    }).reset_index()

    agg_prices.rename(columns={
        'price': 'avg_price_lkr',
        'usdprice': 'avg_price_usd'
    }, inplace=True)

    agg_prices = agg_prices.sort_values(['year', 'season', 'commodity'])

    agg_prices.to_csv(output_path, index=False)
    print(f"Successfully created {output_path}")
    print(f"Original rows: {len(riceprice_data)}, Aggregated rows: {len(agg_prices)}")
    return agg_prices

def preprocess_rainfall(rainfall_path, output_path):
    rainfall_data = pd.read_csv(rainfall_path)

    # Handle data types
    rainfall_data['date'] = pd.to_datetime(rainfall_data['date'], format="%Y-%m-%d", errors='coerce')

    numeric_cols = ['n_pixels', 'rfh', 'rfh_avg', 'r1h', 'r1h_avg', 'r3h', 'r3h_avg', 'rfq', 'r1q', 'r3q']
    for col in numeric_cols:
        rainfall_data[col] = pd.to_numeric(rainfall_data[col], errors='coerce')

    rainfall_data = rainfall_data[rainfall_data['version'] == 'final']

    # Drop rows with missing 'date' or key rainfall metrics
    rainfall_data = rainfall_data.dropna(subset=['date', 'rfh', 'rfh_avg'])

    rainfall_data[numeric_cols] = rainfall_data[numeric_cols].fillna(rainfall_data[numeric_cols].median())

    # Filter for Sri Lanka-specific data
    rainfall_data = rainfall_data[rainfall_data['PCODE'].str.startswith('LK', na=False)]

    # DErived Columns
    rainfall_data['year'] = rainfall_data['date'].dt.year
    rainfall_data['month'] = rainfall_data['date'].dt.month
    rainfall_data['season'] = rainfall_data['month'].apply(
        lambda m: 'Maha' if m in [9,10, 11, 12, 1, 2, 3] else ('Yala' if m in [4,5, 6, 7, 8] else 'Unknown') 
    )

    agg_rainfall = rainfall_data.groupby(['year', 'season']).agg({
        'rfh': 'mean',
        'rfh_avg': 'mean',
        'r1h': 'mean',
        'r1h_avg': 'mean',
        'r3h': 'mean',
        'r3h_avg': 'mean',
        'rfq': 'mean'
    }).reset_index()

    agg_rainfall = agg_rainfall.drop_duplicates()

    agg_rainfall = agg_rainfall.sort_values(['year', 'season'])

    agg_rainfall.to_csv(output_path, index=False)
    print(f"Successfully created {output_path}")
    print(f"Original rows: {len(rainfall_data)}, Aggregated rows: {len(agg_rainfall)}")
    return agg_rainfall

def preprocess_paddy_maha_season(maha_season_path, output_path):
    maha_data = pd.read_excel(maha_season_path, sheet_name='Maha Season', skiprows=3)

    # drop column A(empty column)
    maha_data.dropna(axis=1, how='all', inplace=True)

    # renamed for clarity
    maha_data.columns = ['Year', 'Sown_Acres', 'Sown_Ha', 'Harvested_Acres', 'Harvested_Ha', 'Avg_Yield_Bushels_Acre', 'Avg_Yield_Kg_Ha', 'Production_Bushels', 'Production_Mt']

    # dropped second row with units
    maha_data = maha_data.drop(index=0).reset_index(drop=True)

    # process year columns
    maha_data['Year'] = maha_data['Year'].str.extract(r'(\d{4})/(\d{2})')[0].astype(int)
    # maha_data['Year'] = maha_data['Year'].apply(lambda
    #                                                     x: x + 1900 if x >=50 else x + 2000)
    # maha_data['Year'] = maha_data['Year'].astype(int)

    maha_data['season'] = 'Maha'

    numeric_cols = ['Sown_Acres', 'Sown_Ha', 'Harvested_Acres', 'Harvested_Ha', 'Avg_Yield_Bushels_Acre', 'Avg_Yield_Kg_Ha', 'Production_Bushels', 'Production_Mt']
    for col in numeric_cols:
       maha_data[col] = pd.to_numeric(maha_data[col], errors='coerce')

    # Handle missing values
    maha_data = maha_data.dropna(subset=['Avg_Yield_Kg_Ha', 'Production_Mt', 'Year'])
    maha_data[numeric_cols] = maha_data[numeric_cols].fillna(maha_data[numeric_cols].median())

    # Feature engineering
    # Sown to harvest ratio
    maha_data['Sown_to_Harvest_Ratio'] = maha_data['Harvested_Ha'] / maha_data['Sown_Ha']

    maha_data = maha_data.drop_duplicates()

    maha_data = maha_data.sort_values(['Year'])

    maha_data.to_csv(output_path, index=False)
    print(f"Successfully created {output_path}")
    return maha_data

def preprocess_paddy_yala_season(yala_seasin_path, output_path):
    yala_data = pd.read_excel(yala_seasin_path, sheet_name='Yala Season', skiprows=3)

    # drop column A(empty column)
    yala_data.dropna(axis=1, how='all', inplace=True)

    # renaming
    yala_data.columns = [
        'Year', 'Sown_Acres', 'Sown_Ha', 'Harvested_Acres', 'Harvested_Ha',
        'Avg_Yield_Bushels_Acre', 'Avg_Yield_Kg_Ha', 'Production_Bushels', 'Production_Mt'
    ]

    # dropped second row with units
    yala_data = yala_data.drop(index=0).reset_index(drop=True)

    # clean year column
    yala_data['Year'] = pd.to_numeric(yala_data['Year'], errors='coerce').astype('Int64')

    yala_data['season'] = 'Yala'

    numeric_cols = ['Sown_Acres', 'Sown_Ha', 'Harvested_Acres', 'Harvested_Ha', 'Avg_Yield_Bushels_Acre', 'Avg_Yield_Kg_Ha', 'Production_Bushels', 'Production_Mt'] 
    for col in numeric_cols:
        yala_data[col] = pd.to_numeric(yala_data[col], errors='coerce')

    # Handle missing values
    yala_data = yala_data.dropna(subset=['Avg_Yield_Kg_Ha', 'Production_Mt', 'Year'])
    yala_data[numeric_cols] = yala_data[numeric_cols].fillna(yala_data[numeric_cols].median())

    # Feature engineering
    # Sown to harvest ratio
    yala_data['Sown_to_Harvest_Ratio'] = yala_data['Harvested_Ha'] / yala_data['Sown_Ha']

    yala_data = yala_data.drop_duplicates()

    yala_data = yala_data.sort_values(['Year'])

    yala_data.to_csv(output_path, index=False)
    print(f"Successfully created {output_path}")
    return yala_data

def merge_seasonal_data(maha_path, yala_path, output_path):
    maha_season_data = pd.read_csv(maha_path)
    yala_season_data = pd.read_csv(yala_path)

    # concatenate
    combined_yield_data = pd.concat([maha_season_data, yala_season_data], ignore_index=True)

    combined_yield_data['Year'] = combined_yield_data['Year'].astype('Int64')

    numeric_cols = ['Sown_Acres', 'Sown_Ha', 'Harvested_Acres', 'Harvested_Ha', 'Avg_Yield_Bushels_Acre', 'Avg_Yield_Kg_Ha', 'Production_Bushels', 'Production_Mt', 'Sown_to_Harvest_Ratio']
    for col in numeric_cols:
        combined_yield_data[col] = pd.to_numeric(combined_yield_data[col], errors='coerce')

    # sort by year and season
    combined_yield_data['season'] = pd.Categorical(combined_yield_data['season'], categories=['Maha', 'Yala'], ordered=True)
    combined_yield_data = combined_yield_data.sort_values(['Year', 'season'])

    combined_yield_data.to_csv(output_path, index=False)
    print(f"Successfully created {output_path}")
    print(f"Maha rows: {len(maha_season_data)}, Yala rows: {len(yala_season_data)}, Combined rows: {len(combined_yield_data)}")
    return combined_yield_data
 
def preprocess_population_data(population_path, output_path):
    population_data = pd.read_csv(population_path, skiprows=3)

    population_data.to_csv(output_path, index=False)

    # Filter for Sri Lanka
    population_data = population_data[population_data['Country Name'] == 'Sri Lanka'].copy()

    # Reshape from wide to long format using melt
    id_vars = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
    value_vars = [str(year) for year in range(1960, 2025)]
    population_data = population_data.melt(id_vars=id_vars, value_vars=value_vars, var_name='Year', value_name='Population')

    # Clean and convert data types
    population_data['Year'] = pd.to_numeric(population_data['Year'], errors='coerce').astype('Int64')
    population_data['Population'] = pd.to_numeric(population_data['Population'], errors='coerce')

    # handle missing values
    population_data = population_data.dropna(subset=['Year', 'Population'])

    # Feature Engineering
    population_data = population_data.sort_values('Year')
    population_data['Population_Growth_Rate'] = population_data['Population'].pct_change()
    population_data['Population_Growth_Rate'] = population_data['Population_Growth_Rate'].fillna(0)

    # remove unwanted columns
    population_data = population_data[['Year', 'Population', 'Population_Growth_Rate']].copy()

    population_data = population_data.drop_duplicates()

    population_data = population_data.sort_values('Year')

    population_data.to_csv(output_path, index=False)
    print(f"Successfully created {output_path}")
    return population_data

if __name__ == "__main__":
    # preprocess_data(
    #     "data/raw/weather_current.csv",
    #     "data/raw/market_prices.csv",
    #     "data/raw/ndvi.csv",
    #     "data/processed/merged_data.csv"
    # )

    preprocess_price(
        "data/raw/prices.csv",
        "data/processed/seasonal_rice_prices.csv"
    )

    preprocess_rainfall(
        "data/raw/rainfall.csv",
        "data/processed/seasonal_rainfall.csv"
    )

    preprocess_paddy_maha_season(
        "data/raw/Paddy_Maha_Season.xlsx",
        "data/processed/yeild_maha_season.csv"
    )

    preprocess_paddy_yala_season(
        "data/raw/Paddy_Yala_Season.xlsx",
        "data/processed/yeild_yala_season.csv"
    )

    merge_seasonal_data(
        "data/processed/yeild_maha_season.csv",
        "data/processed/yeild_yala_season.csv",
        "data/processed/combined_yield_data.csv"
    )

    preprocess_population_data(
        "data/raw/population.csv",
        "data/processed/population.csv")