import pandas as pd

df = pd.read_csv('data/processed/merged_data.csv')

# Lagged features (previous season's yield and rainfall)
df = df.sort_values(['Year', 'season'])
df['Prev_Yield'] = df['Avg_Yield_Kg_Ha'].shift(1)
df['Prev_Rainfall'] = df['rfh_avg'].shift(1)

# Price-Yield Ratio
df['Price_Yield_Ratio'] = df['avg_price_lkr'] / df['Avg_Yield_Kg_Ha']

# Crisis indicator (2021-2022 economic crisis)
df['Crisis_Indicator'] = df['Year'].isin([2021, 2022, 2023]).astype(int)


df[['Prev_Yield', 'Prev_Rainfall', 'Price_Yield_Ratio']] = df[
    ['Prev_Yield', 'Prev_Rainfall', 'Price_Yield_Ratio']
].fillna(df[['Prev_Yield', 'Prev_Rainfall', 'Price_Yield_Ratio']].median())

df.to_csv('data/processed/feature_engineered_dataset.csv', index=False)
print("Feature-engineered dataset saved to 'data/processed/feature_engineered_dataset.csv'")