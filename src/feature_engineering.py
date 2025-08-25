import pandas as pd

df = pd.read_csv('data/processed/merged_data.csv')

# Lagged features (previous season's yield and rainfall)
df = df.sort_values(['Year', 'season'])
df['Prev_Yield'] = df.groupby('season')['Avg_Yield_Kg_Ha'].shift(1)
df['Prev_Rainfall'] = df.groupby('season')['rfh_avg'].shift(1)

# Crisis indicator (2021-2022 economic crisis)
df['Crisis_Indicator'] = df['Year'].isin([1968, 1969, 1970, 1973, 1974, 1981, 1983, 1987, 1988, 1989, 1997, 2001, 2020, 2021, 2022, 2023]).astype(int)

df[['Prev_Yield', 'Prev_Rainfall']] = df[
    ['Prev_Yield', 'Prev_Rainfall']
].fillna(df[['Prev_Yield', 'Prev_Rainfall']].median())

df.to_csv('data/processed/feature_engineered_dataset.csv', index=False)
print("Feature-engineered dataset saved to 'data/processed/feature_engineered_dataset.csv'")