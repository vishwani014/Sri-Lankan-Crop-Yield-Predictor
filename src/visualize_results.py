import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

df = pd.read_csv('data/processed/feature_engineered_dataset.csv')

df['Season_Encoded'] = df['season'].map({'Maha': 1, 'Yala': 0})
if 'Season_Encoded' not in df.columns:
    raise ValueError("Season_Encoded column not created. Check 'Season' column values.")

test_df = df[df['Year'] > 2018].copy()

model = joblib.load('models/random_forest_model.pkl')
features = [
    'Sown_Ha', 'Sown_to_Harvest_Ratio', 'rfh_avg', 'r1h_avg', 'r3h_avg', 'rfq', 'Inflation',
     'Prev_Rainfall', 'Season_Encoded', 'Crisis_Indicator',
]
test_df['Predicted_Yield'] = model.predict(test_df[features])

# Plot actual vs. predicted yields
plt.figure(figsize=(10, 6))
for season in ['Maha', 'Yala']:
    season_data = test_df[test_df['season'] == season]
    plt.plot(season_data['Year'], season_data['Avg_Yield_Kg_Ha'], label=f'{season} Actual', marker='o')
    plt.plot(season_data['Year'], season_data['Predicted_Yield'], label=f'{season} Predicted', marker='x', linestyle='--')
plt.title('Actual vs. Predicted Rice Yield (Test Set)')
plt.xlabel('Year')
plt.ylabel('Average Yield (Kg/Ha)')
plt.legend()
plt.savefig('results/actual_vs_predicted.png')
plt.close()

plt.figure(figsize=(10, 6))
for season in ['Maha', 'Yala']:
    season_data = test_df[test_df['season'] == season]
    errors = season_data['Avg_Yield_Kg_Ha'] - season_data['Predicted_Yield']
    sns.kdeplot(errors, label=season, fill=True)
plt.title('Prediction Error Distribution by Season')
plt.xlabel('Error (Kg/Ha)')
plt.ylabel('Density')
plt.legend()
plt.savefig('results/prediction_errors.png')
plt.close()