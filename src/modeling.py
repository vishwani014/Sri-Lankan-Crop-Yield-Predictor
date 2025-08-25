import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import joblib

df = pd.read_csv('data/processed/feature_engineered_dataset.csv')

df['Season_Encoded'] = df['season'].map({'Maha': 1, 'Yala': 0})
if 'Season_Encoded' not in df.columns:
    raise ValueError("Season_Encoded column not created. Check 'Season' column values.")

features = [
    'Sown_Ha', 'Sown_to_Harvest_Ratio', 'rfh_avg', 'r1h_avg', 'r3h_avg', 'rfq', 'Inflation',
     'Prev_Rainfall', 'Season_Encoded', 'Crisis_Indicator',
]
target = 'Avg_Yield_Kg_Ha'

# Split data
train_df = df[df['Year'] <= 2018]
test_df = df[df['Year'] > 2018]
X_train = train_df[features]
y_train = train_df[target]
X_test = test_df[features]
y_test = test_df[target]

param_grid = {
    'n_estimators': [100, 150, 200],         # Number of trees
    'max_depth': [10, 20, None],              # Max depth of the trees (None means no limit)
    'min_samples_leaf': [1, 2, 4],            # Min samples required at a leaf node
    'max_features': ['sqrt', 'log2']          # Number of features to consider for a split
}


# Train Random Forest
model = RandomForestRegressor(random_state=42)

tscv = TimeSeriesSplit(n_splits=5)

grid_search = GridSearchCV(estimator=model, 
                           param_grid=param_grid, 
                           cv=tscv, 
                           scoring='neg_mean_squared_error', 
                           n_jobs=-1, 
                           verbose=2)

grid_search.fit(X_train, y_train)

print(f"Best Hyperparameters: {grid_search.best_params_}")

model = grid_search.best_estimator_
# model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
test_df['Predicted_Yield'] = y_pred
rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae = mean_absolute_error(y_test, y_pred)
print(f"Overrall RMSE: {rmse:.2f} Kg/Ha")
print(f"Overall MAE: {mae:.2f} Kg/Ha")

# Evaluate by season
for season in ['Maha', 'Yala']:
    season_test = test_df[test_df['season'] == season]
    if len(season_test) > 0:  # Check if season has data
        season_rmse = mean_squared_error(season_test['Avg_Yield_Kg_Ha'], season_test['Predicted_Yield']) ** 0.5
        season_mae = mean_absolute_error(season_test['Avg_Yield_Kg_Ha'], season_test['Predicted_Yield'])
        print(f"{season} RMSE: {season_rmse:.2f} Kg/Ha")
        print(f"{season} MAE: {season_mae:.2f} Kg/Ha")
    else:
        print(f"No test data for {season}")

joblib.dump(model, 'models/random_forest_model.pkl')

# Feature importance
feature_importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

feature_importance.to_csv('results/feature_importance.csv', index=False)
