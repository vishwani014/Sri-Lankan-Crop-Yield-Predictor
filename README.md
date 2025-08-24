# Sri-Lankan-Rice-Yield-Predictor

A data science project predicting rice yields in Sri Lanka using weather and market.

Overview
The Sri Lankan Rice Yield Predictor is a data science project designed to predict rice yields (Kg/Ha) for Sri Lanka’s Maha and Yala seasons, addressing food security and agricultural planning in the context of the 2022 economic crisis. Using historical paddy production, rainfall, rice prices, population, and inflation data, the project employs a Random Forest model to forecast season-specific yields. The model is deployed via a Streamlit app, allowing users to input features like rainfall and sown area to predict yields for future seasons (e.g., Maha 2025).

Season-Specific Predictions: Predicts rice yields for Maha (wet season) and Yala (dry season) using features like rainfall, sown area, inflation, and population.
Data Preprocessing: Cleans and merges datasets from paddy production, rainfall, rice prices, population, and inflation, aligned by year and season.
Feature Engineering: Includes lagged yield/rainfall, sown-to-harvested ratio, and a 2021–2022 crisis indicator.
EDA: Visualizes yield trends, correlations, and seasonal differences (e.g., Maha yields ~10–20% higher).
Modeling: Uses Random Forest for robust, non-linear predictions.
Visualization: Plots actual vs. predicted yields and error distributions by season.
Deployment: Streamlit app for interactive yield predictions.

Datasets
The project uses the following preprocessed datasets:

**combined_paddy_seasons.csv: Paddy production data (yield, sown/harvested area) by year and season (1952–2024).
sources: https://www.statistics.gov.lk/Agriculture/StaticalInformation/PaddyStatistics/PaddyExtent_Maha_Season
https://www.statistics.gov.lk/Agriculture/StaticalInformation/PaddyStatistics/PaddyExtent_Yala_Season
**aggregated_rice_prices.csv: Rice prices (LKR/Kg) by year and season.
source: https://data.humdata.org/dataset/wfp-food-prices-for-sri-lanka
**cleaned_rainfall.csv: Rainfall metrics (e.g., average rainfall) by year and season.
source: https://data.humdata.org/dataset/lka-rainfall-subnational
**cleaned_population.csv: Sri Lankan population and growth rate by year.
source: https://data.worldbank.org
\*\*cleaned_inflation.csv: Annual inflation rates (1960–2024).
source: https://data.worldbank.org

Project Structure
Sri-Lankan-Crop-Yield-Predictor/
├── data/
│ ├── processed/
│ │ ├── combined_yield_data.csv
│ │ ├── seasonal_rice_prices.csv
│ │ ├── seasonal_rainfall.csv
│ │ ├── population.csv
│ │ ├── inflation.csv
│ │ ├── feature_engineered_dataset.csv
| | |-- merged_data.csv
├── src/
│ ├── preprocessing.py
│ ├── eda.py
│ ├── feature_engineering.py
│ ├── modeling.py
│ ├── visualize_results.py
├── models/
│ ├── random_forest_model.pkl
├── results/
│ ├── feature_importance.csv
│ ├── actual_vs_predicted.png
│ ├── prediction_errors.png
| |-- app.py
├── requirements.txt
├── README.md

Setup Instructions

Clone the Repository:
git clone https://github.com/your-username/Sri-Lankan-Crop-Yield-Predictor.git
cd Sri-Lankan-Crop-Yield-Predictor

Create Virtual Environment (optional but recommended):
python -m venv .conda
source .conda/bin/activate # On macOS/Linux
.conda\Scripts\activate # On Windows

Install Dependencies:
pip install -r requirements.txt

Requirements: pandas, scikit-learn, matplotlib, seaborn, streamlit, joblib.

Run the Pipeline:
python src/preprocessing.py
python src/eda.py
python src/feature_engineering.py
python src/modeling.py
python src/visualize_resultsc.py
streamlit run app.py

Key Results

Model Performance (Test Set: 2019–2024):
Overall RMSE: ~300–500 Kg/Ha
Maha RMSE: ~250–350 Kg/Ha
Yala RMSE: ~300–400 Kg/Ha

Key Features: Rainfall (rfh_avg), previous yield, inflation (notable in 2022 crisis), season.
Visualizations:
actual_vs_predicted_by_season.png: Actual vs. predicted yields for Maha/Yala.
prediction_errors_by_season.png: Error distributions, showing Maha’s higher accuracy.

Usage

Training/Evaluation: Run modeling.py to train the Random Forest model and evaluate season-specific performance.
Visualization: Run visualize_results.py to generate plots in results/.
Prediction: Use the Streamlit app (app_season_specific.py) to input features (e.g., year, season, rainfall, sown area) and predict yields for future seasons (e.g., Maha 2025).

Future Improvements

Improve the accuracy of model predictions
Incorporate district-level data (e.g., Ampara, Anuradhapura) for regional predictions.
Add time-series models (e.g., SARIMA) for seasonal forecasting.
Include additional features (e.g., fertilizer use, NDVI from satellite data).
Use SHAP values for enhanced model explainability.

License: This project is licensed under the MIT License.
Contact: For questions or collaboration, contact Vishwani Vilochana at vishwani2002@gmail.com or via LinkedIn.
