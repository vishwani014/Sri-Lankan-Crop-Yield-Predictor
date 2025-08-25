import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('models/random_forest_model.pkl')

# Streamlit app
st.title("Sri Lankan Rice Yield Predictor (Season-Specific)")

# Input fields
year = st.number_input("Year", min_value=2025, max_value=2030, value=2025)
season = st.selectbox("Season", ["Maha", "Yala"])
sown_ha = st.number_input("Sown Area (Ha)", min_value=0.0, value=500.0)
sown_to_harvested_ratio = st.number_input("Sown-to-Harvested Ratio", min_value=0.0, max_value=1.0, value=0.95)
rfh_avg = st.number_input("Average Rainfall (mm)", min_value=0.0, value=100.0)

# Prepare input data
input_data = pd.DataFrame({
    'Sown_Ha': [sown_ha],
    'Sown_to_Harvest_Ratio': [sown_to_harvested_ratio],
    'rfh_avg': [rfh_avg],
    'r1h_avg': [rfh_avg],
    'Prev_Rainfall': [rfh_avg],
    'Season_Encoded': [1 if season == 'Maha' else 0]
})

# Predict
if st.button("Predict Yield"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Rice Yield for {season} {year}: {prediction:.2f} Kg/Ha")