import streamlit as st
import pandas as pd
import joblib


st.title('House Price Prediction Application')

model = joblib.load('model_pipeline.pkl')

st.sidebar.title('User Input Parameters')

longitude = st.sidebar.slider('Longitude', -124.35, -114.31, -122.23)
latitude = st.sidebar.slider('Latitude', 32.54, 41.95, 37.88)
housing_median_age = st.sidebar.slider('Housing Median Age', 1.0, 52.0, 41.0)
total_rooms = st.sidebar.number_input('Total Rooms', 2, 39320, 2000)
total_bedrooms = st.sidebar.number_input('Total Bedrooms', 1, 6445, 200)
population = st.sidebar.number_input('Population', 3, 35682, 200)
households = st.sidebar.number_input('Households', 1, 6082, 200)
median_income = st.sidebar.slider('Median Income', 0.5, 15.0, 8.3252)
ocean_proximity = st.sidebar.selectbox('Ocean Proximity', ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])

input_data = {
    'longitude': longitude,
    'latitude': latitude,
    'housing_median_age': housing_median_age,
    'total_rooms': total_rooms,
    'total_bedrooms': total_bedrooms,
    'population': population,
    'households': households,
    'median_income': median_income,
    'ocean_proximity': ocean_proximity
}

input_df = pd.DataFrame([input_data])

st.write('User Input Parameters')
st.write(input_df)

prediction = model.predict(input_df)

st.header('Prediction')
st.metric('Price', prediction[0])
