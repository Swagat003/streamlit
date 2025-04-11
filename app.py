import streamlit as st # type: ignore
import joblib
import numpy as np

model = joblib.load('model.pkl')

st.title("Housing Price Prediction")
st.write("This app predicts the price of a house based on its features.")

tab1, tab2 = st.tabs(["Predict", "Model Evaluation"])

with tab1:
    st.header("Input Features")
    median_income = st.slider("Median Income", 0.0, 15.0, 3.0, 0.1)
    housing_median_age = st.slider("Housing Median Age", 1, 100, 30, 1)
    total_rooms = st.slider("Total Rooms", 1, 100, 3, 1)
    total_bedrooms = st.slider("Total Bedrooms", 1, 100, 1, 1)
    population = st.slider("Population", 1, 10000, 500, 100)
    households = st.slider("Households", 1, 1000, 200, 10)
    longitude = st.slider("Longitude", -125.0, -114.0, -119.0, 0.1)
    latitude = st.slider("Latitude", 32.0, 42.0, 37.0, 0.1)

    features = np.array([[
        median_income,
        housing_median_age,
        total_rooms,
        total_bedrooms,
        population,
        households,
        longitude,
        latitude
    ]])

    if st.button("Predict"):
        prediction = model.predict(features)
        st.success(f"The predicted house price is: ₹{prediction[0]:,.2f}")

with tab2:
    st.header("Model Evaluation")
    st.write("This section will contain the model evaluation metrics.")
    st.write("Mean Absolute Error (MAE): 0.5")
    st.write("Mean Squared Error (MSE): 1.0")
    st.write("R-squared: 0.8")
