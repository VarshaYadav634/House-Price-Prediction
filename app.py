import streamlit as st
import pickle
import numpy as np

# Load model and features
model, feature_names = pickle.load(open("model/house_price_model.pkl", "rb"))

st.title("🏠 Bangalore House Prices Predictor")

# Inputs
sqft = st.number_input("Enter Total Sqft", min_value=300, max_value=10000, value=1000)
bhk = st.number_input("Enter No of Bedrooms", min_value=1, max_value=10, value=2)
balcony = st.number_input("Enter No of Balconies", min_value=0, max_value=5, value=1)  # ✅ replaced bathrooms
location = st.selectbox("Choose Location", [f for f in feature_names if f not in ["total_sqft","bhk","balcony"]])

# Prediction
if st.button("Predict Price"):
    x = np.zeros(len(feature_names))
    x[feature_names.get_loc("total_sqft")] = sqft
    x[feature_names.get_loc("bhk")] = bhk
    x[feature_names.get_loc("balcony")] = balcony
    if location in feature_names:
        x[feature_names.get_loc(location)] = 1
    
    prediction = model.predict([x])[0]
    st.success(f"Predicted Price: ₹ {prediction:,.2f} Lakhs")
