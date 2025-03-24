import streamlit as st
import joblib
import numpy as np

# Load model and feature columns
model = joblib.load("cologuard_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.title("Cologuard+ AI Risk Predictor")
st.markdown("Predicts risk of advanced neoplasia in patients with positive Cologuard")

# Input fields
age = st.number_input("Age", min_value=40, max_value=100, value=60)
gender = st.selectbox("Gender", ["Male", "Female"])
ethnicity = st.selectbox("Ethnicity", ["White", "Black", "Asian", "Hispanic", "Other"])
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
wait_time = st.number_input("Wait Time (days from positive test to colonoscopy)", min_value=0, max_value=365, value=30)

# Encode inputs
gender_val = 1 if gender == "Male" else 0
ethnicity_val = {"White": 0, "Black": 1, "Asian": 2, "Hispanic": 3, "Other": 4}.get(ethnicity, 0)

# Create input for model
input_data = np.array([[age, gender_val, ethnicity_val, bmi, wait_time]])

# Predict
if st.button("Predict"):
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]
    
    st.subheader("🔍 Prediction Result:")
    if pred == 1:
        st.error(f"⚠️ High Risk of Advanced Neoplasia\nProbability: {prob*100:.1f}%")
    else:
        st.success(f"✅ Low Risk of Advanced Neoplasia\nProbability: {prob*100:.1f}%")
