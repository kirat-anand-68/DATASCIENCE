import streamlit as st
import numpy as np
import pickle

# Load Model
with open("heart_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💓 Heart Disease Prediction App")
st.write("Enter the patient details to predict heart disease.")

# Feature Inputs
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250)
chol = st.number_input("Cholesterol", min_value=50, max_value=700)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
restecg = st.selectbox("Resting ECG (0, 1, 2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=20, max_value=250)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, format="%.1f")
slope = st.selectbox("Slope (0, 1, 2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)", [1, 2, 3])

# Collect all values
input_values = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])

# Predict button
if st.button("🔍 Predict"):
    prediction = model.predict(input_values)

    if prediction == 0:
        st.success("✅ You Are Healthy. No Heart Disease Detected!")
    else:
        st.error("⚠️ Heart Disease Detected! Please Consult a Doctor.")
