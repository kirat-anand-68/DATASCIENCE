import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

model=joblib.load("best_model.pkl")

st.title("Student Exam Score Predictor")
study_hours=st.slider("study Hours per Day",0.0, 12.0, 2.0)
attendance=st.slider("Attendance Percentage",0.0,100.0,80.0)
mental_health= st.slider("mental Health Rating (1-10)",1, 10, 5)
sleep_hours=st.slider("Sleep Hours Per Night", 0.0, 12.0, 7.0)
Part_time_job= st.selectbox("part-Time Job",["No","Yes"])

ptj_encoded=1 if Part_time_job == "Yes" else 0

if st.button("Predict Exam Score"):
    input_data = np.array([[study_hours, attendance, mental_health, sleep_hours , Part_time_job]])
    prediction = model.predict(input_data)[0]

    prediction = max(0,min(100,prediction))

    st.success(f"Predicted Exam Score: {prediction:.2f}")