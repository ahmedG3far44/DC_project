import streamlit as st
import pandas as pd
import joblib
import numpy as np


model = joblib.load("svm_diabetes_model.pkl") 

st.title("🩺 Diabetes Prediction System")
st.write("Enter patient metrics below to predict diabetes risk using our SVM model.")

with st.form("patient_data"):
    col1, col2 = st.columns(2)
    
    with col1:
        high_bp = st.selectbox("High Blood Pressure", [0, 1])
        high_chol = st.selectbox("High Cholesterol", [0, 1])
        bmi = st.number_input("BMI", min_value=10.0, max_value=90.0, value=25.0)
        age = st.slider("Age Category (1-13)", 1, 13, 5)
        
    with col2:
        smoker = st.selectbox("Smoker", [0, 1])
        stroke = st.selectbox("History of Stroke", [0, 1])
        heart_disease = st.selectbox("Heart Disease/Attack", [0, 1])
        gen_hlth = st.slider("General Health (1-Excellent to 5-Poor)", 1, 5, 3)


    submit_button = st.form_submit_button(label="Predict Risk")

if submit_button:
    input_data = np.array([[high_bp, high_chol, 1, bmi, smoker, stroke, heart_disease, 
                            1, 1, 1, 0, 1, 0, gen_hlth, 0, 0, 0, 1, age, 4, 5]])
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("⚠️ The model predicts a high risk of Diabetes/Prediabetes.")
    else:
        st.success("✅ The model predicts no current indicators of Diabetes.")