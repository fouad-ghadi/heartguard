import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
# import joblib  # For your ML Engineer
# import shap    # For your ML Engineer

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Heart Failure Risk Predictor", page_icon="🫀", layout="wide")

# Centrale Casablanca Branding
st.markdown("<h5 style='text-align: right; color: gray;'>Centrale Casablanca - Coding Week</h5>", unsafe_allow_html=True)
st.title("🫀 Clinical Decision Support: Heart Failure Risk")
st.markdown("""
This application assists physicians in accurately predicting the risk of heart failure. 
It utilizes an explainable Machine Learning model trained on clinical records.
""")

st.divider()

# --- 2. DATA COLLECTION FUNCTION ---
def save_doctor_feedback(patient_df, actual_outcome):
    """Saves patient data and the confirmed outcome for future V2.0 model retraining."""
    # Ensure the data directory exists
    os.makedirs('data', exist_ok=True)
    
    feedback_data = patient_df.iloc[0].to_dict()
    feedback_data['DEATH_EVENT'] = actual_outcome
    
    file_path = 'data/clinical_feedback_log.csv'
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=feedback_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(feedback_data)

# --- 3. PATIENT DATA INPUT (SIDEBAR) ---
st.sidebar.header("📋 Input Patient Data")
st.sidebar.markdown("Please enter the clinical parameters below:")

def get_user_input():
    age = st.sidebar.slider("Age", 40, 95, 60)
    ejection_fraction = st.sidebar.slider("Ejection Fraction (%)", 14, 80, 38)
    serum_creatinine = st.sidebar.slider("Serum Creatinine (mg/dL)", 0.5, 9.4, 1.0)
    serum_sodium = st.sidebar.slider("Serum Sodium (mEq/L)", 113, 148, 137)
    time = st.sidebar.slider("Follow-up Period (Days)", 4, 285, 130)
    
    anaemia = st.sidebar.selectbox("Anaemia", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
    diabetes = st.sidebar.selectbox("Diabetes", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
    high_blood_pressure = st.sidebar.selectbox("High Blood Pressure", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
    sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)")
    smoking = st.sidebar.selectbox("Smoking", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")

    user_data = {
        'age': age,
        'anaemia': anaemia,
        'diabetes': diabetes,
        'ejection_fraction': ejection_fraction,
        'high_blood_pressure': high_blood_pressure,
        'serum_creatinine': serum_creatinine,
        'serum_sodium': serum_sodium,
        'sex': sex,
        'smoking': smoking,
        'time': time
    }
    return pd.DataFrame(user_data, index=[0])

patient_data = get_user_input()

st.subheader("Current Patient Profile")
st.dataframe(patient_data, use_container_width=True)

# --- 4. PREDICTION & SESSION STATE LOGIC ---
if 'prediction_made' not in st.session_state:
    st.session_state.prediction_made = False

if st.button("Predict Heart Failure Risk", type="primary"):
    st.session_state.prediction_made = True
    st.session_state.current_patient = patient_data
    
    with st.spinner("Analyzing clinical data..."):
        # ⚠️ ML TEAM: Insert real model prediction here
        prediction_result = 1 # Mock prediction
        
        st.divider()
        st.subheader("📊 Prediction Results")
        
        if prediction_result == 1:
            st.error("⚠️ **HIGH RISK:** The model predicts a high risk of heart failure.")
        else:
            st.success("✅ **LOW RISK:** The model predicts a low risk of heart failure.")
            
        st.subheader("🧠 Model Explainability (SHAP)")
        st.info("(ML Engineer: Insert SHAP summary plot code here).")

# --- 5. THE CLINICAL FEEDBACK LOOP ---
if st.session_state.prediction_made:
    st.divider()
    st.subheader("🧑‍⚕️ Clinical Validation (Continuous Learning)")
    st.write("Help improve the model. What was the actual outcome for this patient?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Confirm: Survived (Class 0)"):
            save_doctor_feedback(st.session_state.current_patient, 0)
            st.success("✅ Case successfully logged to 'data/clinical_feedback_log.csv' for V2.0 retraining!")
            st.session_state.prediction_made = False 
            
    with col2:
        if st.button("Confirm: Deceased (Class 1)"):
            save_doctor_feedback(st.session_state.current_patient, 1)
            st.success("✅ Case successfully logged to 'data/clinical_feedback_log.csv' for V2.0 retraining!")
            st.session_state.prediction_made = False
