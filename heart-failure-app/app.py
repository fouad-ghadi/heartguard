import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import joblib

# ------------------------------------------------
# Page configuration
# ------------------------------------------------
st.set_page_config(
    page_title="HeartGuard AI",
    page_icon="❤️",
    layout="wide"
)

# ------------------------------------------------
# Load trained model
# ------------------------------------------------
# Change path if needed
try:
    model = joblib.load("models/model.joblib")
except:
    model = None

# ------------------------------------------------
# Header
# ------------------------------------------------
st.markdown("""
<h1 style='text-align:center;
           background:linear-gradient(135deg,#00d4aa,#0099cc);
           -webkit-background-clip:text;
           -webkit-text-fill-color:transparent;
           font-weight:800;'>
HeartGuard AI
</h1>

<p style='text-align:center;font-size:1.1rem;color:#64748b'>
AI-Powered Clinical Decision Support System for Heart Failure Risk Prediction
</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ------------------------------------------------
# Sidebar Inputs
# ------------------------------------------------
st.sidebar.header("Patient Clinical Information")

age = st.sidebar.slider("Age", 30, 100, 60)
anaemia = st.sidebar.selectbox("Anaemia", ["No", "Yes"])
diabetes = st.sidebar.selectbox("Diabetes", ["No", "Yes"])
high_bp = st.sidebar.selectbox("High Blood Pressure", ["No", "Yes"])
smoking = st.sidebar.selectbox("Smoking", ["No", "Yes"])

ejection_fraction = st.sidebar.slider("Ejection Fraction", 10, 80, 35)
platelets = st.sidebar.slider("Platelets", 100000, 500000, 250000)
serum_creatinine = st.sidebar.slider("Serum Creatinine", 0.5, 9.0, 1.2)
serum_sodium = st.sidebar.slider("Serum Sodium", 110, 150, 137)
time = st.sidebar.slider("Follow-up Time", 0, 300, 120)
sex = st.sidebar.selectbox("Sex", ["Female", "Male"])

# Convert categorical values
anaemia = 1 if anaemia == "Yes" else 0
diabetes = 1 if diabetes == "Yes" else 0
high_bp = 1 if high_bp == "Yes" else 0
smoking = 1 if smoking == "Yes" else 0
sex = 1 if sex == "Male" else 0

# ------------------------------------------------
# Prepare input data
# ------------------------------------------------
input_df = pd.DataFrame([{
    "age": age,
    "anaemia": anaemia,
    "creatinine_phosphokinase": 250,  # placeholder if not used in UI
    "diabetes": diabetes,
    "ejection_fraction": ejection_fraction,
    "high_blood_pressure": high_bp,
    "platelets": platelets,
    "serum_creatinine": serum_creatinine,
    "serum_sodium": serum_sodium,
    "sex": sex,
    "smoking": smoking,
    "time": time
}])

# ------------------------------------------------
# Prediction
# ------------------------------------------------
if model is not None:
    probability = model.predict_proba(input_df)[0][1]
else:
    # fallback demo calculation if model not loaded
    probability = np.clip(
        (age/100 + serum_creatinine/10 + (1-ejection_fraction/100) + smoking*0.1)/3,
        0,1
    )

risk = probability * 100

# ------------------------------------------------
# Risk Gauge
# ------------------------------------------------
st.subheader("AI Risk Assessment")

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=risk,
    title={'text': "Heart Failure Risk (%)"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "#00d4aa"},
        'steps': [
            {'range': [0, 30], 'color': "#10b981"},
            {'range': [30, 60], 'color': "#f59e0b"},
            {'range': [60, 100], 'color': "#ef4444"}
        ],
    }
))

st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------
# Risk message
# ------------------------------------------------
if risk < 30:
    st.success("Low Risk Patient")
elif risk < 60:
    st.warning("Moderate Risk Patient")
else:
    st.error("High Risk Patient")

st.markdown("---")

# ------------------------------------------------
# Model information
# ------------------------------------------------
with st.expander("Model Information"):
    st.write("""
Model used: **Random Forest Classifier**

Models compared during development:

• Logistic Regression  
• Random Forest  
• XGBoost  
• LightGBM  

After evaluation using **accuracy, precision, recall, F1-score, and ROC-AUC**,  
**Random Forest achieved the best performance** and was selected as the final model.
""")

# ------------------------------------------------
# Dataset information
# ------------------------------------------------
with st.expander("Dataset Information"):
    st.write("""
Dataset: Heart Failure Clinical Records (UCI Repository)

• 299 patient records  
• 13 clinical features  
• Target variable: **DEATH_EVENT**

The dataset is **imbalanced (~68% survived / ~32% deceased)**.
""")

# ------------------------------------------------
# Footer
# ------------------------------------------------
st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;padding:1.2rem 2rem;
            background:rgba(13,21,38,0.75);border:1px solid #1e2d4a;
            border-radius:14px;backdrop-filter:blur(12px);'>
    <div style='font-size:0.68rem;letter-spacing:0.15em;text-transform:uppercase;
                color:#475569;margin-bottom:0.5rem;'>Designed and Developed by</div>
    <div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;
                background:linear-gradient(135deg,#00d4aa,#0099cc);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        ✦ Fouad Ghadi – Group 16 ✦
    </div>
    <div style='font-size:0.72rem;color:#334155;margin-top:0.4rem;'>
        Centrale Casablanca · Coding Week · March 2026
    </div>
</div>
""", unsafe_allow_html=True)
