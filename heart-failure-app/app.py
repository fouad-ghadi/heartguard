import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib, os
import csv
from train_model import entrainer_modele  # Importation pour le réentraînement

st.set_page_config(page_title="HeartGuard — Prédiction Insuffisance Cardiaque", page_icon="❤️", layout="wide")

# ── Image de Fond (Conservez votre longue chaîne ici) ────────────────────────
BG_B64 = "VOTRE_TRES_LONGUE_CHAINE_BASE64_ICI"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{BG_B64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ── Fonction d'Apprentissage Continu ──────────────────────────────────────────
def enregistrer_et_reentrainer(features_patient, vrai_diagnostic):
    features_patient['DEATH_EVENT'] = vrai_diagnostic
    colonnes = [
        'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
        'ejection_fraction', 'high_blood_pressure', 'platelets', 
        'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time', 'DEATH_EVENT'
    ]
    
    # Ajout de la nouvelle donnée dans le dataset
    with open('nouvelle_dataset_equilibree.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=colonnes)
        writer.writerow(features_patient)
        
    # Relance de l'entraînement
    try:
        entrainer_modele()
        return True
    except Exception as e:
        st.error(f"Erreur lors de l'entraînement : {e}")
        return False

# ── Sidebar pour la saisie des données ────────────────────────────────────────
st.sidebar.header("📝 Saisie des Paramètres")

age = st.sidebar.number_input("Âge", min_value=1, max_value=120, value=60)
anaemia = st.sidebar.selectbox("Anémie", options=[0, 1], format_func=lambda x: "Oui" if x==1 else "Non")
creatinine_phosphokinase = st.sidebar.number_input("Créatinine Phosphokinase (mcg/L)", min_value=10, max_value=8000, value=250)
diabetes = st.sidebar.selectbox("Diabète", options=[0, 1], format_func=lambda x: "Oui" if x==1 else "Non")
ejection_fraction = st.sidebar.number_input("Fraction d'éjection (%)", min_value=10, max_value=80, value=38)
high_blood_pressure = st.sidebar.selectbox("Hypertension artérielle", options=[0, 1], format_func=lambda x: "Oui" if x==1 else "Non")
platelets = st.sidebar.number_input("Plaquettes (kiloplatelets/mL)", min_value=20000.0, max_value=900000.0, value=265000.0)
serum_creatinine = st.sidebar.number_input("Créatinine sérique (mg/dL)", min_value=0.5, max_value=10.0, value=1.1)
serum_sodium = st.sidebar.number_input("Sodium sérique (mEq/L)", min_value=110, max_value=150, value=137)
sex = st.sidebar.selectbox("Sexe", options=[0, 1], format_func=lambda x: "Homme" if x==1 else "Femme")
smoking = st.sidebar.selectbox("Fumeur", options=[0, 1], format_func=lambda x: "Oui" if x==1 else "Non")
time = st.sidebar.number_input("Période de suivi (jours)", min_value=1, max_value=300, value=4)

# Données formatées pour le modèle et le réentraînement
patient_data = {
    'age': age,
    'anaemia': anaemia,
    'creatinine_phosphokinase': creatinine_phosphokinase,
    'diabetes': diabetes,
    'ejection_fraction': ejection_fraction,
    'high_blood_pressure': high_blood_pressure,
    'platelets': platelets,
    'serum_creatinine': serum_creatinine,
    'serum_sodium': serum_sodium,
    'sex': sex,
    'smoking': smoking,
    'time': time
}

df_patient = pd.DataFrame([patient_data])

# ── Chargement du modèle et Prédiction ────────────────────────────────────────
st.title("🫀 HeartGuard")
st.markdown("### Assistant IA pour la Prédiction du Risque d'Insuffisance Cardiaque")

try:
    modele = joblib.load("modele.pkl")
    probabilites = modele.predict_proba(df_patient)[0]
    pourcentage_risque = round(probabilites[1] * 100, 2)
    
    st.markdown("---")
    st.markdown("### Résultat de la Prédiction")
    
    if pourcentage_risque >= 70:
        st.error(f"⚠️ RISQUE ÉLEVÉ ({pourcentage_risque}%)")
    elif pourcentage_risque >= 40:
        st.warning(f"⚡ RISQUE MODÉRÉ ({pourcentage_risque}%)")
    else:
        st.success(f"✅ FAIBLE RISQUE ({pourcentage_risque}%)")
        
except FileNotFoundError:
    st.warning("Veuillez entraîner le modèle en exécutant d'abord `train_model.py`.")

# ── Validation Clinique (Apprentissage Continu) ──────────────────────────────
st.markdown("---")
st.markdown("### 👨‍⚕️ Validation Clinique (Apprentissage Continu)")
st.caption("Confirmez l'issue réelle de ce patient pour ré-entrainer l'intelligence artificielle.")

col_feedback1, col_feedback2 = st.columns(2)

with col_feedback1:
    if st.button("🟢 Confirmer : Le patient a survécu (0)", use_container_width=True):
        with st.spinner("💾 Enregistrement et réentraînement du modèle en cours..."):
            succes = enregistrer_et_reentrainer(patient_data, 0)
            if succes:
                st.success("✅ Le modèle a appris de ce cas et a été mis à jour !")
                
with col_feedback2:
    if st.button("🔴 Confirmer : Le patient est décédé (1)", use_container_width=True):
        with st.spinner("💾 Enregistrement et réentraînement du modèle en cours..."):
            succes = enregistrer_et_reentrainer(patient_data, 1)
            if succes:
                st.success("✅ Le modèle a appris de ce cas et a été mis à jour !")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;padding:1.2rem 2rem;
            background:rgba(13,21,38,0.75);border:1px solid #1e2d4a;
            border-radius:14px;backdrop-filter:blur(12px);'>
    <div style='font-size:0.68rem;letter-spacing:0.15em;text-transform:uppercase;
                color:#475569;margin-bottom:0.5rem;'>Made by</div>
    <div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:700;
                background:linear-gradient(90deg, #94a3b8, #e2e8f0);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        HeartGuard Team
    </div>
</div>
""", unsafe_allow_html=True)
