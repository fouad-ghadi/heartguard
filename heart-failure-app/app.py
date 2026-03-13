import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib, os

st.set_page_config(page_title="HeartGuard — Prédiction Insuffisance Cardiaque", page_icon="❤️", layout="wide")

# ── Image de Fond (Remettez votre chaîne complète ici) ────────────────────────
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
    
    # 1. 🌟 CORRECTION DE L'ORDRE (Sécurisée : sans feature_names_in_)
    colonnes_attendues = [
        'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 
        'ejection_fraction', 'high_blood_pressure', 'platelets', 
        'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
    ]
    df_patient = df_patient[colonnes_attendues]
    
    # 2. 🌟 PRÉDICTION BRUTE
    probabilites = modele.predict_proba(df_patient)[0]
    prob_deces = probabilites[1]
    
    # 3. 🛡️ CORRECTION CLINIQUE POUR LE TABAGISME (Override)
    if smoking == 1:
        prob_deces += 0.05  # Ajoute 5% au risque final
        prob_deces = min(prob_deces, 1.0)  # Sécurité : empêche de dépasser 100%
        
    pourcentage_risque = round(prob_deces * 100, 2)
    
    st.markdown("---")
    st.markdown("### Résultat de la Prédiction")
    
    if pourcentage_risque >= 70:
        st.error(f"⚠️ RISQUE ÉLEVÉ ({pourcentage_risque}%)")
    elif pourcentage_risque >= 40:
        st.warning(f"⚡ RISQUE MODÉRÉ ({pourcentage_risque}%)")
    else:
        st.success(f"✅ FAIBLE RISQUE ({pourcentage_risque}%)")
        
    # 4. 📊 GRAPHIQUE SÉCURISÉ (Try/Except indépendant)
    try:
        st.markdown("### Facteurs de risque principaux du patient")
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        
        importances = pd.Series(modele.feature_importances_, index=colonnes_attendues)
        importances.nlargest(5).sort_values().plot(kind='barh', color='#00d4aa', ax=ax)
        
        ax.set_title("Poids des indicateurs", color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        st.pyplot(fig)
    except Exception as e_graph:
        st.info("Le graphique détaillé n'a pas pu s'afficher, mais la prédiction est correcte.")

except FileNotFoundError:
    st.warning("Veuillez entraîner le modèle en exécutant d'abord `train_model.py` (ou assurez-vous que `modele.pkl` est dans le même dossier).")
except Exception as e:
    st.error(f"Une erreur système est survenue lors de la prédiction : {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
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
