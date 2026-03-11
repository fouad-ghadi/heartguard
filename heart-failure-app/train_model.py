import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# ═══════════════════════════════════════════════════
#  ÉTAPE 1 — ENTRAÎNEMENT DU MODÈLE
# ═══════════════════════════════════════════════════

def entrainer_modele():
    # Charger les données
    df = pd.read_csv('nouvelle_dataset_equilibrée.csv')
    print(f"✅ Dataset chargé : {len(df)} patients")
    print(f"   Survivants : {(df['DEATH_EVENT']==0).sum()}")
    print(f"   Décédés    : {(df['DEATH_EVENT']==1).sum()}")

    # Séparer features et cible
    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']

    # Split 80% train / 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Créer et entraîner le modèle
    modele = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    modele.fit(X_train, y_train)

    # Évaluer le modèle
    y_pred       = modele.predict(X_test)
    y_pred_proba = modele.predict_proba(X_test)[:, 1]

    print(f"\n📊 Résultats du modèle :")
    print(f"   Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"   ROC-AUC   : {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"   F1-Score  : {f1_score(y_test, y_pred):.4f}")

    # Sauvegarder le modèle
    os.makedirs('models', exist_ok=True)
    joblib.dump(modele, 'models/random_forest.pkl')
    print(f"\n💾 Modèle sauvegardé : models/random_forest.pkl")

    return modele


# ═══════════════════════════════════════════════════
#  ÉTAPE 2 — FONCTION DE PRÉDICTION
# ═══════════════════════════════════════════════════

def prediction(age, anaemia, creatinine_phosphokinase, diabetes,
               ejection_fraction, high_blood_pressure, platelets,
               serum_creatinine, serum_sodium, sex, smoking, time):

    modele = joblib.load('models/random_forest.pkl')

    patient = pd.DataFrame([{
        'age': age, 'anaemia': anaemia,
        'creatinine_phosphokinase': creatinine_phosphokinase,
        'diabetes': diabetes, 'ejection_fraction': ejection_fraction,
        'high_blood_pressure': high_blood_pressure, 'platelets': platelets,
        'serum_creatinine': serum_creatinine, 'serum_sodium': serum_sodium,
        'sex': sex, 'smoking': smoking, 'time': time
    }])

    probabilites       = modele.predict_proba(patient)[0]
    pourcentage_risque = round(probabilites[1] * 100, 2)

    if pourcentage_risque >= 70:   diagnostic = "⚠️  RISQUE ÉLEVÉ"
    elif pourcentage_risque >= 40: diagnostic = "⚡ RISQUE MODÉRÉ"
    else:                          diagnostic = "✅ FAIBLE RISQUE"

    print(f"\n{'═'*45}")
    print(f"  🫀 RÉSULTAT DE PRÉDICTION")
    print(f"{'═'*45}")
    print(f"  Probabilité de survie       : {round(probabilites[0]*100, 2)}%")
    print(f"  Probabilité d'insuffisance  : {pourcentage_risque}%")
    print(f"  Diagnostic                  : {diagnostic}")
    print(f"{'═'*45}")

    return pourcentage_risque, diagnostic


# ═══════════════════════════════════════════════════
#  ÉTAPE 3 — PROGRAMME PRINCIPAL
# ═══════════════════════════════════════════════════

if __name__ == "__main__":

    print("=" * 45)
    print("  ENTRAÎNEMENT DU MODÈLE")
    print("=" * 45)
    entrainer_modele()

    print("\n\n" + "=" * 45)
    print("  EXEMPLES DE PRÉDICTIONS")
    print("=" * 45)

    print("\n👤 Patient 1 — Profil critique")
    prediction(65, 0, 160, 1, 20, 0, 327000, 2.7, 116, 0, 0, 8)

    print("\n👤 Patient 2 — Profil sain")
    prediction(45, 0, 582, 0, 38, 0, 265000, 1.1, 136, 1, 0, 60)

    print("\n👤 Patient 3 — Profil très critique")
    prediction(80, 1, 123, 0, 35, 1, 388000, 9.4, 133, 1, 1, 10)
