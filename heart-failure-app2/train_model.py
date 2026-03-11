"""
train_model.py
Trains and evaluates multiple ML models, selects the best, and saves it.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Optional heavy models — graceful fallback if not installed
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed — skipping.")

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("LightGBM not installed — skipping.")

sys.path.insert(0, os.path.dirname(__file__))
from data_processing import load_data, optimize_memory, handle_missing_values, handle_outliers, get_feature_target, scale_features

DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data", "heart_failure_clinical_records_dataset.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "model.joblib")
SCALER_PATH= os.path.join(os.path.dirname(__file__), "..", "models", "scaler.joblib")
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    metrics = {
        "ROC-AUC":   round(roc_auc_score(y_test, probs), 4),
        "Accuracy":  round(accuracy_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds), 4),
        "Recall":    round(recall_score(y_test, preds), 4),
        "F1":        round(f1_score(y_test, preds), 4),
    }
    print(f"\n── {name} ──")
    for k, v in metrics.items():
        print(f"   {k}: {v}")
    return metrics


def train():
    print("Loading & preprocessing data...")
    df = load_data(DATA_PATH)
    df = optimize_memory(df)
    df = handle_missing_values(df)

    continuous_cols = ["age", "creatinine_phosphokinase", "ejection_fraction",
                       "platelets", "serum_creatinine", "serum_sodium", "time"]
    df = handle_outliers(df, continuous_cols, method="clip")

    X, y = get_feature_target(df)
    print(f"Class distribution:\n{y.value_counts(normalize=True).round(2)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Handle class imbalance with SMOTE
    print("\nApplying SMOTE to training set...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    X_train_s, X_test_s, scaler = scale_features(
        pd.DataFrame(X_train_res, columns=X.columns),
        X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=200, use_label_encoder=False,
                                           eval_metric="logloss", random_state=42)
    if HAS_LGB:
        models["LightGBM"] = LGBMClassifier(n_estimators=200, random_state=42)

    results = {}
    trained = {}
    for name, m in models.items():
        m.fit(X_train_s, y_train_res)
        results[name] = evaluate(name, m, X_test_s, y_test)
        trained[name] = m

    # Select best by ROC-AUC
    best_name = max(results, key=lambda n: results[n]["ROC-AUC"])
    best_model = trained[best_name]
    print(f"\n✅ Best model: {best_name} (ROC-AUC={results[best_name]['ROC-AUC']})")

    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler,     SCALER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")


if __name__ == "__main__":
    train()
