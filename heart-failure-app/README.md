# 🫀 HeartGuard — Heart Failure Risk Predictor

> Centrale Casablanca · Coding Week · March 2026  
> **Web App Module** — Streamlit clinical decision-support interface with SHAP explainability

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the dataset and place it at:
#    data/heart_failure_clinical_records_dataset.csv
#    Source: https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records

# 3. Train the model
python src/train_model.py

# 4. Run the web application
streamlit run app/app.py
```

---

## Project Structure

```
project/
├── data/
│   └── heart_failure_clinical_records_dataset.csv
├── notebooks/
│   └── eda.ipynb                 # Exploratory data analysis
├── src/
│   ├── data_processing.py        # Preprocessing + memory optimization
│   ├── train_model.py            # Model training + evaluation
│   └── evaluate_model.py         # Standalone evaluation utilities
├── app/
│   └── app.py                    # Streamlit web interface ← YOU ARE HERE
├── models/
│   ├── model.joblib              # Saved best model (generated after training)
│   └── scaler.joblib             # Saved scaler
├── tests/
│   └── test_data_processing.py   # Automated pytest tests
├── .github/workflows/ci.yml      # GitHub Actions CI/CD
├── requirements.txt
└── README.md
```

---

## Critical Questions

### Was the dataset balanced?
No. The dataset is imbalanced: ~68% survived (DEATH_EVENT=0) and ~32% deceased (DEATH_EVENT=1).  
**Handling:** SMOTE (Synthetic Minority Oversampling Technique) was applied to the training set only.  
**Impact:** Improved recall on the minority class (deceased), raising F1 from ~0.72 to ~0.85.

### Which ML model performed best?
| Model | ROC-AUC | Accuracy | F1 |
|---|---|---|---|
| **XGBoost** ✅ | **0.91** | **87%** | **0.85** |
| LightGBM | 0.90 | 86% | 0.84 |
| Random Forest | 0.89 | 85% | 0.82 |
| Logistic Regression | 0.82 | 79% | 0.76 |

**XGBoost** selected as final model.

### Which medical features most influenced predictions (SHAP)?
1. **Follow-up time** — strongest predictor overall
2. **Ejection fraction** — key cardiac function marker
3. **Serum creatinine** — top biochemical risk indicator

### Prompt Engineering
A dedicated prompt engineering log is in `docs/prompt_engineering.md`.

---

## Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- Missing value imputation correctness
- `optimize_memory()` — dtype casting and value preservation
- Outlier clipping bounds
- Feature/target split integrity

---

## Web App Features

- 🩺 **Patient input panel** — sidebar with all 12 clinical features
- 🔴/🟢 **Risk score gauge** — probability with HIGH/LOW classification
- 📊 **Patient-specific SHAP chart** — per-prediction feature attribution
- ⚕️ **Clinical flags** — automated alerts for critical values
- ℹ️ **Model info expander** — metrics, methodology, reproducibility guide
