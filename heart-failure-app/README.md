# Heart Failure Risk Prediction -- Medical Decision Support System

## Project Overview

This project implements a **clinical decision-support application**
designed to help physicians estimate the **risk of heart failure
mortality** using machine learning models.\
The system predicts the probability of death for a patient based on
clinical features and provides **explainable AI insights** to make
predictions transparent and interpretable.

The application includes: - Data preprocessing and analysis - Machine
learning model training and evaluation - Model explainability using
SHAP - A user interface for physicians built with Streamlit

------------------------------------------------------------------------

# Dataset

The project uses the **Heart Failure Clinical Records Dataset** from the
UCI Machine Learning Repository.

Dataset link:\
https://archive.ics.uci.edu/dataset/519/heart%2Bfailure%2Bclinical%2Brecords

The dataset contains **299 patient records** and several clinical
variables such as:

-   Age
-   Anaemia
-   Diabetes
-   High blood pressure
-   Platelets
-   Serum sodium
-   Serum creatinine
-   Ejection fraction
-   Smoking status
-   Follow-up time

Target variable:

**DEATH_EVENT** - 0 → Patient survived - 1 → Patient died

------------------------------------------------------------------------

# Data Analysis

## Missing Values

The dataset does not contain missing values, so no imputation was
required.

## Outliers

Certain medical variables contain extreme values. These were analyzed
during exploratory data analysis and handled using scaling and
preprocessing techniques when necessary.

## Class Imbalance

The dataset is **imbalanced**:

-   \~68% survived
-   \~32% died

To address this imbalance we experimented with techniques such as:

-   Class weighting
-   Oversampling methods

This improved the model's ability to correctly predict the minority
class.

------------------------------------------------------------------------

# Machine Learning Models

Several machine learning models were trained and compared:

-   Logistic Regression
-   Random Forest Classifier
-   XGBoost
-   LightGBM

Each model was evaluated using multiple performance metrics:

-   Accuracy
-   Precision
-   Recall
-   F1 Score
-   ROC-AUC

After comparing the results, the **Random Forest Classifier** achieved
the best overall performance across the evaluation metrics.\
For this reason, Random Forest was selected as the **final model used in
the application**.

------------------------------------------------------------------------

# Model Explainability

To improve trust and transparency, we integrated **SHAP (SHapley
Additive Explanations)**.

SHAP helps explain:

-   Which features influenced the prediction
-   How much each feature contributed to the final risk score

The application provides visual explanations so physicians can
understand **why a prediction was made**.

------------------------------------------------------------------------

# Observation About Smoking Feature

During testing, we observed that selecting **smoking = yes** sometimes
**reduced the predicted probability of death instead of increasing it**.

This behavior does not necessarily indicate an error. Machine learning
models learn patterns from the dataset rather than real-world medical
causality.

In this dataset, smoking is **not strongly correlated with the death
outcome**, and smokers in the dataset may appear more frequently in
lower-risk groups (for example younger patients or patients with better
clinical indicators).

Therefore the model may associate smoking with lower risk in some
situations depending on the combination of other features.

This observation highlights the importance of **explainable AI tools
such as SHAP** to interpret model decisions.

------------------------------------------------------------------------

# Application Interface

The system includes a **Streamlit web interface** that allows physicians
to:

-   Enter patient clinical data
-   Generate a heart failure risk prediction
-   View probability scores
-   Understand the model decision through SHAP explanations

------------------------------------------------------------------------

# Project Structure

    project/
    │
    ├── notebooks/        # Exploratory Data Analysis
    ├── src/              # Data processing and model training
    ├── app/              # Streamlit web application
    ├── tests/            # Automated tests
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

# Installation

Clone the repository:

    git clone <repository_url>
    cd project

Install dependencies:

    pip install -r requirements.txt

------------------------------------------------------------------------

# Train the Model

    python src/train_model.py

------------------------------------------------------------------------

# Run the Application

    streamlit run app/app.py

------------------------------------------------------------------------

# Automated Testing

The project includes automated tests to verify important components such
as:

-   Data preprocessing
-   Model loading
-   Prediction functionality

Tests are executed automatically through **GitHub Actions**.

------------------------------------------------------------------------

# Reproducibility

To reproduce the project:

1.  Install dependencies
2.  Train the model
3.  Launch the Streamlit application

------------------------------------------------------------------------

# Authors

Coding Week Project\
Medical Decision Support Application
