# 💳 Credit Card Fraud Detection using Machine Learning

This project showcases the use of machine learning algorithms to identify fraudulent credit card transactions with high precision and recall. The implementation includes data preprocessing, handling imbalanced data, model evaluation, and deployment via a web app.
📂 **Project Portfolio:** [GitHub Repository](https://github.com/dheerajsingh544/credit-card-fraud-detection)  
📃 **Resume Tagline:** "Built a real-time Credit Card Fraud Detection system using ML (92% ROC-AUC, SMOTE balancing, Streamlit app)."

---

## 📊 Overview

- 🚀 Built ML models to classify transactions as fraudulent or legitimate
- 📉 Handled extreme class imbalance using SMOTE
- 🔍 Evaluated models using precision, recall, F1-score, and ROC-AUC

---

## 🗂️ Dataset

- **Source**: [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Features: `V1` to `V28` (PCA components), `Amount`, `Time`, and `Class` (target)
- 284,807 total records; only 492 are frauds

---

## 🛠️ ML Pipeline

1. **Preprocessing**: Scaled `Amount`, dropped `Time`
2. **Imbalanced Handling**: SMOTE oversampling
3. **Modeling**: Logistic Regression, Random Forest, XGBoost
4. **Evaluation**: Confusion matrix, F1-score, ROC-AUC
5. **Deployment**: Streamlit app for live prediction
