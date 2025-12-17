![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![ML](https://img.shields.io/badge/Machine%20Learning-Classification-purple)

🏦 Bank Loan Default Prediction — Machine Learning Project
## 🔍 Why This Project Matters

This project demonstrates how machine learning can be applied to **real-world financial risk assessment**, covering the **full ML lifecycle**:

- Data preprocessing using pipelines
- Baseline vs advanced model comparison
- Threshold tuning for business impact
- Model persistence and reproducibility
- Clear evaluation using ROC-AUC and Precision–Recall

The codebase is structured for **reuse in future ML projects** and follows production-style practices.

An end-to-end machine learning pipeline to predict bank loan default risk, using structured financial data and modern ML practices.
The project compares:
Logistic Regression (baseline, interpretable)
Random Forest (high-performance ensemble)
It is designed to be clean, reusable, and recruiter-friendly, with modular code, saved models, and clear evaluation.
📁 Project Structure
bank-loan-default-models/
│
├── data/                      # Raw / processed data (not included)
│
├── notebooks/
│   └── 01_eda.ipynb           # EDA, feature analysis, model experiments
│
├── src/
│   ├── preprocess.py          # Data preprocessing pipeline
│   ├── train.py               # Train & save ML models
│   ├── predict.py             # Load model & run predictions
│   ├── evaluation.py          # Evaluation metrics
│   └── threshold.py           # Threshold tuning
│
├── models/
│   ├── logistic_regression.pkl
│   └── random_forest.pkl
│
├── reports/
│   ├── precision-recall-curve.png
│   ├── roc-curve.png
│   └── bank_loan_default_visual.pdf
│
└── README.md
🧠 Problem Overview
Banks need to assess whether a loan applicant is likely to default on repayments.
This project predicts:
0 → No Default
1 → Default
Accurate predictions help:
Reduce financial risk
Improve credit decision-making
Support automated lending systems
🧹 Data Preprocessing
Handled in preprocess.py using a scikit-learn Pipeline:
Missing value handling
Categorical feature encoding (One-Hot Encoding)
Numerical feature scaling (StandardScaler)
Unified ColumnTransformer
This ensures consistent preprocessing during training and inference.
🤖 Models Used
✔ Logistic Regression (Baseline)
Interpretable and fast
Suitable for initial benchmarking
ROC-AUC ≈ 0.85
✔ Random Forest (Advanced)
Captures non-linear patterns
Handles class imbalance (class_weight='balanced')
ROC-AUC ≈ 1.00
Strong recall on default class
📊 Model Performance
ROC Curve
Precision–Recall Curve
Random Forest clearly outperforms Logistic Regression across both metrics.
🏗 How to Run the Project
1️⃣ Train models
From the project root:
python src/train.py
This saves trained models to:
models/
├── logistic_regression.pkl
└── random_forest.pkl
2️⃣ Run predictions
python src/predict.py
📄 Printable Visual Guide
A step-by-step, print-friendly PDF explaining the full pipeline is included:
📎 reports/bank_loan_default_visual.pdf
Ideal for revision, learning, and interviews.
🎯 Key Results Summary
Model	ROC-AUC	Notes
Logistic Regression	0.849	Strong baseline
Random Forest	1.000	Excellent performance
🚀 Future Improvements
Add XGBoost
Hyperparameter tuning (GridSearchCV / Optuna)
Model explainability (SHAP)
Deploy using FastAPI
Add CI pipeline
👨‍💻 Author
Muhammad Shaan
MSc Computer Science (Data Analytics)
