Bank Loan Default Prediction (ML Project)
Predicting whether a customer will default on a loan using machine learning.
This project reuses and extends the folder structure + workflow from my previous Telco Churn ML project.
📌 1. Project Goal
Build a machine learning pipeline that:
Cleans and preprocesses loan application data
Encodes categorical variables
Splits the dataset into training/testing
Trains multiple ML models
Evaluates model performance
Tunes the probability threshold for business decisions
Target variable:
Status = 1 → Customer defaulted
Status = 0 → Customer paid successfully
Why this matters:
Banks want to reduce financial losses, improve risk assessment, and decide whether to approve or decline a loan application.
📂 2. Folder Structure
bank-loan-default-ml/
│
├── data/
│   ├── raw/loan_default.csv
│   ├── processed/
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate.py
│
├── models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│
├── reports/
│   ├── figures/
│   └── final_metrics.pdf
│
└── README.md
This structure allows clean, reusable code for future ML projects.
📊 3. Dataset Overview
The dataset contains 148k loan applications with:
Categorical features (loan type, credit type, gender, co-applicant type)
Numeric features (loan amount, interest rate, income, property value)
Missing values in income, rate_of_interest, and spread columns
Class imbalance (only ~24% defaults)
Basic cleaning steps included:
Dropping ID-like columns
Handling missing values
Separating numerical vs categorical columns
Checking duplicates
⚙️ 4. ML Pipeline (Reusable Template)
The project uses a unified scikit-learn Pipeline, including:
Numerical preprocessing
SimpleImputer(strategy='median')
StandardScaler()
Categorical preprocessing
SimpleImputer(strategy='most_frequent')
OneHotEncoder(handle_unknown='ignore')
Combined with ColumnTransformer, then fed into a model such as:
Logistic Regression
Random Forest
(XGBoost planned next)
This makes the workflow clean, repeatable, and ready for deployment.
🤖 5. Models Trained
✔ Logistic Regression
Performs realistically
ROC-AUC ≈ 0.85
Balanced between precision and recall
Best for interpretability
⚠ Random Forest (Overfitting Detected)
Returned unrealistic 100% accuracy
Caused by high-cardinality categorical variables + one-hot encoding
Not reliable without category reduction
(Next) XGBoost
Will provide a more powerful and stable alternative.
📈 6. Evaluation Metrics
Metrics used:
Confusion Matrix
Precision, Recall, F1-score
ROC Curve + ROC-AUC
Precision–Recall Curve
Threshold tuning (0.10 → 0.85)
Why threshold tuning matters
Banks may prefer:
High recall → catch every risky borrower
Even if precision drops (more false alarms)
We evaluated thresholds like 0.25, 0.30, 0.35 to improve default detection.
📉 7. Key Results
Logistic Regression (Threshold = 0.35)
Recall for default: 0.77
Precision for default: 0.44
Balanced approach, interpretable
Random Forest
Produced perfect scores (overfitting)
Not suitable without feature engineering
🧠 8. What I learned
How to reuse ML project templates
How to separate categorical & numeric preprocessing
Why Random Forest overfits with high-cardinality categorical features
How threshold tuning changes business decisions
How to evaluate with PR/ROC curves
How to design a clean classification workflow
🚀 9. Next Steps
Implement XGBoost
Add feature importance visualisations
Create a final comparison table
Export a printable PDF report
Push the project cleanly to GitHub
🙌 10. Tools Used
Python
scikit-learn
pandas, numpy
matplotlib
VS Code
Jupyter Notebook


👤 Author
Muhammad Shaan
MSc Computer Science (Data Analytics)
Carlisle, UK


