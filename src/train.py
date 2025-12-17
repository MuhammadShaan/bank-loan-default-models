"""
train.py

Train machine learning models for Bank Loan Default prediction.
Currently focuses on Logistic Regression as the primary model.
"""

import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score

from preprocess import get_data_and_preprocessor

MODELS_DIR = "models"
LOG_REG_PATH = os.path.join(MODELS_DIR, "logistic_regression.pkl")

# Threshold chosen during notebook exploration
DEFAULT_THRESHOLD = 0.35


def train_logistic_regression():
    """Train Logistic Regression model and save it to disk."""
    X_train, X_test, y_train, y_test, preprocessor = get_data_and_preprocessor()

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=200,
            class_weight="balanced",
        )),
    ])

    model.fit(X_train, y_train)

    # Basic evaluation
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= DEFAULT_THRESHOLD).astype(int)

    print("\nLogistic Regression – Classification Report "
          f"(threshold={DEFAULT_THRESHOLD}):")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC–AUC: {auc:.3f}")

    # Save the model
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, LOG_REG_PATH)
    print(f"\nSaved Logistic Regression model to: {LOG_REG_PATH}")

    return model, (X_test, y_test, y_proba)


if __name__ == "__main__":
    train_logistic_regression()

