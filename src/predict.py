"""
predict.py

Simple helper for making predictions on new loan applications
using the saved Logistic Regression model.
"""

import os
import joblib
import pandas as pd

MODEL_PATH = os.path.join("models", "logistic_regression.pkl")
DEFAULT_THRESHOLD = 0.35  # same as used in training / evaluation


def load_model(path: str = MODEL_PATH):
    """Load a trained model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file not found at {path}. "
            "Train the model first by running train.py"
        )
    return joblib.load(path)


def predict_single(application_data: dict, threshold: float = DEFAULT_THRESHOLD):
    """
    Predict default risk for a single loan application.

    application_data: dictionary with the same keys as training features,
    e.g. {
        "year": 2019,
        "loan_limit": "cf",
        "Gender": "Male",
        ...
    }
    """
    model = load_model()
    df = pd.DataFrame([application_data])
    proba = model.predict_proba(df)[0, 1]
    label = int(proba >= threshold)
    return label, proba


if __name__ == "__main__":
    # Example dummy call – adjust values to real ranges if you want to test.
    example = {
        "year": 2019,
        "loan_limit": "cf",
        "Gender": "Male",
        "approv_in_adv": "nopre",
        "loan_type": "type1",
        "loan_purpose": "p1",
        "Credit_Worthiness": "l1",
        "open_credit": "nopc",
        "business_or_commercial": "nob/c",
        "loan_amount": 200000,
        "rate_of_interest": 9.5,
        "Interest_rate_spread": 2.5,
        "Upfront_charges": 1000.0,
        "term": 360.0,
        "Neg_ammortization": "notneg",
        "interest_only": "notint",
        "lump_sum_payment": "notlump",
        "property_value": 250000,
        "construction_type": "unknown",
        "occupancy_type": "home",
        "Secured_by": "home",
        "total_units": 1.0,
        "income": 60000,
        "credit_type": "EQUI",
        "Credit_Score": 720,
        "co-applicant_credit_type": "none",
        "age": "45",
        "LTV": 80.0,
        "dtir1": 35.0,
    }

    label, proba = predict_single(example)
    print(f"Predicted label: {label} (1=default, 0=non-default)")
    print(f"Predicted probability of default: {proba:.3f}")

