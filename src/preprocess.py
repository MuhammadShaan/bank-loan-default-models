"""
preprocess.py

All data loading and preprocessing helpers for the
Bank Loan Default prediction project.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------------
# Project-level constants
# ------------------------------------------------------------------

DATA_PATH = os.path.join("data", "raw", "loan_default.csv")
TARGET_COL = "Status"

# Columns to drop entirely
DROP_COLS = ["ID", "Region", "Security_Type", "submission_of_application"]

# Categorical and numeric feature lists
CATEGORICAL_FEATURES = [
    "loan_limit",
    "Gender",
    "approv_in_adv",
    "loan_type",
    "loan_purpose",
    "Credit_Worthiness",
    "open_credit",
    "business_or_commercial",
    "Neg_ammortization",
    "interest_only",
    "lump_sum_payment",
    "construction_type",
    "occupancy_type",
    "Secured_by",
    "credit_type",
    "co-applicant_credit_type",
    "age",
]

NUMERIC_FEATURES = [
    "year",
    "loan_amount",
    "rate_of_interest",
    "Interest_rate_spread",
    "Upfront_charges",
    "term",
    "property_value",
    "income",
    "Credit_Score",
    "LTV",
    "dtir1",
]


# ------------------------------------------------------------------
# Core helper functions
# ------------------------------------------------------------------

def load_raw_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw loan default dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find data file at: {path}")
    df = pd.read_csv(path)
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unused columns and return a clean DataFrame."""
    existing_drop_cols = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=existing_drop_cols)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Create the ColumnTransformer used for all models."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor


def train_test_split_data(df: pd.DataFrame, test_size: float = 0.2,
                          random_state: int = 42):
    """Split features/target into train and test sets with stratification."""
    if TARGET_COL not in df.columns:
        raise ValueError(f"TARGET_COL '{TARGET_COL}' not found in DataFrame.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def get_data_and_preprocessor():
    """
    Convenience function used by train/evaluation scripts.

    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    df = load_raw_data()
    df = clean_dataframe(df)
    preprocessor = build_preprocessor()
    X_train, X_test, y_train, y_test = train_test_split_data(df)
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    # Quick sanity check if you run: python src/preprocess.py
    df = load_raw_data()
    print("Raw shape:", df.shape)
    df = clean_dataframe(df)
    print("After dropping columns:", df.shape)
    X_train, X_test, y_train, y_test = train_test_split_data(df)
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
