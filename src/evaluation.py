"""
evaluation.py

Utility functions for evaluating trained models:
- confusion matrix
- classification report
- ROC & Precision–Recall curves
"""

import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)


def print_basic_metrics(y_true, y_proba, threshold: float = 0.5, label: str = ""):
    """Print confusion matrix & classification report for a given model."""
    y_pred = (y_proba >= threshold).astype(int)

    print(f"\n=== {label} – threshold={threshold} ===")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    auc = roc_auc_score(y_true, y_proba)
    print(f"ROC–AUC: {auc:.3f}")


def plot_roc_curves(y_true, proba_dict, title="ROC Curve – Loan Default Models"):
    """
    Plot ROC curves for a dictionary of model probabilities.

    proba_dict: {"Logistic Regression": y_proba_log, "Random Forest": y_proba_rf}
    """
    plt.figure()

    for label, y_proba in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_pr_curves(y_true, proba_dict, title="Precision–Recall Curve – Loan Default Models"):
    """
    Plot Precision–Recall curves for a dictionary of model probabilities.
    Focuses on the positive (default) class.
    """
    plt.figure()

    for label, y_proba in proba_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        plt.plot(recall, precision, label=label)

    plt.xlabel("Recall (Default class)")
    plt.ylabel("Precision (Default class)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

