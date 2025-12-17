"""
threshold.py

Helpers for exploring and selecting decision thresholds
for the default (positive) class.
"""

import numpy as np
from sklearn.metrics import classification_report


def scan_thresholds(y_true, y_proba, thresholds=None):
    """
    Print recall/precision for multiple thresholds.

    thresholds: iterable of threshold values. If None, use 0.10 → 0.85 step 0.05.
    """
    if thresholds is None:
        thresholds = np.arange(0.10, 0.90, 0.05)

    results = []

    for t in thresholds:
        y_pred_t = (y_proba >= t).astype(int)
        report = classification_report(
            y_true,
            y_pred_t,
            output_dict=True,
            zero_division=0,
        )
        rec = report["1"]["recall"]
        prec = report["1"]["precision"]
        results.append((t, prec, rec))
        print(f"Threshold {t:.2f} -> Precision (1): {prec:.3f}, Recall (1): {rec:.3f}")

    return results


def pick_threshold_for_recall(results, target_recall: float = 0.8):
    """
    Given results from scan_thresholds(), pick a threshold that
    achieves at least `target_recall` for class 1, with highest precision.
    """
    candidates = [r for r in results if r[2] >= target_recall]
    if not candidates:
        return None

    # Sort by precision descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    best = candidates[0]
    print(f"\nBest threshold for recall>={target_recall}: "
          f"t={best[0]:.2f}, precision={best[1]:.3f}, recall={best[2]:.3f}")
    return best

