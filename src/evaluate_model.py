"""
Model Evaluation and Metrics Module.
Calculates rigorous classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC),
confusion matrices, and comparison tables with zero fabrication.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)


def evaluate_single_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model"
) -> Dict[str, Any]:
    """
    Evaluates a trained classification model on test data using standard geotechnical ML metrics.
    """
    y_pred = model.predict(X_test)

    # Predicted probability for risk class (1)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        # For classifiers without probability, use decision function scaled via sigmoid
        df_vals = model.decision_function(X_test)
        y_prob = 1 / (1 + np.exp(-df_vals))
    else:
        y_prob = y_pred.astype(float)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    roc_auc = float(roc_auc_score(y_test, y_prob)) if len(np.unique(y_test)) > 1 else 0.5
    cm = confusion_matrix(y_test, y_pred).tolist()

    fpr, tpr, thresholds = roc_curve(y_test, y_prob)

    return {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "confusion_matrix": cm,
        "true_negatives": cm[0][0],
        "false_positives": cm[0][1],
        "false_negatives": cm[1][0],
        "true_positives": cm[1][1],
        "roc_curve": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist()
        }
    }


def build_comparison_dataframe(evaluations: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Constructs a structured pandas DataFrame summarizing performance metrics across models.
    """
    records = []
    for model_name, metrics in evaluations.items():
        records.append({
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1-score": metrics["f1_score"],
            "ROC-AUC": metrics["roc_auc"]
        })
    df = pd.DataFrame(records)
    # Sort primarily by F1-score and Recall (critical for slope hazard screening)
    df = df.sort_values(by=["F1-score", "Recall", "ROC-AUC"], ascending=False).reset_index(drop=True)
    return df
