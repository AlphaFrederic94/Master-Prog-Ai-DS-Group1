# src/evaluation/metrics.py
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

@dataclass
class EvalResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: list

def _predict_proba_safe(model, X):
    # Certains modèles ont predict_proba, d'autres decision_function
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # normalisation simple vers [0,1]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        return scores
    # fallback
    return model.predict(X)

def evaluate_model(name: str, model, X_test, y_test) -> EvalResult:
    y_pred = model.predict(X_test)
    y_score = _predict_proba_safe(model, X_test)

    return EvalResult(
        name=name,
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, y_score)),
        confusion=confusion_matrix(y_test, y_pred).tolist()
    )

def as_dict(res: EvalResult) -> Dict[str, Any]:
    return {
        "model": res.name,
        "accuracy": res.accuracy,
        "precision": res.precision,
        "recall": res.recall,
        "f1": res.f1,
        "roc_auc": res.roc_auc,
        "confusion": res.confusion,
    }
