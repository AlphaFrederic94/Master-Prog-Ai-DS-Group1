# src/visualization/plots.py
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

from src.config import CFG

def _ensure_out() -> Path:
    CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return CFG.OUTPUT_DIR

def plot_target_distribution(df: pd.DataFrame, target_col: str):
    out = _ensure_out()
    plt.figure()
    df[target_col].value_counts().plot(kind="bar")
    plt.title("Target distribution (0=no disease, 1=disease)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    path = out / "target_distribution.png"
    plt.savefig(path, dpi=CFG.FIG_DPI)
    plt.close()
    return path

def plot_confusion_matrix(model_name: str, model, X_test, y_test):
    out = _ensure_out()
    disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    disp.ax_.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    path = out / f"confusion_{model_name}.png"
    plt.savefig(path, dpi=CFG.FIG_DPI)
    plt.close()
    return path

def plot_roc_curve(model_name: str, model, X_test, y_test):
    out = _ensure_out()
    RocCurveDisplay.from_estimator(model, X_test, y_test)
    plt.title(f"ROC Curve - {model_name}")
    plt.tight_layout()
    path = out / f"roc_{model_name}.png"
    plt.savefig(path, dpi=CFG.FIG_DPI)
    plt.close()
    return path
