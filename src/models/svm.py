# src/models/svm.py
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

def build_svm(preprocessor) -> Pipeline:
    model = SVC(probability=True)  # nécessaire pour ROC-AUC
    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])
