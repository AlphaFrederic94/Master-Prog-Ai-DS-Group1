# src/models/logistic_regression.py
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

def build_logreg(preprocessor) -> Pipeline:
    model = LogisticRegression(max_iter=2000)
    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])
