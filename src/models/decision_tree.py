# src/models/decision_tree.py
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

def build_tree(preprocessor) -> Pipeline:
    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=None
    )
    return Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model),
    ])
