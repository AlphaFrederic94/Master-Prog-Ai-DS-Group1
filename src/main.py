# src/main.py
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import CFG
from src.data.build_dataset import build_dataset
from src.data.transform import build_preprocessor
from src.models import build_logreg, build_tree, build_svm
from src.evaluation.metrics import evaluate_model, as_dict
from src.visualization.plots import (
    plot_target_distribution, plot_confusion_matrix, plot_roc_curve
)
from src.structures import HashEncoder, PatientBST, FeatureGraph


def ensure_processed_dataset():
    if not CFG.PROCESSED_CSV.exists():
        build_dataset(zip_path=CFG.RAW_ZIP, out_path=CFG.PROCESSED_CSV)


def main():
    ensure_processed_dataset()
    df = pd.read_csv(CFG.PROCESSED_CSV)

    # ===== Structures de données (Graph / Tree / Hash) =====

    # 1) Hash table demo : encoder age_group
    he = HashEncoder()
    he.fit(df["age_group"].astype(str).tolist())
    encoded_age_group = he.transform(df["age_group"].astype(str).tolist())
    print(f"\n[HashEncoder] catégories age_group = {len(he)} ; exemple encodage: {encoded_age_group[:10]}")

    # 2) Tree demo : BST indexé par âge (organiser des patients)
    records = df[["age", "sex", "chol", "trestbps", "target"]].head(80).to_dict(orient="records")
    bst = PatientBST(key_fn=lambda r: r["age"])
    for r in records:
        bst.insert(r)

    found = bst.search(float(records[10]["age"]))
    print(f"[PatientBST] taille = {len(bst)} ; recherche(age={records[10]['age']}):", found)

    # 3) Feature graph demo : corrélations entre features (on exclut les features dérivées)
    graph_df = df.drop(columns=["target", "weight_est", "bmi_est"], errors="ignore")
    g = FeatureGraph.from_dataframe(graph_df, threshold=0.6)
    print("[FeatureGraph] nodes=", len(g.nodes()), "; top edges:", g.top_edges(8))

    # ===== Visualisation dataset =====
    plot_target_distribution(df, CFG.TARGET)

    # ===== ML =====
    X = df.drop(columns=[CFG.TARGET])
    y = df[CFG.TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CFG.TEST_SIZE,
        random_state=CFG.RANDOM_STATE,
        stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    models = {
        "LogReg": build_logreg(preprocessor),
        "DecisionTree": build_tree(preprocessor),
        "SVM": build_svm(preprocessor),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)

        res = evaluate_model(name, model, X_test, y_test)
        results.append(as_dict(res))

        plot_confusion_matrix(name, model, X_test, y_test)
        plot_roc_curve(name, model, X_test, y_test)

    results_df = pd.DataFrame(results).sort_values(by="roc_auc", ascending=False)
    print("\n=== Model comparison (sorted by ROC-AUC) ===")
    print(results_df.to_string(index=False))

    # Export du tableau pour le rapport
    CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = CFG.OUTPUT_DIR / "model_comparison.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
