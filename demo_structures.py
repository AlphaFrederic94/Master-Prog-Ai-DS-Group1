# src/structures/demo_structures.py

import pandas as pd

from src.config import CFG
from src.data.build_dataset import build_dataset
from src.structures import HashEncoder, PatientBST, FeatureGraph


def ensure_processed_dataset():
    """
    Vérifie que le dataset prétraité existe.
    Le construit à partir du zip UCI si nécessaire.
    """
    if not CFG.PROCESSED_CSV.exists():
        build_dataset(
            zip_path=CFG.RAW_ZIP,
            out_path=CFG.PROCESSED_CSV
        )


def main():
    # ===============================
    # Chargement des données
    # ===============================
    ensure_processed_dataset()
    df = pd.read_csv(CFG.PROCESSED_CSV)

    # ===============================
    # 1) HASH TABLE : HashEncoder
    # ===============================
    he = HashEncoder()
    he.fit(df["age_group"].astype(str).tolist())

    encoded_sample = he.transform(df["age_group"].astype(str).tolist())[:10]
    print(
        f"[HashEncoder] nb catégories = {len(he)} ; sample encodage:",
        encoded_sample
    )

    # ===============================
    # 2) TREE : PatientBST
    # ===============================
    records = (
        df[["age", "sex", "chol", "trestbps", "target"]]
        .head(80)
        .to_dict(orient="records")
    )

    bst = PatientBST(key_fn=lambda r: r["age"])
    for r in records:
        bst.insert(r)

    searched_age = float(records[10]["age"])
    found_record = bst.search(searched_age)

    print(
        f"[PatientBST] taille = {len(bst)} ; search(age={searched_age}):",
        found_record
    )

    # ===============================
    # 3) GRAPH : FeatureGraph
    # ===============================
    # On exclut les variables dérivées pour éviter
    # des corrélations artificielles
    graph_df = df.drop(
        columns=[
            "target",
            "weight_est",
            "bmi_est",
            "age_chol_interaction"
        ],
        errors="ignore"
    )

    # Construction du graphe
    g = FeatureGraph.from_dataframe(
        graph_df,
        threshold=0.35
    )

    top_edges = g.top_edges(10)

    print(
        f"[FeatureGraph] nodes={len(g.nodes())} ; top edges:",
        top_edges
    )

    # Élément "rapportable"
    print(
        "[FeatureGraph] number_of_edges =",
        len(top_edges)
    )


if __name__ == "__main__":
    main()
