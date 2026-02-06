import pandas as pd
from src.structures import FeatureGraph

def test_feature_graph_build_nodes_edges():
    # f2 parfaitement corrélée à f1
    df = pd.DataFrame({
        "f1": [1, 2, 3, 4],
        "f2": [2, 4, 6, 8],
        "f3": [1, 1, 1, 1],
    })

    g = FeatureGraph.from_dataframe(df, threshold=0.9)

    # nodes présents
    nodes = g.nodes()
    assert "f1" in nodes and "f2" in nodes and "f3" in nodes

    # f1-f2 doit être une top edge
    top = g.top_edges(5)
    pairs = {tuple(sorted([u, v])) for (u, v, w) in top}
    assert ("f1", "f2") in pairs

def test_feature_graph_degree():
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [1, 2, 3, 4],  # corrélation parfaite avec a
        "c": [4, 3, 2, 1],  # corrélation -1 avec a
    })
    g = FeatureGraph.from_dataframe(df, threshold=0.9)
    # a connecté à b et c
    assert g.degree("a") >= 2
