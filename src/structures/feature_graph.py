# src/structures/feature_graph.py
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd

@dataclass
class Edge:
    to: str
    weight: float

class FeatureGraph:
    """
    Graphe de caractéristiques:
    - nœuds: features numériques
    - arêtes: corrélations |corr| >= threshold
    Stockage: liste d'adjacence.
    """
    def __init__(self):
        self.adj: Dict[str, List[Edge]] = {}

    def add_node(self, node: str):
        self.adj.setdefault(node, [])

    def add_edge(self, u: str, v: str, weight: float):
        self.add_node(u)
        self.add_node(v)
        self.adj[u].append(Edge(to=v, weight=weight))
        self.adj[v].append(Edge(to=u, weight=weight))

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, threshold: float = 0.5) -> "FeatureGraph":
        g = cls()
        corr = df.corr(numeric_only=True)

        cols = list(corr.columns)
        for c in cols:
            g.add_node(c)

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                u, v = cols[i], cols[j]
                w = float(corr.loc[u, v])
                if abs(w) >= threshold:
                    g.add_edge(u, v, w)
        return g

    def top_edges(self, k: int = 10) -> List[Tuple[str, str, float]]:
        edges = []
        seen = set()
        for u, neigh in self.adj.items():
            for e in neigh:
                v = e.to
                key = tuple(sorted([u, v]))
                if key in seen:
                    continue
                seen.add(key)
                edges.append((u, v, e.weight))
        edges.sort(key=lambda t: abs(t[2]), reverse=True)
        return edges[:k]

    def degree(self, node: str) -> int:
        return len(self.adj.get(node, []))

    def nodes(self):
        return list(self.adj.keys())
