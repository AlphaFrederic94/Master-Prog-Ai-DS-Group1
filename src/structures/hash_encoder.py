# src/structures/hash_encoder.py
from typing import Dict, Any

class HashEncoder:
    """
    Table de hachage (hash table) pour encoder des valeurs catégorielles.
    Exemple: "male"->0, "female"->1, etc.
    """
    def __init__(self):
        self.mapping: Dict[Any, int] = {}

    def fit(self, values):
        for v in values:
            if v not in self.mapping:
                self.mapping[v] = len(self.mapping)
        return self

    def transform(self, values):
        return [self.mapping.get(v, -1) for v in values]

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def __len__(self):
        return len(self.mapping)
