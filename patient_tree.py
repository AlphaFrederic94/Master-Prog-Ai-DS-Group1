# src/structures/patient_tree.py
from dataclasses import dataclass
from typing import Optional, Any, Callable, List

@dataclass
class _Node:
    key: float
    value: Any
    left: Optional["_Node"] = None
    right: Optional["_Node"] = None

class PatientBST:
    """
    Arbre binaire de recherche (BST) pour organiser des patients par une clé.
    Exemples d'usage:
      - insertion de dossiers patients indexés par âge
      - recherche rapide par âge
    """
    def __init__(self, key_fn: Callable[[Any], float]):
        self.root: Optional[_Node] = None
        self.key_fn = key_fn
        self.size = 0

    def insert(self, item: Any):
        k = float(self.key_fn(item))
        self.root = self._insert(self.root, k, item)
        self.size += 1

    def _insert(self, node: Optional[_Node], key: float, value: Any) -> _Node:
        if node is None:
            return _Node(key=key, value=value)
        if key < node.key:
            node.left = self._insert(node.left, key, value)
        else:
            node.right = self._insert(node.right, key, value)
        return node

    def search(self, key: float) -> Optional[Any]:
        node = self.root
        while node is not None:
            if key == node.key:
                return node.value
            node = node.left if key < node.key else node.right
        return None

    def inorder(self) -> List[Any]:
        out: List[Any] = []
        self._inorder(self.root, out)
        return out

    def _inorder(self, node: Optional[_Node], out: List[Any]):
        if node is None:
            return
        self._inorder(node.left, out)
        out.append(node.value)
        self._inorder(node.right, out)

    def __len__(self):
        return self.size
