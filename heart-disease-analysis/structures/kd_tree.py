"""
KD-Tree Implementation
======================

A k-dimensional tree for efficient nearest neighbor search in
multi-dimensional space. Optimized for patient similarity queries.

Time Complexity:
---------------
- Tree construction: O(n log²n)
- KNN search: O(log n) average, O(n) worst case
- Range query: O(n^(1-1/k) + m) where m = results

Space Complexity: O(n)

Author: Senior Data Scientist
Date: February 2026
"""

from typing import List, Tuple, Optional
import numpy as np
import heapq
from dataclasses import dataclass


@dataclass
class KDNode:
    """
    Node in a K-dimensional tree.
    
    Attributes:
    -----------
    point : np.ndarray
        The k-dimensional point (feature vector)
    patient_id : int
        Identifier for the patient
    axis : int
        Splitting axis for this node
    left : Optional[KDNode]
        Left subtree (points with smaller value on axis)
    right : Optional[KDNode]
        Right subtree (points with larger value on axis)
    """
    point: np.ndarray
    patient_id: int
    axis: int
    left: Optional['KDNode'] = None
    right: Optional['KDNode'] = None


class KDTree:
    """
    K-dimensional tree for efficient nearest neighbor search.
    
    Supports k-nearest neighbors (KNN) and range queries for finding
    similar patients based on continuous features.
    
    Parameters:
    -----------
    data : np.ndarray
        2D array of shape (n_samples, n_features)
    patient_ids : List[int]
        List of patient identifiers corresponding to data rows
        
    Attributes:
    -----------
    root : KDNode
        Root node of the tree
    k : int
        Number of dimensions (features)
    patient_map : Dict[int, np.ndarray]
        Mapping from patient ID to feature vector
        
    Examples:
    ---------
    >>> data = np.array([[45, 120, 200], [50, 130, 220], [55, 140, 210]])
    >>> patient_ids = [1, 2, 3]
    >>> tree = KDTree(data, patient_ids)
    >>> neighbors = tree.k_nearest_neighbors([48, 125, 205], k=2)
    >>> print(neighbors)  # [(1, distance1), (2, distance2)]
    """
    
    def __init__(self, data: np.ndarray, patient_ids: List[int]):
        """Initialize KD-Tree from data."""
        if len(data) != len(patient_ids):
            raise ValueError("Data and patient_ids must have same length")
        
        if len(data) == 0:
            raise ValueError("Cannot build tree from empty data")
        
        self.k = data.shape[1]  # Number of dimensions
        self.patient_map = {pid: point for pid, point in zip(patient_ids, data)}
        
        # Build tree
        points_with_ids = list(zip(data, patient_ids))
        self.root = self._build_tree(points_with_ids, depth=0)
    
    def _build_tree(self, points_with_ids: List[Tuple[np.ndarray, int]], 
                   depth: int = 0) -> Optional[KDNode]:
        """
        Recursively build KD-Tree.
        
        Parameters:
        -----------
        points_with_ids : List[Tuple[np.ndarray, int]]
            List of (point, patient_id) tuples
        depth : int
            Current depth in tree
            
        Returns:
        --------
        Optional[KDNode]
            Root node of subtree
        """
        if not points_with_ids:
            return None
        
        # Select axis based on depth (cycle through dimensions)
        axis = depth % self.k
        
        # Sort points by current axis and select median
        points_with_ids.sort(key=lambda x: x[0][axis])
        median_idx = len(points_with_ids) // 2
        
        median_point, median_id = points_with_ids[median_idx]
        
        # Create node and recursively build subtrees
        node = KDNode(
            point=median_point,
            patient_id=median_id,
            axis=axis,
            left=self._build_tree(points_with_ids[:median_idx], depth + 1),
            right=self._build_tree(points_with_ids[median_idx + 1:], depth + 1)
        )
        
        return node
    
    def _euclidean_distance(self, point1: np.ndarray, point2: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two points.
        
        Parameters:
        -----------
        point1 : np.ndarray
            First point
        point2 : np.ndarray
            Second point
            
        Returns:
        --------
        float
            Euclidean distance
        """
        return np.sqrt(np.sum((point1 - point2) ** 2))
    
    def k_nearest_neighbors(self, query_point: np.ndarray, k: int = 1) -> List[Tuple[int, float]]:
        """
        Find k nearest neighbors to query point.
        
        Uses a max-heap to efficiently track k nearest neighbors during
        tree traversal with branch-and-bound pruning.
        
        Parameters:
        -----------
        query_point : np.ndarray
            Query point (feature vector)
        k : int, default=1
            Number of neighbors to find
            
        Returns:
        --------
        List[Tuple[int, float]]
            List of (patient_id, distance) tuples, sorted by distance
        """
        if k <= 0:
            raise ValueError("k must be positive")
        
        query_point = np.array(query_point)
        
        if len(query_point) != self.k:
            raise ValueError(f"Query point must have {self.k} dimensions")
        
        # Max-heap to store k nearest neighbors (negative distance for max-heap)
        heap = []
        
        def search(node: Optional[KDNode]) -> None:
            """Recursive search with pruning."""
            if node is None:
                return
            
            # Calculate distance to current node
            distance = self._euclidean_distance(query_point, node.point)
            
            # Add to heap if we have fewer than k neighbors or this is closer
            if len(heap) < k:
                heapq.heappush(heap, (-distance, node.patient_id, distance))
            elif distance < -heap[0][0]:  # heap[0][0] is negative max distance
                heapq.heapreplace(heap, (-distance, node.patient_id, distance))
            
            # Determine which subtree to search first
            axis = node.axis
            if query_point[axis] < node.point[axis]:
                near_subtree = node.left
                far_subtree = node.right
            else:
                near_subtree = node.right
                far_subtree = node.left
            
            # Search near subtree
            search(near_subtree)
            
            # Check if we need to search far subtree
            # Only search if the splitting plane could contain closer points
            if len(heap) < k or abs(query_point[axis] - node.point[axis]) < -heap[0][0]:
                search(far_subtree)
        
        # Start search from root
        search(self.root)
        
        # Extract results and sort by distance
        results = [(pid, dist) for _, pid, dist in heap]
        results.sort(key=lambda x: x[1])
        
        return results
    
    def range_query(self, query_point: np.ndarray, radius: float) -> List[int]:
        """
        Find all patients within radius of query point.
        
        Parameters:
        -----------
        query_point : np.ndarray
            Query point (feature vector)
        radius : float
            Search radius
            
        Returns:
        --------
        List[int]
            List of patient IDs within radius
        """
        query_point = np.array(query_point)
        
        if len(query_point) != self.k:
            raise ValueError(f"Query point must have {self.k} dimensions")
        
        results = []
        
        def search(node: Optional[KDNode]) -> None:
            """Recursive range search."""
            if node is None:
                return
            
            # Check if current node is within radius
            distance = self._euclidean_distance(query_point, node.point)
            if distance <= radius:
                results.append(node.patient_id)
            
            # Determine which subtrees to search
            axis = node.axis
            axis_distance = abs(query_point[axis] - node.point[axis])
            
            # Search both subtrees if splitting plane intersects sphere
            if query_point[axis] - radius <= node.point[axis]:
                search(node.left)
            if query_point[axis] + radius >= node.point[axis]:
                search(node.right)
        
        search(self.root)
        return results
    
    def nearest_neighbor(self, query_point: np.ndarray) -> Tuple[int, float]:
        """
        Find single nearest neighbor.
        
        Parameters:
        -----------
        query_point : np.ndarray
            Query point (feature vector)
            
        Returns:
        --------
        Tuple[int, float]
            (patient_id, distance) of nearest neighbor
        """
        result = self.k_nearest_neighbors(query_point, k=1)
        return result[0] if result else (None, float('inf'))
    
    def get_patient_features(self, patient_id: int) -> Optional[np.ndarray]:
        """
        Get feature vector for a patient.
        
        Parameters:
        -----------
        patient_id : int
            Patient identifier
            
        Returns:
        --------
        Optional[np.ndarray]
            Feature vector if patient exists, None otherwise
        """
        return self.patient_map.get(patient_id)
    
    def get_tree_depth(self) -> int:
        """
        Calculate depth of the tree.
        
        Returns:
        --------
        int
            Maximum depth of tree
        """
        def depth(node: Optional[KDNode]) -> int:
            if node is None:
                return 0
            return 1 + max(depth(node.left), depth(node.right))
        
        return depth(self.root)
    
    def get_statistics(self) -> dict:
        """
        Get tree statistics.
        
        Returns:
        --------
        dict
            Dictionary containing:
            - num_points: Number of points in tree
            - dimensions: Number of dimensions
            - depth: Tree depth
            - balance_factor: Ratio of actual to optimal depth
        """
        num_points = len(self.patient_map)
        depth = self.get_tree_depth()
        optimal_depth = np.ceil(np.log2(num_points + 1))
        balance_factor = depth / optimal_depth if optimal_depth > 0 else 1.0
        
        return {
            'num_points': num_points,
            'dimensions': self.k,
            'depth': depth,
            'optimal_depth': int(optimal_depth),
            'balance_factor': balance_factor
        }
    
    def __len__(self) -> int:
        """Return number of points in tree."""
        return len(self.patient_map)
    
    def __str__(self) -> str:
        """String representation of tree."""
        stats = self.get_statistics()
        return (f"KDTree(points={stats['num_points']}, "
                f"dimensions={stats['dimensions']}, "
                f"depth={stats['depth']})")
    
    def __repr__(self) -> str:
        """Representation of tree."""
        return self.__str__()
