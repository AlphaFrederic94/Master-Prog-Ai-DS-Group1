"""
Feature Graph Implementation
============================

A graph-based representation of feature correlations for analyzing
multicollinearity and performing intelligent feature selection.

Nodes represent features, edges represent correlations weighted by
correlation coefficients. Supports graph algorithms for feature analysis.

Time Complexity:
---------------
- Graph construction: O(n²) where n = number of features
- Add edge: O(1)
- Get neighbors: O(degree)
- Connected components (DFS): O(V + E)
- MST (Kruskal's): O(E log E)

Author: Senior Data Scientist
Date: February 2026
"""

from typing import List, Set, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from collections import defaultdict, deque


class FeatureGraph:
    """
    Weighted undirected graph representing feature correlations.
    
    Nodes are features, edges are weighted by absolute correlation coefficients.
    Provides methods for multicollinearity detection and feature selection.
    
    Parameters:
    -----------
    correlation_matrix : pd.DataFrame
        Correlation matrix with features as both index and columns
    threshold : float, default=0.3
        Minimum absolute correlation to create an edge
        
    Attributes:
    -----------
    nodes : Set[str]
        Set of all feature names
    adjacency_list : Dict[str, List[Tuple[str, float]]]
        Adjacency list representation: {feature: [(neighbor, weight), ...]}
        
    Examples:
    ---------
    >>> corr_matrix = df.corr()
    >>> graph = FeatureGraph(corr_matrix, threshold=0.3)
    >>> clusters = graph.detect_multicollinearity_clusters(threshold=0.7)
    >>> central_features = graph.get_central_features()
    """
    
    def __init__(self, correlation_matrix: pd.DataFrame, threshold: float = 0.3):
        """Initialize feature graph from correlation matrix."""
        self.nodes: Set[str] = set()
        self.adjacency_list: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.threshold = threshold
        
        # Build graph from correlation matrix
        self._build_from_correlation_matrix(correlation_matrix)
    
    def _build_from_correlation_matrix(self, corr_matrix: pd.DataFrame) -> None:
        """
        Build graph from correlation matrix.
        
        Creates edges between features with absolute correlation above threshold.
        
        Parameters:
        -----------
        corr_matrix : pd.DataFrame
            Correlation matrix
        """
        features = corr_matrix.columns.tolist()
        
        # Add all features as nodes
        for feature in features:
            self.nodes.add(feature)
            if feature not in self.adjacency_list:
                self.adjacency_list[feature] = []
        
        # Add edges for correlations above threshold
        for i, feature1 in enumerate(features):
            for j, feature2 in enumerate(features):
                if i < j:  # Avoid duplicates and self-loops
                    corr_value = abs(corr_matrix.loc[feature1, feature2])
                    
                    if corr_value >= self.threshold and corr_value < 1.0:
                        self.add_edge(feature1, feature2, corr_value)
    
    def add_edge(self, feature1: str, feature2: str, weight: float) -> None:
        """
        Add an edge between two features.
        
        Parameters:
        -----------
        feature1 : str
            First feature name
        feature2 : str
            Second feature name
        weight : float
            Edge weight (correlation coefficient)
        """
        # Add nodes if they don't exist
        self.nodes.add(feature1)
        self.nodes.add(feature2)
        
        # Add edge in both directions (undirected graph)
        self.adjacency_list[feature1].append((feature2, weight))
        self.adjacency_list[feature2].append((feature1, weight))
    
    def get_neighbors(self, feature: str) -> List[Tuple[str, float]]:
        """
        Get all neighbors of a feature with edge weights.
        
        Parameters:
        -----------
        feature : str
            Feature name
            
        Returns:
        --------
        List[Tuple[str, float]]
            List of (neighbor, weight) tuples
        """
        return self.adjacency_list.get(feature, [])
    
    def get_degree(self, feature: str) -> int:
        """
        Get degree (number of neighbors) of a feature.
        
        Parameters:
        -----------
        feature : str
            Feature name
            
        Returns:
        --------
        int
            Degree of the feature
        """
        return len(self.adjacency_list.get(feature, []))
    
    def find_connected_components(self) -> List[Set[str]]:
        """
        Find all connected components using DFS.
        
        A connected component is a maximal set of features where
        each feature is reachable from every other feature.
        
        Returns:
        --------
        List[Set[str]]
            List of connected components (sets of feature names)
        """
        visited = set()
        components = []
        
        def dfs(node: str, component: Set[str]) -> None:
            """Depth-first search to explore component."""
            visited.add(node)
            component.add(node)
            
            for neighbor, _ in self.get_neighbors(node):
                if neighbor not in visited:
                    dfs(neighbor, component)
        
        # Explore each unvisited node
        for node in self.nodes:
            if node not in visited:
                component = set()
                dfs(node, component)
                components.append(component)
        
        return components
    
    def detect_multicollinearity_clusters(self, threshold: float = 0.7) -> List[Set[str]]:
        """
        Detect clusters of highly correlated features (multicollinearity).
        
        Finds connected components in a subgraph containing only edges
        with weight >= threshold.
        
        Parameters:
        -----------
        threshold : float, default=0.7
            Minimum correlation to consider features as multicollinear
            
        Returns:
        --------
        List[Set[str]]
            List of multicollinear feature clusters
        """
        # Create subgraph with only high-correlation edges
        high_corr_graph = FeatureGraph.__new__(FeatureGraph)
        high_corr_graph.nodes = self.nodes.copy()
        high_corr_graph.adjacency_list = defaultdict(list)
        high_corr_graph.threshold = threshold
        
        # Add only high-correlation edges
        for feature in self.nodes:
            for neighbor, weight in self.get_neighbors(feature):
                if weight >= threshold:
                    # Avoid duplicates by checking if edge already added
                    if not any(n == feature for n, _ in high_corr_graph.adjacency_list[neighbor]):
                        high_corr_graph.adjacency_list[feature].append((neighbor, weight))
        
        # Find connected components in high-correlation subgraph
        clusters = high_corr_graph.find_connected_components()
        
        # Filter out single-node components
        return [cluster for cluster in clusters if len(cluster) > 1]
    
    def get_central_features(self, method: str = 'degree') -> List[str]:
        """
        Get most central features based on centrality measure.
        
        Parameters:
        -----------
        method : str, default='degree'
            Centrality measure to use:
            - 'degree': Features with most connections
            - 'weighted_degree': Features with highest sum of edge weights
            
        Returns:
        --------
        List[str]
            Features sorted by centrality (descending)
        """
        if method == 'degree':
            # Sort by number of neighbors
            centrality = {feature: self.get_degree(feature) for feature in self.nodes}
        
        elif method == 'weighted_degree':
            # Sort by sum of edge weights
            centrality = {}
            for feature in self.nodes:
                total_weight = sum(weight for _, weight in self.get_neighbors(feature))
                centrality[feature] = total_weight
        
        else:
            raise ValueError(f"Unknown centrality method: {method}")
        
        # Sort features by centrality (descending)
        sorted_features = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
        return [feature for feature, _ in sorted_features]
    
    def maximum_spanning_tree(self) -> 'FeatureGraph':
        """
        Compute Maximum Spanning Tree using Kruskal's algorithm.
        
        MST connects all features with maximum total correlation while
        avoiding cycles. Useful for feature selection.
        
        Returns:
        --------
        FeatureGraph
            New graph containing only MST edges
        """
        # Get all edges
        edges = []
        seen_edges = set()
        
        for feature in self.nodes:
            for neighbor, weight in self.get_neighbors(feature):
                edge = tuple(sorted([feature, neighbor]))
                if edge not in seen_edges:
                    edges.append((weight, feature, neighbor))
                    seen_edges.add(edge)
        
        # Sort edges by weight (descending for maximum)
        edges.sort(reverse=True)
        
        # Union-Find data structure for cycle detection
        parent = {node: node for node in self.nodes}
        rank = {node: 0 for node in self.nodes}
        
        def find(node: str) -> str:
            """Find root of node's set."""
            if parent[node] != node:
                parent[node] = find(parent[node])  # Path compression
            return parent[node]
        
        def union(node1: str, node2: str) -> bool:
            """Union two sets. Returns True if successful."""
            root1, root2 = find(node1), find(node2)
            
            if root1 == root2:
                return False  # Already in same set (would create cycle)
            
            # Union by rank
            if rank[root1] < rank[root2]:
                parent[root1] = root2
            elif rank[root1] > rank[root2]:
                parent[root2] = root1
            else:
                parent[root2] = root1
                rank[root1] += 1
            
            return True
        
        # Build MST
        mst = FeatureGraph.__new__(FeatureGraph)
        mst.nodes = self.nodes.copy()
        mst.adjacency_list = defaultdict(list)
        mst.threshold = self.threshold
        
        for weight, feature1, feature2 in edges:
            if union(feature1, feature2):
                mst.add_edge(feature1, feature2, weight)
        
        return mst
    
    def select_features_from_clusters(self, clusters: List[Set[str]], 
                                     method: str = 'degree') -> List[str]:
        """
        Select one representative feature from each multicollinearity cluster.
        
        Parameters:
        -----------
        clusters : List[Set[str]]
            List of feature clusters
        method : str, default='degree'
            Method to select representative ('degree' or 'weighted_degree')
            
        Returns:
        --------
        List[str]
            Selected representative features
        """
        selected = []
        
        for cluster in clusters:
            # Get centrality for features in this cluster
            if method == 'degree':
                centrality = {f: self.get_degree(f) for f in cluster}
            else:  # weighted_degree
                centrality = {
                    f: sum(w for _, w in self.get_neighbors(f)) 
                    for f in cluster
                }
            
            # Select feature with highest centrality
            best_feature = max(centrality.items(), key=lambda x: x[1])[0]
            selected.append(best_feature)
        
        return selected
    
    def get_edge_count(self) -> int:
        """
        Get total number of edges in the graph.
        
        Returns:
        --------
        int
            Number of edges
        """
        total = sum(len(neighbors) for neighbors in self.adjacency_list.values())
        return total // 2  # Divide by 2 because graph is undirected
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get graph statistics.
        
        Returns:
        --------
        Dict[str, any]
            Dictionary containing:
            - num_nodes: Number of features
            - num_edges: Number of edges
            - avg_degree: Average degree
            - max_degree: Maximum degree
            - density: Graph density
        """
        num_nodes = len(self.nodes)
        num_edges = self.get_edge_count()
        degrees = [self.get_degree(f) for f in self.nodes]
        
        max_possible_edges = num_nodes * (num_nodes - 1) / 2
        density = num_edges / max_possible_edges if max_possible_edges > 0 else 0
        
        return {
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_degree': np.mean(degrees) if degrees else 0,
            'max_degree': max(degrees) if degrees else 0,
            'density': density
        }
    
    def __str__(self) -> str:
        """String representation of graph."""
        stats = self.get_statistics()
        return (f"FeatureGraph(nodes={stats['num_nodes']}, "
                f"edges={stats['num_edges']}, "
                f"density={stats['density']:.3f})")
    
    def __repr__(self) -> str:
        """Representation of graph."""
        return self.__str__()
