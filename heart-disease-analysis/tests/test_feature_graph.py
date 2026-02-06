"""
Unit Tests for Feature Graph
=============================

Tests for feature graph implementation including graph construction,
connected components, MST, and multicollinearity detection.

Author: Senior Data Scientist
Date: February 2026
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from structures.feature_graph import FeatureGraph


def test_graph_construction():
    """Test graph construction from correlation matrix."""
    print("Testing Feature Graph Construction...")
    
    # Create sample correlation matrix
    corr_data = {
        'age': [1.0, 0.2, 0.5, 0.1],
        'chol': [0.2, 1.0, 0.8, 0.3],
        'trestbps': [0.5, 0.8, 1.0, 0.2],
        'thalach': [0.1, 0.3, 0.2, 1.0]
    }
    corr_matrix = pd.DataFrame(corr_data, index=['age', 'chol', 'trestbps', 'thalach'])
    
    graph = FeatureGraph(corr_matrix, threshold=0.3)
    
    # Check nodes
    assert len(graph.nodes) == 4, "Should have 4 nodes"
    assert 'age' in graph.nodes, "Should contain 'age'"
    
    # Check edges (correlations above 0.3)
    age_neighbors = graph.get_neighbors('age')
    assert len(age_neighbors) == 1, "Age should have 1 neighbor (trestbps with 0.5)"
    
    chol_neighbors = graph.get_neighbors('chol')
    assert len(chol_neighbors) == 2, "Chol should have 2 neighbors"
    
    print("✓ Graph construction passed")


def test_connected_components():
    """Test connected components detection."""
    print("Testing Connected Components...")
    
    # Create correlation matrix with two separate components
    corr_data = {
        'f1': [1.0, 0.8, 0.1, 0.0],
        'f2': [0.8, 1.0, 0.0, 0.1],
        'f3': [0.1, 0.0, 1.0, 0.7],
        'f4': [0.0, 0.1, 0.7, 1.0]
    }
    corr_matrix = pd.DataFrame(corr_data, index=['f1', 'f2', 'f3', 'f4'])
    
    graph = FeatureGraph(corr_matrix, threshold=0.5)
    
    components = graph.find_connected_components()
    
    # Should have 2 components: {f1, f2} and {f3, f4}
    assert len(components) == 2, f"Should have 2 components, got {len(components)}"
    
    component_sizes = sorted([len(c) for c in components])
    assert component_sizes == [2, 2], "Each component should have 2 features"
    
    print("✓ Connected components passed")


def test_multicollinearity_detection():
    """Test multicollinearity cluster detection."""
    print("Testing Multicollinearity Detection...")
    
    # Create correlation matrix with high correlation cluster
    corr_data = {
        'f1': [1.0, 0.9, 0.85, 0.2],
        'f2': [0.9, 1.0, 0.88, 0.1],
        'f3': [0.85, 0.88, 1.0, 0.15],
        'f4': [0.2, 0.1, 0.15, 1.0]
    }
    corr_matrix = pd.DataFrame(corr_data, index=['f1', 'f2', 'f3', 'f4'])
    
    graph = FeatureGraph(corr_matrix, threshold=0.3)
    
    clusters = graph.detect_multicollinearity_clusters(threshold=0.8)
    
    # Should detect one cluster with f1, f2, f3
    assert len(clusters) > 0, "Should detect at least one cluster"
    
    largest_cluster = max(clusters, key=len)
    assert len(largest_cluster) == 3, "Largest cluster should have 3 features"
    
    print(f"✓ Multicollinearity detection passed (found {len(clusters)} cluster(s))")


def test_centrality_measures():
    """Test centrality-based feature ranking."""
    print("Testing Centrality Measures...")
    
    corr_data = {
        'f1': [1.0, 0.7, 0.6, 0.5],
        'f2': [0.7, 1.0, 0.4, 0.3],
        'f3': [0.6, 0.4, 1.0, 0.2],
        'f4': [0.5, 0.3, 0.2, 1.0]
    }
    corr_matrix = pd.DataFrame(corr_data, index=['f1', 'f2', 'f3', 'f4'])
    
    graph = FeatureGraph(corr_matrix, threshold=0.3)
    
    # Test degree centrality
    central_features = graph.get_central_features(method='degree')
    assert central_features[0] == 'f1', "f1 should be most central by degree"
    
    # Test weighted degree centrality
    central_features_weighted = graph.get_central_features(method='weighted_degree')
    assert len(central_features_weighted) == 4, "Should rank all features"
    
    print("✓ Centrality measures passed")


def test_maximum_spanning_tree():
    """Test MST construction."""
    print("Testing Maximum Spanning Tree...")
    
    corr_data = {
        'f1': [1.0, 0.8, 0.3, 0.2],
        'f2': [0.8, 1.0, 0.7, 0.4],
        'f3': [0.3, 0.7, 1.0, 0.6],
        'f4': [0.2, 0.4, 0.6, 1.0]
    }
    corr_matrix = pd.DataFrame(corr_data, index=['f1', 'f2', 'f3', 'f4'])
    
    graph = FeatureGraph(corr_matrix, threshold=0.3)
    
    mst = graph.maximum_spanning_tree()
    
    # MST should have n-1 edges for n nodes
    assert mst.get_edge_count() == 3, "MST should have 3 edges for 4 nodes"
    
    # All nodes should still be present
    assert len(mst.nodes) == 4, "MST should have all 4 nodes"
    
    print("✓ Maximum spanning tree passed")


def test_graph_statistics():
    """Test graph statistics calculation."""
    print("Testing Graph Statistics...")
    
    corr_data = {
        'f1': [1.0, 0.5, 0.4],
        'f2': [0.5, 1.0, 0.6],
        'f3': [0.4, 0.6, 1.0]
    }
    corr_matrix = pd.DataFrame(corr_data, index=['f1', 'f2', 'f3'])
    
    graph = FeatureGraph(corr_matrix, threshold=0.3)
    
    stats = graph.get_statistics()
    
    assert stats['num_nodes'] == 3, "Should have 3 nodes"
    assert stats['num_edges'] == 3, "Should have 3 edges"
    assert 0 <= stats['density'] <= 1, "Density should be between 0 and 1"
    assert stats['avg_degree'] > 0, "Average degree should be positive"
    
    print(f"✓ Graph statistics passed (density: {stats['density']:.2f})")


def run_all_tests():
    """Run all feature graph tests."""
    print("\n" + "="*60)
    print("FEATURE GRAPH UNIT TESTS")
    print("="*60 + "\n")
    
    test_graph_construction()
    test_connected_components()
    test_multicollinearity_detection()
    test_centrality_measures()
    test_maximum_spanning_tree()
    test_graph_statistics()
    
    print("\n" + "="*60)
    print("✓ ALL FEATURE GRAPH TESTS PASSED")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
