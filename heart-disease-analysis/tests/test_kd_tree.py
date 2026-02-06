"""
Unit Tests for KD-Tree
=======================

Tests for KD-Tree implementation including tree construction,
nearest neighbor search, range queries, and performance.

Author: Senior Data Scientist
Date: February 2026
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from structures.kd_tree import KDTree, KDNode


def test_tree_construction():
    """Test KD-Tree construction."""
    print("Testing KD-Tree Construction...")
    
    # Create sample data
    data = np.array([
        [45, 120, 200],
        [50, 130, 220],
        [55, 140, 210],
        [48, 125, 205],
        [52, 135, 215]
    ])
    patient_ids = [1, 2, 3, 4, 5]
    
    tree = KDTree(data, patient_ids)
    
    assert tree.root is not None, "Tree should have a root"
    assert len(tree) == 5, "Tree should have 5 points"
    assert tree.k == 3, "Tree should have 3 dimensions"
    
    print("✓ Tree construction passed")


def test_nearest_neighbor():
    """Test single nearest neighbor search."""
    print("Testing Nearest Neighbor Search...")
    
    data = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [10, 10]
    ])
    patient_ids = [1, 2, 3, 4]
    
    tree = KDTree(data, patient_ids)
    
    # Query point close to [1, 1]
    query = np.array([1.1, 1.1])
    nearest_id, distance = tree.nearest_neighbor(query)
    
    assert nearest_id == 2, f"Nearest should be patient 2, got {nearest_id}"
    assert distance < 0.2, f"Distance should be small, got {distance}"
    
    # Query point close to [10, 10]
    query = np.array([9.5, 9.5])
    nearest_id, distance = tree.nearest_neighbor(query)
    
    assert nearest_id == 4, f"Nearest should be patient 4, got {nearest_id}"
    
    print("✓ Nearest neighbor search passed")


def test_k_nearest_neighbors():
    """Test k-nearest neighbors search."""
    print("Testing K-Nearest Neighbors...")
    
    data = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [10, 10]
    ])
    patient_ids = [1, 2, 3, 4, 5]
    
    tree = KDTree(data, patient_ids)
    
    # Find 3 nearest neighbors to [0.5, 0.5]
    query = np.array([0.5, 0.5])
    neighbors = tree.k_nearest_neighbors(query, k=3)
    
    assert len(neighbors) == 3, f"Should return 3 neighbors, got {len(neighbors)}"
    
    # Check that results are sorted by distance
    distances = [dist for _, dist in neighbors]
    assert distances == sorted(distances), "Results should be sorted by distance"
    
    # The 3 nearest should be patients 1, 2, 3, or 4 (not 5)
    neighbor_ids = [pid for pid, _ in neighbors]
    assert 5 not in neighbor_ids, "Patient 5 should not be in top 3"
    
    print("✓ K-nearest neighbors passed")


def test_range_query():
    """Test range query."""
    print("Testing Range Query...")
    
    data = np.array([
        [0, 0],
        [1, 0],
        [0, 1],
        [1, 1],
        [5, 5]
    ])
    patient_ids = [1, 2, 3, 4, 5]
    
    tree = KDTree(data, patient_ids)
    
    # Find all points within radius 1.5 of origin
    query = np.array([0, 0])
    results = tree.range_query(query, radius=1.5)
    
    # Should find patients 1, 2, 3, 4 (all within sqrt(2) ≈ 1.41)
    assert len(results) >= 3, f"Should find at least 3 points, got {len(results)}"
    assert 5 not in results, "Patient 5 should not be in range"
    
    print(f"✓ Range query passed (found {len(results)} points)")


def test_tree_balance():
    """Test tree balance and depth."""
    print("Testing Tree Balance...")
    
    # Create larger dataset
    np.random.seed(42)
    data = np.random.rand(100, 3) * 100
    patient_ids = list(range(100))
    
    tree = KDTree(data, patient_ids)
    
    stats = tree.get_statistics()
    
    # Check that tree is reasonably balanced
    assert stats['balance_factor'] < 2.0, f"Tree should be reasonably balanced, got {stats['balance_factor']:.2f}"
    assert stats['depth'] > 0, "Tree should have positive depth"
    
    print(f"✓ Tree balance passed (depth: {stats['depth']}, balance: {stats['balance_factor']:.2f})")


def test_accuracy_vs_brute_force():
    """Test KD-Tree accuracy against brute force search."""
    print("Testing Accuracy vs Brute Force...")
    
    np.random.seed(42)
    data = np.random.rand(50, 3) * 100
    patient_ids = list(range(50))
    
    tree = KDTree(data, patient_ids)
    
    # Random query point
    query = np.random.rand(3) * 100
    
    # KD-Tree result
    kd_neighbors = tree.k_nearest_neighbors(query, k=5)
    kd_ids = set(pid for pid, _ in kd_neighbors)
    
    # Brute force result
    distances = []
    for i, point in enumerate(data):
        dist = np.sqrt(np.sum((query - point) ** 2))
        distances.append((patient_ids[i], dist))
    
    distances.sort(key=lambda x: x[1])
    bf_ids = set(pid for pid, _ in distances[:5])
    
    # Results should match
    assert kd_ids == bf_ids, "KD-Tree results should match brute force"
    
    print("✓ Accuracy test passed (matches brute force)")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("Testing Edge Cases...")
    
    data = np.array([[1, 2], [3, 4]])
    patient_ids = [1, 2]
    
    tree = KDTree(data, patient_ids)
    
    # Test with k larger than dataset
    neighbors = tree.k_nearest_neighbors([2, 3], k=10)
    assert len(neighbors) == 2, "Should return all available points"
    
    # Test with wrong dimension query
    try:
        tree.k_nearest_neighbors([1, 2, 3], k=1)  # 3D query for 2D tree
        assert False, "Should raise ValueError for wrong dimensions"
    except ValueError:
        pass
    
    # Test get_patient_features
    features = tree.get_patient_features(1)
    assert features is not None, "Should return features for existing patient"
    assert np.array_equal(features, [1, 2]), "Should return correct features"
    
    features = tree.get_patient_features(999)
    assert features is None, "Should return None for non-existent patient"
    
    print("✓ Edge cases passed")


def test_performance():
    """Test performance characteristics."""
    print("Testing Performance...")
    
    import time
    
    # Create larger dataset
    np.random.seed(42)
    n = 1000
    data = np.random.rand(n, 5) * 100
    patient_ids = list(range(n))
    
    # Build tree
    start = time.time()
    tree = KDTree(data, patient_ids)
    build_time = time.time() - start
    
    # Test search performance
    query = np.random.rand(5) * 100
    
    start = time.time()
    for _ in range(100):
        tree.k_nearest_neighbors(query, k=10)
    search_time = (time.time() - start) / 100
    
    print(f"✓ Performance test passed")
    print(f"  Build tree ({n} points): {build_time*1000:.2f}ms")
    print(f"  KNN search (k=10): {search_time*1000:.2f}ms")
    print(f"  Tree depth: {tree.get_tree_depth()}")


def run_all_tests():
    """Run all KD-Tree tests."""
    print("\n" + "="*60)
    print("KD-TREE UNIT TESTS")
    print("="*60 + "\n")
    
    test_tree_construction()
    test_nearest_neighbor()
    test_k_nearest_neighbors()
    test_range_query()
    test_tree_balance()
    test_accuracy_vs_brute_force()
    test_edge_cases()
    test_performance()
    
    print("\n" + "="*60)
    print("✓ ALL KD-TREE TESTS PASSED")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
