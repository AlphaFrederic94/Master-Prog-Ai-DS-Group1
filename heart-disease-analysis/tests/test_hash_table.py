"""
Unit Tests for Hash Table
==========================

Comprehensive tests for custom hash table implementation including
basic operations, collision handling, resizing, and categorical encoding.

Author: Senior Data Scientist
Date: February 2026
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from structures.hash_table import HashTable, CategoricalEncoder


def test_hash_table_basic_operations():
    """Test basic insert, get, delete operations."""
    print("Testing Hash Table Basic Operations...")
    
    ht = HashTable(size=10)
    
    # Test insert and get
    ht.insert("age", 45)
    ht.insert("sex", 1)
    ht.insert("cp", 3)
    
    assert ht.get("age") == 45, "Failed to retrieve age"
    assert ht.get("sex") == 1, "Failed to retrieve sex"
    assert ht.get("cp") == 3, "Failed to retrieve cp"
    assert ht.get("nonexistent") is None, "Should return None for missing key"
    
    # Test update
    ht.insert("age", 50)
    assert ht.get("age") == 50, "Failed to update value"
    
    # Test contains
    assert ht.contains("age"), "Should contain 'age'"
    assert not ht.contains("missing"), "Should not contain 'missing'"
    
    # Test delete
    assert ht.delete("sex"), "Should successfully delete 'sex'"
    assert not ht.contains("sex"), "Should not contain deleted key"
    assert not ht.delete("nonexistent"), "Should return False for missing key"
    
    print("✓ Basic operations passed")


def test_hash_table_collisions():
    """Test collision handling with separate chaining."""
    print("Testing Hash Table Collision Handling...")
    
    ht = HashTable(size=5)  # Small size to force collisions
    
    # Insert multiple items
    for i in range(20):
        ht.insert(f"key{i}", i)
    
    # Verify all items can be retrieved
    for i in range(20):
        assert ht.get(f"key{i}") == i, f"Failed to retrieve key{i}"
    
    # Check collision stats
    stats = ht.get_collision_stats()
    assert stats['num_collisions'] > 0, "Should have collisions with small table"
    assert stats['max_chain_length'] > 1, "Should have chains longer than 1"
    
    print(f"✓ Collision handling passed (collisions: {stats['num_collisions']})")


def test_hash_table_resizing():
    """Test dynamic resizing."""
    print("Testing Hash Table Dynamic Resizing...")
    
    ht = HashTable(size=10, load_factor_threshold=0.75)
    
    initial_size = ht.size
    
    # Insert enough items to trigger resize
    for i in range(20):
        ht.insert(f"item{i}", i)
    
    assert ht.size > initial_size, "Table should have resized"
    
    # Verify all items still retrievable after resize
    for i in range(20):
        assert ht.get(f"item{i}") == i, f"Failed to retrieve item{i} after resize"
    
    print(f"✓ Dynamic resizing passed (size: {initial_size} → {ht.size})")


def test_categorical_encoder():
    """Test categorical encoder functionality."""
    print("Testing Categorical Encoder...")
    
    encoder = CategoricalEncoder()
    
    # Test fit
    categories = ['typical', 'atypical', 'non-anginal', 'asymptomatic']
    encoder.fit(categories)
    
    # Test encode
    assert encoder.encode('typical') == 0
    assert encoder.encode('atypical') == 1
    assert encoder.encode('non-anginal') == 2
    assert encoder.encode('asymptomatic') == 3
    
    # Test decode
    assert encoder.decode(0) == 'typical'
    assert encoder.decode(1) == 'atypical'
    assert encoder.decode(2) == 'non-anginal'
    assert encoder.decode(3) == 'asymptomatic'
    
    # Test batch operations
    batch = ['typical', 'atypical', 'typical', 'asymptomatic']
    encoded = encoder.encode_batch(batch)
    assert encoded == [0, 1, 0, 3], "Batch encoding failed"
    
    decoded = encoder.decode_batch(encoded)
    assert decoded == batch, "Batch decoding failed"
    
    # Test error handling
    try:
        encoder.encode('unknown')
        assert False, "Should raise ValueError for unknown category"
    except ValueError:
        pass
    
    print("✓ Categorical encoder passed")


def test_hash_table_performance():
    """Test performance characteristics."""
    print("Testing Hash Table Performance...")
    
    import time
    
    ht = HashTable(size=100)
    n = 1000
    
    # Test insert performance
    start = time.time()
    for i in range(n):
        ht.insert(f"key{i}", i)
    insert_time = time.time() - start
    
    # Test get performance
    start = time.time()
    for i in range(n):
        _ = ht.get(f"key{i}")
    get_time = time.time() - start
    
    stats = ht.get_collision_stats()
    
    print(f"✓ Performance test passed")
    print(f"  Insert {n} items: {insert_time*1000:.2f}ms")
    print(f"  Get {n} items: {get_time*1000:.2f}ms")
    print(f"  Load factor: {stats['load_factor']:.2f}")
    print(f"  Avg chain length: {stats['avg_chain_length']:.2f}")


def run_all_tests():
    """Run all hash table tests."""
    print("\n" + "="*60)
    print("HASH TABLE UNIT TESTS")
    print("="*60 + "\n")
    
    test_hash_table_basic_operations()
    test_hash_table_collisions()
    test_hash_table_resizing()
    test_categorical_encoder()
    test_hash_table_performance()
    
    print("\n" + "="*60)
    print("✓ ALL HASH TABLE TESTS PASSED")
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()
