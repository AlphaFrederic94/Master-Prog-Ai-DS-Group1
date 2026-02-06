"""
Custom Hash Table Implementation
=================================

A from-scratch implementation of a hash table using separate chaining
for collision resolution. Includes a specialized CategoricalEncoder
for efficient categorical variable encoding/decoding.

Time Complexity:
---------------
- Insert: O(1) average, O(n) worst case
- Get: O(1) average, O(n) worst case
- Delete: O(1) average, O(n) worst case
- Resize: O(n)

Author: Senior Data Scientist
Date: February 2026
"""

from typing import Any, Optional, List, Dict, Tuple
import hashlib


class _HashNode:
    """
    Node for separate chaining in hash table.
    
    Attributes:
    -----------
    key : str
        The key for this entry
    value : Any
        The value associated with the key
    next : Optional[_HashNode]
        Pointer to next node in chain (for collision resolution)
    """
    
    def __init__(self, key: str, value: Any):
        self.key = key
        self.value = value
        self.next: Optional[_HashNode] = None


class HashTable:
    """
    Custom hash table implementation with separate chaining.
    
    Features:
    ---------
    - Polynomial rolling hash function
    - Separate chaining for collision resolution
    - Dynamic resizing when load factor exceeds threshold
    - Collision statistics tracking
    
    Parameters:
    -----------
    size : int, default=100
        Initial size of the hash table
    load_factor_threshold : float, default=0.75
        Threshold for triggering resize operation
        
    Examples:
    ---------
    >>> ht = HashTable(size=10)
    >>> ht.insert("age", 45)
    >>> ht.get("age")
    45
    >>> ht.contains("age")
    True
    """
    
    def __init__(self, size: int = 100, load_factor_threshold: float = 0.75):
        """Initialize hash table with given size and load factor threshold."""
        self.size = size
        self.load_factor_threshold = load_factor_threshold
        self.table: List[Optional[_HashNode]] = [None] * size
        self.num_elements = 0
        self.num_collisions = 0
        
    def _hash(self, key: str) -> int:
        """
        Compute hash value for a key using polynomial rolling hash.
        
        Uses a prime base (31) and modulo operation for distribution.
        
        Parameters:
        -----------
        key : str
            The key to hash
            
        Returns:
        --------
        int
            Hash value in range [0, size)
        """
        hash_value = 0
        prime = 31
        
        for i, char in enumerate(key):
            hash_value += ord(char) * (prime ** i)
        
        return hash_value % self.size
    
    def insert(self, key: str, value: Any) -> None:
        """
        Insert a key-value pair into the hash table.
        
        If key already exists, updates the value.
        Triggers resize if load factor exceeds threshold.
        
        Parameters:
        -----------
        key : str
            The key to insert
        value : Any
            The value to associate with the key
        """
        # Check if resize needed
        if self.get_load_factor() > self.load_factor_threshold:
            self._resize(self.size * 2)
        
        index = self._hash(key)
        
        # If bucket is empty, create new node
        if self.table[index] is None:
            self.table[index] = _HashNode(key, value)
            self.num_elements += 1
        else:
            # Traverse chain to find key or end
            current = self.table[index]
            prev = None
            
            while current is not None:
                if current.key == key:
                    # Key exists, update value
                    current.value = value
                    return
                prev = current
                current = current.next
            
            # Key not found, add to end of chain
            prev.next = _HashNode(key, value)
            self.num_elements += 1
            self.num_collisions += 1
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve value associated with key.
        
        Parameters:
        -----------
        key : str
            The key to look up
            
        Returns:
        --------
        Optional[Any]
            The value if key exists, None otherwise
        """
        index = self._hash(key)
        current = self.table[index]
        
        while current is not None:
            if current.key == key:
                return current.value
            current = current.next
        
        return None
    
    def delete(self, key: str) -> bool:
        """
        Delete a key-value pair from the hash table.
        
        Parameters:
        -----------
        key : str
            The key to delete
            
        Returns:
        --------
        bool
            True if key was found and deleted, False otherwise
        """
        index = self._hash(key)
        current = self.table[index]
        prev = None
        
        while current is not None:
            if current.key == key:
                if prev is None:
                    # Deleting head of chain
                    self.table[index] = current.next
                else:
                    # Deleting from middle/end of chain
                    prev.next = current.next
                
                self.num_elements -= 1
                return True
            
            prev = current
            current = current.next
        
        return False
    
    def contains(self, key: str) -> bool:
        """
        Check if key exists in hash table.
        
        Parameters:
        -----------
        key : str
            The key to check
            
        Returns:
        --------
        bool
            True if key exists, False otherwise
        """
        return self.get(key) is not None
    
    def get_load_factor(self) -> float:
        """
        Calculate current load factor.
        
        Returns:
        --------
        float
            Load factor (num_elements / size)
        """
        return self.num_elements / self.size
    
    def _resize(self, new_size: int) -> None:
        """
        Resize the hash table and rehash all elements.
        
        Parameters:
        -----------
        new_size : int
            New size for the hash table
        """
        old_table = self.table
        self.size = new_size
        self.table = [None] * new_size
        self.num_elements = 0
        self.num_collisions = 0
        
        # Rehash all elements
        for bucket in old_table:
            current = bucket
            while current is not None:
                self.insert(current.key, current.value)
                current = current.next
    
    def get_collision_stats(self) -> Dict[str, Any]:
        """
        Get statistics about hash table performance.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing:
            - num_elements: Total number of elements
            - size: Current table size
            - load_factor: Current load factor
            - num_collisions: Number of collisions
            - avg_chain_length: Average chain length
            - max_chain_length: Maximum chain length
        """
        chain_lengths = []
        
        for bucket in self.table:
            length = 0
            current = bucket
            while current is not None:
                length += 1
                current = current.next
            if length > 0:
                chain_lengths.append(length)
        
        return {
            'num_elements': self.num_elements,
            'size': self.size,
            'load_factor': self.get_load_factor(),
            'num_collisions': self.num_collisions,
            'avg_chain_length': sum(chain_lengths) / len(chain_lengths) if chain_lengths else 0,
            'max_chain_length': max(chain_lengths) if chain_lengths else 0
        }
    
    def __len__(self) -> int:
        """Return number of elements in hash table."""
        return self.num_elements
    
    def __str__(self) -> str:
        """String representation of hash table."""
        stats = self.get_collision_stats()
        return f"HashTable(size={self.size}, elements={self.num_elements}, load_factor={stats['load_factor']:.2f})"


class CategoricalEncoder(HashTable):
    """
    Specialized hash table for categorical variable encoding.
    
    Provides bidirectional mapping between categorical values and integer codes.
    Useful for encoding categorical features for machine learning models.
    
    Examples:
    ---------
    >>> encoder = CategoricalEncoder()
    >>> encoder.fit(['typical', 'atypical', 'non-anginal', 'asymptomatic'])
    >>> encoder.encode('typical')
    0
    >>> encoder.decode(0)
    'typical'
    >>> encoder.encode_batch(['typical', 'atypical', 'typical'])
    [0, 1, 0]
    """
    
    def __init__(self):
        """Initialize categorical encoder."""
        super().__init__(size=50)
        self.reverse_map: Dict[int, str] = {}
        self.next_code = 0
    
    def fit(self, categories: List[str]) -> None:
        """
        Fit encoder to a list of categories.
        
        Creates bidirectional mapping between categories and integer codes.
        
        Parameters:
        -----------
        categories : List[str]
            List of unique category values
        """
        for category in categories:
            if not self.contains(str(category)):
                code = self.next_code
                self.insert(str(category), code)
                self.reverse_map[code] = str(category)
                self.next_code += 1
    
    def encode(self, category: str) -> int:
        """
        Encode a single category to integer code.
        
        Parameters:
        -----------
        category : str
            Category value to encode
            
        Returns:
        --------
        int
            Integer code for the category
            
        Raises:
        -------
        ValueError
            If category not found in fitted categories
        """
        code = self.get(str(category))
        if code is None:
            raise ValueError(f"Category '{category}' not found. Call fit() first.")
        return code
    
    def decode(self, code: int) -> str:
        """
        Decode an integer code back to category value.
        
        Parameters:
        -----------
        code : int
            Integer code to decode
            
        Returns:
        --------
        str
            Original category value
            
        Raises:
        -------
        ValueError
            If code not found in mapping
        """
        if code not in self.reverse_map:
            raise ValueError(f"Code {code} not found in mapping.")
        return self.reverse_map[code]
    
    def encode_batch(self, categories: List[str]) -> List[int]:
        """
        Encode a batch of categories.
        
        Parameters:
        -----------
        categories : List[str]
            List of category values to encode
            
        Returns:
        --------
        List[int]
            List of integer codes
        """
        return [self.encode(str(cat)) for cat in categories]
    
    def decode_batch(self, codes: List[int]) -> List[str]:
        """
        Decode a batch of integer codes.
        
        Parameters:
        -----------
        codes : List[int]
            List of integer codes to decode
            
        Returns:
        --------
        List[str]
            List of original category values
        """
        return [self.decode(code) for code in codes]
    
    def get_mapping(self) -> Dict[str, int]:
        """
        Get the complete category-to-code mapping.
        
        Returns:
        --------
        Dict[str, int]
            Dictionary mapping categories to codes
        """
        mapping = {}
        for bucket in self.table:
            current = bucket
            while current is not None:
                mapping[current.key] = current.value
                current = current.next
        return mapping
    
    def __repr__(self) -> str:
        """Representation of encoder."""
        return f"CategoricalEncoder(num_categories={len(self)})"
