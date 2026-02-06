"""
Custom Data Structures for Heart Disease Analysis
==================================================

This module provides custom implementations of fundamental data structures
optimized for medical data analysis and feature engineering.

Components:
-----------
- FeatureGraph: Correlation-based graph for feature analysis
- KDTree: K-dimensional tree for patient similarity search
- HashTable: Custom hash table for categorical encoding

Author: Senior Data Scientist
Date: February 2026
"""

__version__ = "1.0.0"
__author__ = "Senior Data Scientist"

from .hash_table import HashTable, CategoricalEncoder
from .feature_graph import FeatureGraph
from .kd_tree import KDTree, KDNode

__all__ = [
    'HashTable',
    'CategoricalEncoder',
    'FeatureGraph',
    'KDTree',
    'KDNode'
]
