"""
UCI Heart Disease Dataset - Algorithmic Analysis
=================================================

This script implements custom data structures and algorithms to analyze
the heart disease dataset. It focuses on identifying high-risk patients
using a Priority Queue and finding patient clusters using a Graph.

Components:
-----------
1. PatientNode: Custom class to represent patient data.
2. RiskPriorityQueue: Custom Min-Heap implementation for risk ranking.
3. SimilarityGraph: Custom Graph implementation for finding patient clusters.

"""

import csv
import heapq
import math
from pathlib import Path
from collections import deque

# -----------------------------------------------------------------------------
# 1. Custom Data Structure: Patient Node
# -----------------------------------------------------------------------------
class PatientNode:
    """
    Represents a patient with their medical attributes and calculated risk score.
    """
    def __init__(self, patient_id, attributes):
        self.id = patient_id
        self.attributes = attributes
        self.risk_score = self._calculate_risk_score()
        
    def _calculate_risk_score(self):
        """
        Calculate a heuristic heart disease risk score based on key factors.
        Higher score = Higher risk.
        """
        score = 0
        attr = self.attributes
        
        # Helper to safely get float/int values
        def get_val(key, default=0):
            try:
                return float(attr.get(key, default))
            except (ValueError, TypeError):
                return default

        # 1. Chest Pain (cp): Type 4 is asymptomatic but correlated with disease in this dataset
        cp = get_val('cp')
        if cp == 4: score += 20
        elif cp == 3: score += 15
        
        # 2. Thalassemia (thal): 7 (reversible defect) and 6 (fixed defect) are high risk
        thal = get_val('thal')
        if thal == 7: score += 25
        elif thal == 6: score += 15
        
        # 3. ST depression (oldpeak)
        score += get_val('oldpeak') * 5
        
        # 4. Number of vessels (ca)
        score += get_val('ca') * 10
        
        # 5. Max Heart Rate (thalach) - Lower max heart rate can be riskier
        thalach = get_val('thalach', 200)
        if thalach < 140: score += 10
        
        # 6. Age factor
        if get_val('age') > 60: score += 10
        
        return score

    def __lt__(self, other):
        # Inverted comparison for Max-Heap behavior using a Min-Heap
        # We want the HIGHEST risk at the top.
        return self.risk_score > other.risk_score

    def __repr__(self):
        return f"Patient(ID={self.id}, Risk={self.risk_score:.1f}, Age={self.attributes.get('age')})"

