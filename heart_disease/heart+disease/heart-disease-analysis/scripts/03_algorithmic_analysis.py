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


# -----------------------------------------------------------------------------
# 2. Custom Data Structure: Risk Priority Queue (Min-Heap)
# -----------------------------------------------------------------------------
class RiskPriorityQueue:
    """
    Priority Queue to manage patients by risk score.
    Uses Python's heapq (Min-Heap) internally but simulates Max-Heap behavior
    via the PatientNode's __lt__ method.
    """
    def __init__(self):
        self.heap = []
        self.size = 0
        
    def push(self, patient):
        """Add a patient to the queue."""
        heapq.heappush(self.heap, patient)
        self.size += 1
        
    def pop(self):
        """Remove and return the highest risk patient."""
        if self.is_empty():
            return None
        self.size -= 1
        return heapq.heappop(self.heap)
        
    def peek(self):
        """View the highest risk patient without removing."""
        if self.is_empty():
            return None
        return self.heap[0]
    
    def is_empty(self):
        return self.size == 0
    
    def get_all_sorted(self):
        """Return all patients sorted by risk (destructive)."""
        sorted_patients = []
        while not self.is_empty():
            sorted_patients.append(self.pop())
        return sorted_patients


# -----------------------------------------------------------------------------
# 3. Custom Data Structure: Similarity Graph
# -----------------------------------------------------------------------------
class SimilarityGraph:
    """
    Graph representing patient similarities. Nodes are patient IDs.
    Edges exist between patients if their similarity exceeds a threshold.
    """
    def __init__(self):
        self.adj_list = {} # dictionary mapping patient_id -> list of neighbors
        self.nodes = {}    # map id -> PatientNode
        
    def add_node(self, patient):
        self.nodes[patient.id] = patient
        if patient.id not in self.adj_list:
            self.adj_list[patient.id] = []
            
    def add_edge(self, id1, id2):
        if id1 in self.adj_list and id2 in self.adj_list:
            # Undirected graph
            if id2 not in self.adj_list[id1]: self.adj_list[id1].append(id2)
            if id1 not in self.adj_list[id2]: self.adj_list[id2].append(id1)

    def calculate_similarity(self, p1, p2):
        """
        Calculate similarity between two patients using Euclidean distance 
        (inverse) on normalized key features.
        """
        features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        
        sum_sq_diff = 0
        for feat in features:
            try:
                v1 = float(p1.attributes.get(feat, 0))
                v2 = float(p2.attributes.get(feat, 0))
            except ValueError:
                v1, v2 = 0.0, 0.0
                
            # Simple normalization implicitly by range would be better but keeping simple
            sum_sq_diff += (v1 - v2) ** 2
            
        distance = math.sqrt(sum_sq_diff)
        # Similarity is inverse of distance. Adding epsilon to avoid div/0
        return 1 / (1 + distance)

    def build_graph(self, threshold=0.015):
        """
        Connect nodes that are similar enough.
        Note: O(N^2) complexity - acceptable for small datasets (~300-900 nodes).
        """
        ids = list(self.nodes.keys())
        n = len(ids)
        print(f"Building graph connections for {n} nodes...")
        
        edge_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                p1 = self.nodes[ids[i]]
                p2 = self.nodes[ids[j]]
                
                sim = self.calculate_similarity(p1, p2)
                if sim > threshold:
                    self.add_edge(ids[i], ids[j])
                    edge_count += 1
        
        print(f"Graph built with {edge_count} edges.")

    def find_connected_components(self):
        """
        Use BFS to find clusters of similar patients.
        Returns a list of sets, where each set is a cluster of patient IDs.
        """
        visited = set()
        components = []
        
        for pid in self.adj_list:
            if pid not in visited:
                cluster = set()
                queue = deque([pid])
                visited.add(pid)
                cluster.add(pid)
                
                while queue:
                    curr = queue.popleft()
                    for neighbor in self.adj_list[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            cluster.add(neighbor)
                            queue.append(neighbor)
                
                components.append(cluster)
        
        return components

# -----------------------------------------------------------------------------
# Main Execution Pipeline
# -----------------------------------------------------------------------------
class AlgorithmicAnalyzer:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.data_path = self.base_dir / 'data' / 'processed' / 'heart_disease_combined.csv'
        self.reports_dir = self.base_dir / 'reports'
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
    def load_patients(self):
        """Load CSV and convert to PatientNode objects."""
        if not self.data_path.exists():
            print(f"Error: Data file {self.data_path} not found.")
            return []
            
        patients = []
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader):
                    # Use row index as simple ID
                    patients.append(PatientNode(idx, row))
                    
            print(f"✓ Loaded {len(patients)} patients from {self.data_path.name}")
        except Exception as e:
            print(f"Error loading data: {e}")
            return []
            
        return patients

    def run_analysis(self):
        print("\n" + "⚙️ " + "="*58)
        print("ALGORITHMIC COMPONENTS ANALYSIS")
        print("="*60 + "\n")
        
        patients = self.load_patients()
        if not patients: return
        
        # 1. Priority Queue Analysis (High Risk Identification)
        print("1. RISK ANALYSIS (Priority Queue)")
        print("-" * 40)
        pq = RiskPriorityQueue()
        for p in patients:
            pq.push(p)
            
        top_n = 10
        print(f"Top {top_n} Highest Risk Patients:")
        
        high_risk_patients = []
        for i in range(top_n):
            p = pq.pop()
            if p:
                high_risk_patients.append(p)
                print(f"  {i+1}. ID: {p.id:03d} | Risk Score: {p.risk_score:.1f} | "
                      f"Age: {p.attributes.get('age')} | CP: {p.attributes.get('cp')} | "
                      f"Thal: {p.attributes.get('thal', '?')}")
        
        # 2. Graph Analysis (Cluster Detection)
        print("\n2. SIMILARITY CLUSTERING (Graph Analysis)")
        print("-" * 40)
        graph = SimilarityGraph()
        for p in patients:
            graph.add_node(p)
            
        # Using a strict threshold to find meaningful small clusters
        # Note: Threshold tuning depends on data scaling
        graph.build_graph(threshold=0.04) 
        
        clusters = graph.find_connected_components()
        print(f"Found {len(clusters)} connected components (clusters).")
        
        # Filter for non-trivial clusters
        significant_clusters = [c for c in clusters if len(c) > 1]
        significant_clusters.sort(key=len, reverse=True)
        
        print("\nLargest Clusters Detected:")
        for i, cluster in enumerate(significant_clusters[:5]):
            print(f"  Cluster {i+1}: {len(cluster)} patients")
            # Analyze common trait in cluster - simple approach showing mean age
            p_ids = list(cluster)
            ages = []
            for pid in p_ids:
                try:
                    ages.append(float(graph.nodes[pid].attributes['age']))
                except ValueError:
                    pass
            mean_age = sum(ages) / len(ages) if ages else 0
            print(f"    Average Age: {mean_age:.1f}")

        # 3. Save Report
        self.save_report(high_risk_patients, significant_clusters, graph)
        print("\n" + "="*60)
        print("✓ ANALYSIS COMPLETE!")
        print("="*60 + "\n")

    def save_report(self, high_risk, clusters, graph):
        outfile = self.reports_dir / 'algorithmic_analysis_results.txt'
        
        with open(outfile, 'w') as f:
            f.write("ALGORITHMIC ANALYSIS REPORT\n")
            f.write("===========================\n\n")
            
            f.write("1. HIGH RISK PATIENTS (Priority Queue)\n")
            f.write("------------------------------------\n")
            f.write("Patients identified as high priority for triage based on risk factors:\n\n")
            for p in high_risk:
                f.write(f"ID {p.id:03d}: Risk Score {p.risk_score:.1f}\n")
                f.write(f"   Details: Age {p.attributes.get('age')}, CP {p.attributes.get('cp')}, "
                        f"Oldpeak {p.attributes.get('oldpeak')}, Thal {p.attributes.get('thal')}\n\n")
            
            f.write("\n2. PATIENT SIMILARITY CLUSTERS (Graph Theory)\n")
            f.write("-------------------------------------------\n")
            f.write(f"Total Clusters Found: {len(clusters) + (len(graph.nodes) - sum(len(c) for c in clusters))}\n") # adjust logic if needed
            f.write("Significant clusters (size > 1):\n\n")
            
            for i, cluster in enumerate(clusters):
                if i >= 10: break # limit report
                p_ids = list(cluster)
                f.write(f"Cluster {i+1} (Size: {len(cluster)}): {p_ids[:10]}...\n")

        print(f"Report saved to: {outfile}")

def main():
    base_dir = Path(__file__).parent.parent
    analyzer = AlgorithmicAnalyzer(base_dir)
    analyzer.run_analysis()

if __name__ == "__main__":
    main()
