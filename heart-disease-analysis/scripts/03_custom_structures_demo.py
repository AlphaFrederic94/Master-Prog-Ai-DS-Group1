"""
Custom Data Structures Demonstration
=====================================

Comprehensive demonstration of custom data structures applied to
the UCI Heart Disease dataset. Shows practical applications for:
- Feature selection using Feature Graph
- Patient similarity search using KD-Tree
- Categorical encoding using Hash Table

Author: Senior Data Scientist
Date: February 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add structures to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from structures import FeatureGraph, KDTree, CategoricalEncoder, HashTable


class HeartDiseaseStructuresDemo:
    """Demonstration of custom data structures on heart disease data."""
    
    def __init__(self, data_path: str):
        """
        Initialize demo with heart disease data.
        
        Parameters:
        -----------
        data_path : str
            Path to processed heart disease CSV file
        """
        self.data_path = Path(data_path)
        self.df = None
        self.feature_graph = None
        self.kd_tree = None
        self.encoders = {}
        
    def load_data(self):
        """Load processed heart disease data."""
        print("\n" + "="*70)
        print("LOADING HEART DISEASE DATA")
        print("="*70)
        
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df)} patient records")
        print(f"✓ Features: {list(self.df.columns)}")
        print(f"✓ Target distribution:")
        print(self.df['target'].value_counts())
        
    def demo_hash_table_encoding(self):
        """Demonstrate categorical encoding with custom hash table."""
        print("\n" + "="*70)
        print("DEMO 1: CATEGORICAL ENCODING WITH HASH TABLE")
        print("="*70)
        
        # Encode chest pain type (cp)
        print("\n1. Encoding Chest Pain Type (cp)")
        print("-" * 70)
        
        cp_encoder = CategoricalEncoder()
        unique_cp = self.df['cp'].unique()
        cp_encoder.fit(unique_cp)
        
        print(f"Unique values: {sorted(unique_cp)}")
        print(f"Encoding mapping: {cp_encoder.get_mapping()}")
        
        # Encode the column
        self.df['cp_encoded'] = cp_encoder.encode_batch(self.df['cp'].tolist())
        
        print(f"\nSample encodings:")
        sample = self.df[['cp', 'cp_encoded']].head(10)
        print(sample.to_string(index=False))
        
        # Store encoder
        self.encoders['cp'] = cp_encoder
        
        # Encode thalassemia (thal)
        print("\n2. Encoding Thalassemia (thal)")
        print("-" * 70)
        
        thal_encoder = CategoricalEncoder()
        unique_thal = self.df['thal'].unique()
        thal_encoder.fit(unique_thal)
        
        print(f"Unique values: {sorted(unique_thal)}")
        print(f"Encoding mapping: {thal_encoder.get_mapping()}")
        
        self.df['thal_encoded'] = thal_encoder.encode_batch(self.df['thal'].tolist())
        self.encoders['thal'] = thal_encoder
        
        # Show hash table statistics
        print("\n3. Hash Table Performance Statistics")
        print("-" * 70)
        stats = cp_encoder.get_collision_stats()
        print(f"Table size: {stats['size']}")
        print(f"Elements: {stats['num_elements']}")
        print(f"Load factor: {stats['load_factor']:.2f}")
        print(f"Collisions: {stats['num_collisions']}")
        print(f"Avg chain length: {stats['avg_chain_length']:.2f}")
        
    def demo_feature_graph(self):
        """Demonstrate feature analysis with Feature Graph."""
        print("\n" + "="*70)
        print("DEMO 2: FEATURE ANALYSIS WITH FEATURE GRAPH")
        print("="*70)
        
        # Select numeric features for correlation analysis
        numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 
                           'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        
        # Compute correlation matrix
        corr_matrix = self.df[numeric_features].corr()
        
        # Build feature graph
        print("\n1. Building Feature Graph")
        print("-" * 70)
        
        self.feature_graph = FeatureGraph(corr_matrix, threshold=0.3)
        stats = self.feature_graph.get_statistics()
        
        print(f"Nodes (features): {stats['num_nodes']}")
        print(f"Edges (correlations > 0.3): {stats['num_edges']}")
        print(f"Graph density: {stats['density']:.3f}")
        print(f"Average degree: {stats['avg_degree']:.2f}")
        print(f"Max degree: {stats['max_degree']}")
        
        # Detect multicollinearity
        print("\n2. Detecting Multicollinearity Clusters")
        print("-" * 70)
        
        clusters = self.feature_graph.detect_multicollinearity_clusters(threshold=0.7)
        
        if clusters:
            print(f"Found {len(clusters)} multicollinear cluster(s):")
            for i, cluster in enumerate(clusters, 1):
                print(f"\nCluster {i}: {cluster}")
                
                # Show correlations within cluster
                if len(cluster) > 1:
                    cluster_list = list(cluster)
                    print("Correlations:")
                    for j, feat1 in enumerate(cluster_list):
                        for feat2 in cluster_list[j+1:]:
                            corr = corr_matrix.loc[feat1, feat2]
                            print(f"  {feat1} ↔ {feat2}: {corr:.3f}")
        else:
            print("No strong multicollinearity detected (threshold: 0.7)")
        
        # Feature selection
        print("\n3. Feature Selection Using Centrality")
        print("-" * 70)
        
        central_features = self.feature_graph.get_central_features(method='weighted_degree')
        
        print("Features ranked by weighted degree centrality:")
        for i, feature in enumerate(central_features[:10], 1):
            degree = self.feature_graph.get_degree(feature)
            neighbors = self.feature_graph.get_neighbors(feature)
            total_weight = sum(w for _, w in neighbors)
            print(f"{i:2d}. {feature:12s} - degree: {degree}, total correlation: {total_weight:.3f}")
        
        # Maximum Spanning Tree
        print("\n4. Maximum Spanning Tree (MST)")
        print("-" * 70)
        
        mst = self.feature_graph.maximum_spanning_tree()
        mst_stats = mst.get_statistics()
        
        print(f"MST edges: {mst_stats['num_edges']} (reduced from {stats['num_edges']})")
        print(f"MST preserves connectivity while maximizing total correlation")
        
        # Recommend features to keep/remove
        if clusters:
            print("\n5. Feature Selection Recommendations")
            print("-" * 70)
            
            selected = self.feature_graph.select_features_from_clusters(clusters, method='degree')
            
            all_clustered = set()
            for cluster in clusters:
                all_clustered.update(cluster)
            
            to_remove = all_clustered - set(selected)
            
            print(f"Recommended features to KEEP from clusters: {selected}")
            print(f"Recommended features to REMOVE (redundant): {to_remove}")
    
    def demo_kd_tree(self):
        """Demonstrate patient similarity search with KD-Tree."""
        print("\n" + "="*70)
        print("DEMO 3: PATIENT SIMILARITY SEARCH WITH KD-TREE")
        print("="*70)
        
        # Select continuous features for similarity search
        continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        
        # Prepare data
        print("\n1. Building KD-Tree")
        print("-" * 70)
        
        data = self.df[continuous_features].values
        patient_ids = self.df.index.tolist()
        
        self.kd_tree = KDTree(data, patient_ids)
        stats = self.kd_tree.get_statistics()
        
        print(f"Points (patients): {stats['num_points']}")
        print(f"Dimensions (features): {stats['dimensions']}")
        print(f"Tree depth: {stats['depth']}")
        print(f"Optimal depth: {stats['optimal_depth']}")
        print(f"Balance factor: {stats['balance_factor']:.2f}")
        
        # Find similar patients
        print("\n2. Finding Similar Patients")
        print("-" * 70)
        
        # Select a random patient
        query_idx = 42
        query_patient = self.df.iloc[query_idx]
        query_features = data[query_idx]
        
        print(f"\nQuery Patient (ID: {query_idx}):")
        print(f"  Age: {query_patient['age']:.0f}")
        print(f"  Resting BP: {query_patient['trestbps']:.0f}")
        print(f"  Cholesterol: {query_patient['chol']:.0f}")
        print(f"  Max Heart Rate: {query_patient['thalach']:.0f}")
        print(f"  ST Depression: {query_patient['oldpeak']:.2f}")
        print(f"  Disease: {'Yes' if query_patient['target'] == 1 else 'No'}")
        
        # Find 5 most similar patients
        similar_patients = self.kd_tree.k_nearest_neighbors(query_features, k=6)  # 6 because query is included
        
        print(f"\n5 Most Similar Patients:")
        print("-" * 70)
        
        for i, (pid, distance) in enumerate(similar_patients[1:6], 1):  # Skip first (self)
            similar = self.df.iloc[pid]
            print(f"\n{i}. Patient {pid} (Distance: {distance:.2f})")
            print(f"   Age: {similar['age']:.0f}, BP: {similar['trestbps']:.0f}, "
                  f"Chol: {similar['chol']:.0f}, HR: {similar['thalach']:.0f}, "
                  f"ST: {similar['oldpeak']:.2f}")
            print(f"   Disease: {'Yes' if similar['target'] == 1 else 'No'}")
        
        # Range query
        print("\n3. Range Query (Finding Patients in Neighborhood)")
        print("-" * 70)
        
        radius = 20.0
        neighbors = self.kd_tree.range_query(query_features, radius=radius)
        
        print(f"Patients within radius {radius} of query patient: {len(neighbors)}")
        
        # Analyze outcomes of similar patients
        if len(neighbors) > 1:
            neighbor_outcomes = [self.df.iloc[pid]['target'] for pid in neighbors if pid != query_idx]
            disease_rate = sum(neighbor_outcomes) / len(neighbor_outcomes) * 100
            
            print(f"Disease rate among neighbors: {disease_rate:.1f}%")
            print(f"Query patient disease status: {'Positive' if query_patient['target'] == 1 else 'Negative'}")
    
    def generate_summary_report(self):
        """Generate summary report of all demonstrations."""
        print("\n" + "="*70)
        print("SUMMARY REPORT")
        print("="*70)
        
        print("\n✓ Hash Table Encoding:")
        print(f"  - Encoded {len(self.encoders)} categorical features")
        print(f"  - O(1) average lookup time")
        print(f"  - Efficient collision handling with separate chaining")
        
        if self.feature_graph:
            stats = self.feature_graph.get_statistics()
            print("\n✓ Feature Graph Analysis:")
            print(f"  - Analyzed {stats['num_nodes']} features")
            print(f"  - Identified {stats['num_edges']} significant correlations")
            print(f"  - Detected multicollinearity clusters")
            print(f"  - Recommended features for dimensionality reduction")
        
        if self.kd_tree:
            stats = self.kd_tree.get_statistics()
            print("\n✓ KD-Tree Patient Similarity:")
            print(f"  - Indexed {stats['num_points']} patients")
            print(f"  - O(log n) average search time")
            print(f"  - Enables case-based clinical reasoning")
            print(f"  - Tree balance factor: {stats['balance_factor']:.2f}")
        
        print("\n" + "="*70)
        print("DEMONSTRATION COMPLETE")
        print("="*70)
        print("\nThese custom data structures demonstrate:")
        print("  • Advanced algorithm implementation skills")
        print("  • Understanding of computational complexity")
        print("  • Practical application to real-world medical data")
        print("  • Software engineering best practices")
        print("="*70 + "\n")
    
    def run_all_demos(self):
        """Run all demonstrations."""
        self.load_data()
        self.demo_hash_table_encoding()
        self.demo_feature_graph()
        self.demo_kd_tree()
        self.generate_summary_report()


def main():
    """Main execution function."""
    # Path to processed data
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'heart_disease_cleveland.csv'
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please run 01_data_processing.py first.")
        return
    
    # Run demonstrations
    demo = HeartDiseaseStructuresDemo(data_path)
    demo.run_all_demos()


if __name__ == "__main__":
    main()
