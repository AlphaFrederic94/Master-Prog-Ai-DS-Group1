"""
Feature Engineering Module
===========================

Prepares features for ML models using custom data structures:
- CategoricalEncoder (Hash Table) for categorical features
- FeatureGraph for multicollinearity detection and feature selection
- StandardScaler for normalization

Author: NGANA NOAJ Junior Data Scientist
Date: February 2026
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from structures import CategoricalEncoder, FeatureGraph


class FeatureEngineer:
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.df = None
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_graph = None
        self.selected_features = None
        
    def load_data(self):
        print("="*70)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*70)
        print("\n1. Loading data...")
        self.df = pd.read_csv(self.data_path)
        print(f"   Loaded {len(self.df)} patient records")
        print(f"   Features: {list(self.df.columns)}")
        
    def encode_categorical_features(self):
        print("\n2. Encoding categorical features with CategoricalEncoder...")
        
        categorical_features = {
            'cp': 'Chest Pain Type',
            'restecg': 'Resting ECG',
            'slope': 'ST Slope',
            'thal': 'Thalassemia'
        }
        
        for feature, description in categorical_features.items():
            encoder = CategoricalEncoder()
            unique_values = sorted(self.df[feature].unique())
            encoder.fit([str(v) for v in unique_values])
            
            self.df[f'{feature}_encoded'] = encoder.encode_batch(
                [str(v) for v in self.df[feature]]
            )
            
            self.encoders[feature] = encoder
            
            print(f"   {feature} ({description}): {len(unique_values)} categories")
            print(f"      Mapping: {encoder.get_mapping()}")
        
    def analyze_feature_correlations(self):
        print("\n3. Analyzing feature correlations with FeatureGraph...")
        
        numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak',
                           'cp_encoded', 'restecg_encoded', 'slope_encoded', 
                           'thal_encoded', 'sex', 'fbs', 'exang', 'ca']
        
        corr_matrix = self.df[numeric_features].corr()
        
        self.feature_graph = FeatureGraph(corr_matrix, threshold=0.3)
        stats = self.feature_graph.get_statistics()
        
        print(f"   Graph nodes (features): {stats['num_nodes']}")
        print(f"   Graph edges (correlations > 0.3): {stats['num_edges']}")
        print(f"   Graph density: {stats['density']:.3f}")
        
        clusters = self.feature_graph.detect_multicollinearity_clusters(threshold=0.7)
        
        if clusters:
            print(f"\n   Multicollinearity detected: {len(clusters)} cluster(s)")
            for i, cluster in enumerate(clusters, 1):
                print(f"      Cluster {i}: {cluster}")
        else:
            print("   No strong multicollinearity detected")
        
        central_features = self.feature_graph.get_central_features(method='weighted_degree')
        print(f"\n   Top 5 most central features:")
        for i, feature in enumerate(central_features[:5], 1):
            degree = self.feature_graph.get_degree(feature)
            print(f"      {i}. {feature} (degree: {degree})")
        
        return numeric_features
    
    def select_final_features(self, all_features):
        print("\n4. Selecting final feature set...")
        
        continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        categorical_encoded = ['cp_encoded', 'restecg_encoded', 'slope_encoded', 'thal_encoded']
        binary_features = ['sex', 'fbs', 'exang']
        numeric_categorical = ['ca']
        
        self.selected_features = (continuous_features + categorical_encoded + 
                                 binary_features + numeric_categorical)
        
        print(f"   Selected {len(self.selected_features)} features:")
        print(f"      Continuous: {continuous_features}")
        print(f"      Categorical (encoded): {categorical_encoded}")
        print(f"      Binary: {binary_features}")
        print(f"      Numeric categorical: {numeric_categorical}")
        
        return self.selected_features
    
    def create_train_test_split(self, features):
        print("\n5. Creating train/test split...")
        
        X = self.df[features].values
        y = self.df['target'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        print(f"   Class distribution (train): {np.bincount(y_train)}")
        print(f"   Class distribution (test): {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def normalize_features(self, X_train, X_test):
        print("\n6. Normalizing features with StandardScaler...")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"   Features normalized (mean=0, std=1)")
        print(f"   Training set shape: {X_train_scaled.shape}")
        print(f"   Test set shape: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled
    
    def save_engineered_data(self, X_train, X_test, y_train, y_test, features):
        print("\n7. Saving engineered datasets...")
        
        processed_dir = Path(__file__).parent.parent / 'data' / 'processed'
        processed_dir.mkdir(exist_ok=True, parents=True)
        
        train_df = pd.DataFrame(X_train, columns=features)
        train_df['target'] = y_train
        train_df.to_csv(processed_dir / 'train_engineered.csv', index=False)
        
        test_df = pd.DataFrame(X_test, columns=features)
        test_df['target'] = y_test
        test_df.to_csv(processed_dir / 'test_engineered.csv', index=False)
        
        print(f"   Saved: train_engineered.csv ({len(train_df)} rows)")
        print(f"   Saved: test_engineered.csv ({len(test_df)} rows)")
        
    def save_artifacts(self):
        print("\n8. Saving feature engineering artifacts...")
        
        models_dir = Path(__file__).parent.parent / 'models'
        models_dir.mkdir(exist_ok=True, parents=True)
        
        with open(models_dir / 'encoders.pkl', 'wb') as f:
            pickle.dump(self.encoders, f)
        print("   Saved: encoders.pkl")
        
        with open(models_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        print("   Saved: scaler.pkl")
        
        with open(models_dir / 'feature_names.pkl', 'wb') as f:
            pickle.dump(self.selected_features, f)
        print("   Saved: feature_names.pkl")
        
    def generate_report(self):
        print("\n9. Generating feature engineering report...")
        
        reports_dir = Path(__file__).parent.parent / 'reports'
        reports_dir.mkdir(exist_ok=True, parents=True)
        
        report_path = reports_dir / 'feature_engineering_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FEATURE ENGINEERING REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("1. CATEGORICAL ENCODING (Hash Table)\n")
            f.write("-"*70 + "\n")
            for feature, encoder in self.encoders.items():
                f.write(f"\n{feature}:\n")
                mapping = encoder.get_mapping()
                for category, code in sorted(mapping.items(), key=lambda x: x[1]):
                    f.write(f"  {category} -> {code}\n")
            
            f.write("\n\n2. FEATURE GRAPH ANALYSIS\n")
            f.write("-"*70 + "\n")
            stats = self.feature_graph.get_statistics()
            f.write(f"Nodes: {stats['num_nodes']}\n")
            f.write(f"Edges: {stats['num_edges']}\n")
            f.write(f"Density: {stats['density']:.3f}\n")
            f.write(f"Average degree: {stats['avg_degree']:.2f}\n")
            
            f.write("\n\n3. SELECTED FEATURES\n")
            f.write("-"*70 + "\n")
            for i, feature in enumerate(self.selected_features, 1):
                f.write(f"{i:2d}. {feature}\n")
            
            f.write(f"\n\nTotal features: {len(self.selected_features)}\n")
        
        print(f"   Report saved: {report_path}")
    
    def run_pipeline(self):
        self.load_data()
        self.encode_categorical_features()
        all_features = self.analyze_feature_correlations()
        features = self.select_final_features(all_features)
        X_train, X_test, y_train, y_test = self.create_train_test_split(features)
        X_train_scaled, X_test_scaled = self.normalize_features(X_train, X_test)
        self.save_engineered_data(X_train_scaled, X_test_scaled, y_train, y_test, features)
        self.save_artifacts()
        self.generate_report()
        
        print("\n" + "="*70)
        print("FEATURE ENGINEERING COMPLETE")
        print("="*70)
        print("\nReady for model training!")


def main():
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'heart_disease_cleveland.csv'
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return
    
    engineer = FeatureEngineer(data_path)
    engineer.run_pipeline()


if __name__ == "__main__":
    main()
