"""
Decision Tree Model - Heart Disease Prediction
==============================================

Interpretable model using Decision Trees with feature importance
analysis and validation against FeatureGraph centrality rankings.

Author: NGANA NOAJ Junior Data Scientist
Date: February 2026
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))


class DecisionTreePipeline:
    
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.feature_names = None
        self.best_params = None
        
    def load_engineered_data(self):
        print("="*70)
        print("DECISION TREE MODEL TRAINING")
        print("="*70)
        print("\nLoading pre-engineered data...")
        
        processed_dir = Path(__file__).parent.parent / 'data' / 'processed'
        models_dir = Path(__file__).parent.parent / 'models'
        
        train_df = pd.read_csv(processed_dir / 'train_engineered.csv')
        test_df = pd.read_csv(processed_dir / 'test_engineered.csv')
        
        self.y_train = train_df['target'].values
        self.y_test = test_df['target'].values
        
        self.X_train = train_df.drop('target', axis=1).values
        self.X_test = test_df.drop('target', axis=1).values
        
        with open(models_dir / 'feature_names.pkl', 'rb') as f:
            self.feature_names = pickle.load(f)
        
        print(f"  Training set: {len(self.X_train)} samples")
        print(f"  Test set: {len(self.X_test)} samples")
        print(f"  Features: {len(self.feature_names)}")
        
    def tune_hyperparameters(self):
        print("\nTuning hyperparameters with GridSearchCV...")
        
        param_grid = {
            'max_depth': [3, 5, 7, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
        
        dt = DecisionTreeClassifier(random_state=42)
        
        grid_search = GridSearchCV(
            dt, param_grid, cv=5, scoring='roc_auc', 
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        
        print(f"  Best parameters: {self.best_params}")
        print(f"  Best CV ROC-AUC: {grid_search.best_score_:.4f}")
        
    def evaluate_model(self):
        print("\n" + "="*70)
        print("MODEL EVALUATION")
        print("="*70)
        
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        
        print(f"\nTest Set Performance:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, 
                                    cv=5, scoring='roc_auc')
        print(f"\n5-Fold Cross-Validation ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, 
                                   target_names=['No Disease', 'Disease']))
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'cv_scores': cv_scores,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def analyze_feature_importance(self):
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*70)
        
        importances = self.model.feature_importances_
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))
        
        print(f"\nTree Depth: {self.model.get_depth()}")
        print(f"Number of Leaves: {self.model.get_n_leaves()}")
        
        return importance_df
    
    def generate_visualizations(self, results, importance_df):
        print("\nGenerating visualizations...")
        
        viz_dir = Path(__file__).parent.parent / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        ax1 = fig.add_subplot(gs[0, 0])
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax1,
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        ax1.set_title('Confusion Matrix', fontweight='bold')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        
        ax2 = fig.add_subplot(gs[0, 1])
        top_features = importance_df.head(10)
        colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(top_features)))
        ax2.barh(range(len(top_features)), top_features['Importance'], color=colors)
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features['Feature'])
        ax2.set_xlabel('Importance')
        ax2.set_title('Top 10 Feature Importances', fontweight='bold')
        ax2.invert_yaxis()
        
        ax3 = fig.add_subplot(gs[1, 0])
        fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
        ax3.plot(fpr, tpr, color='green', lw=2, 
                label=f'ROC Curve (AUC = {results["roc_auc"]:.3f})')
        ax3.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curve', fontweight='bold')
        ax3.legend(loc='lower right')
        ax3.grid(alpha=0.3)
        
        ax4 = fig.add_subplot(gs[1, 1])
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        values = [results['accuracy'], results['precision'], results['recall'], 
                 results['f1'], results['roc_auc']]
        colors_bar = ['#27ae60' if v >= 0.8 else '#f39c12' if v >= 0.7 else '#e74c3c' 
                     for v in values]
        ax4.bar(metrics, values, color=colors_bar, alpha=0.7, edgecolor='black')
        ax4.set_ylim(0, 1)
        ax4.set_ylabel('Score')
        ax4.set_title('Performance Metrics', fontweight='bold')
        ax4.axhline(y=0.8, color='green', linestyle='--', linewidth=1, alpha=0.5)
        for i, v in enumerate(values):
            ax4.text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        ax5 = fig.add_subplot(gs[2, :])
        plot_tree(self.model, filled=True, feature_names=self.feature_names,
                 class_names=['No Disease', 'Disease'], ax=ax5, 
                 max_depth=3, fontsize=8, rounded=True)
        ax5.set_title('Decision Tree Structure (Max Depth 3 for Visualization)', 
                     fontweight='bold', pad=10)
        
        fig.suptitle('Decision Tree - Model Analysis', fontsize=16, fontweight='bold', y=0.995)
        
        output_path = viz_dir / 'decision_tree_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()
        
        fig, ax = plt.subplots(figsize=(20, 12))
        plot_tree(self.model, filled=True, feature_names=self.feature_names,
                 class_names=['No Disease', 'Disease'], ax=ax, 
                 fontsize=10, rounded=True)
        ax.set_title('Complete Decision Tree Structure', fontsize=16, fontweight='bold')
        
        full_tree_path = viz_dir / 'decision_tree_full_structure.png'
        plt.savefig(full_tree_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {full_tree_path}")
        plt.close()
        
    def save_model(self):
        print("\nSaving model...")
        
        models_dir = Path(__file__).parent.parent / 'models'
        models_dir.mkdir(exist_ok=True)
        
        with open(models_dir / 'decision_tree.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        print("  Model saved successfully")
        
    def save_report(self, results, importance_df):
        print("\nGenerating text report...")
        
        reports_dir = Path(__file__).parent.parent / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        report_path = reports_dir / 'decision_tree_results.txt'
        
        total_samples = len(self.X_train) + len(self.X_test)
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DECISION TREE MODEL - EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("Dataset Information:\n")
            f.write(f"  Total samples: {total_samples}\n")
            f.write(f"  Training samples: {len(self.X_train)}\n")
            f.write(f"  Test samples: {len(self.X_test)}\n")
            f.write(f"  Features: {len(self.feature_names)}\n\n")
            
            f.write("Model Configuration:\n")
            f.write(f"  Best parameters: {self.best_params}\n")
            f.write(f"  Tree depth: {self.model.get_depth()}\n")
            f.write(f"  Number of leaves: {self.model.get_n_leaves()}\n\n")
            
            f.write("Performance Metrics:\n")
            f.write(f"  Accuracy:  {results['accuracy']:.4f}\n")
            f.write(f"  Precision: {results['precision']:.4f}\n")
            f.write(f"  Recall:    {results['recall']:.4f}\n")
            f.write(f"  F1-Score:  {results['f1']:.4f}\n")
            f.write(f"  ROC-AUC:   {results['roc_auc']:.4f}\n\n")
            
            f.write("Cross-Validation:\n")
            f.write(f"  5-Fold CV ROC-AUC: {results['cv_scores'].mean():.4f} ")
            f.write(f"(+/- {results['cv_scores'].std():.4f})\n\n")
            
            f.write("Confusion Matrix:\n")
            f.write(f"  True Negatives:  {results['confusion_matrix'][0, 0]}\n")
            f.write(f"  False Positives: {results['confusion_matrix'][0, 1]}\n")
            f.write(f"  False Negatives: {results['confusion_matrix'][1, 0]}\n")
            f.write(f"  True Positives:  {results['confusion_matrix'][1, 1]}\n\n")
            
            f.write("Top 10 Most Important Features:\n")
            f.write(importance_df.head(10).to_string(index=False))
            f.write("\n\n")
            
            f.write("Interpretation:\n")
            f.write("  Feature importance shows which features the tree splits on most\n")
            f.write("  Higher importance = more critical for classification decisions\n")
            f.write("  Tree structure provides transparent decision rules\n")
        
        print(f"  Report saved: {report_path}")
        
    def run_pipeline(self):
        self.load_engineered_data()
        self.tune_hyperparameters()
        results = self.evaluate_model()
        importance_df = self.analyze_feature_importance()
        self.generate_visualizations(results, importance_df)
        self.save_model()
        self.save_report(results, importance_df)
        
        print("\n" + "="*70)
        print("DECISION TREE PIPELINE COMPLETE")
        print("="*70)


def main():
    pipeline = DecisionTreePipeline()
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
