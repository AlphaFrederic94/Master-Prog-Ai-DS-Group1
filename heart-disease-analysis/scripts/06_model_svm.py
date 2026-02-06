"""
Support Vector Machine Model - Heart Disease Prediction
========================================================

SVM implementation with kernel optimization (RBF and Linear),
hyperparameter tuning, and decision boundary visualization.

Author: NGANA NOAJ Junior Data Scientist
Date: February 2026
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))


class SVMPipeline:
    
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.feature_names = None
        self.best_params = None
        self.grid_results = None
        
    def load_engineered_data(self):
        print("="*70)
        print("SUPPORT VECTOR MACHINE MODEL TRAINING")
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
        print("  Testing RBF and Linear kernels with class balancing...")
        
        param_grid = [
            {
                'kernel': ['rbf'],
                'C': [1, 10, 50, 100],
                'gamma': ['scale', 0.01, 0.1, 1],
                'class_weight': ['balanced', None]
            },
            {
                'kernel': ['linear'],
                'C': [0.1, 1, 10, 50],
                'class_weight': ['balanced', None]
            }
        ]
        
        svm = SVC(probability=True, random_state=42)
        
        grid_search = GridSearchCV(
            svm, param_grid, cv=5, scoring='roc_auc', 
            n_jobs=-1, verbose=0
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        self.grid_results = pd.DataFrame(grid_search.cv_results_)
        
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
        
        if hasattr(self.model, 'n_support_'):
            print(f"\nSupport Vectors:")
            print(f"  Class 0 (No Disease): {self.model.n_support_[0]} support vectors")
            print(f"  Class 1 (Disease): {self.model.n_support_[1]} support vectors")
            print(f"  Total: {sum(self.model.n_support_)} / {len(self.X_train)} training samples")
        
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
    
    def generate_visualizations(self, results):
        print("\nGenerating visualizations...")
        
        viz_dir = Path(__file__).parent.parent / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Support Vector Machine - Model Analysis', fontsize=16, fontweight='bold')
        
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=axes[0, 0],
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        top_params = self.grid_results.nsmallest(10, 'rank_test_score')[
            ['params', 'mean_test_score', 'std_test_score']
        ]
        
        param_labels = [str(p)[:30] + '...' if len(str(p)) > 30 else str(p) 
                       for p in top_params['params']]
        y_pos = np.arange(len(param_labels))
        
        axes[0, 1].barh(y_pos, top_params['mean_test_score'], 
                       color=plt.cm.Purples(np.linspace(0.4, 0.8, len(y_pos))))
        axes[0, 1].set_yticks(y_pos)
        axes[0, 1].set_yticklabels(param_labels, fontsize=7)
        axes[0, 1].set_xlabel('Mean CV ROC-AUC')
        axes[0, 1].set_title('Top 10 Hyperparameter Configurations')
        axes[0, 1].invert_yaxis()
        
        fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
        axes[1, 0].plot(fpr, tpr, color='purple', lw=2, 
                       label=f'ROC Curve (AUC = {results["roc_auc"]:.3f})')
        axes[1, 0].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curve')
        axes[1, 0].legend(loc='lower right')
        axes[1, 0].grid(alpha=0.3)
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        values = [results['accuracy'], results['precision'], results['recall'], 
                 results['f1'], results['roc_auc']]
        colors_bar = ['#8e44ad' if v >= 0.8 else '#f39c12' if v >= 0.7 else '#e74c3c' 
                     for v in values]
        axes[1, 1].bar(metrics, values, color=colors_bar, alpha=0.7, edgecolor='black')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].axhline(y=0.8, color='purple', linestyle='--', linewidth=1, alpha=0.5)
        for i, v in enumerate(values):
            axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        output_path = viz_dir / 'svm_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()
        
        self._plot_decision_boundary()
        
    def _plot_decision_boundary(self):
        print("  Generating decision boundary visualization (2D PCA projection)...")
        
        pca = PCA(n_components=2, random_state=42)
        X_train_pca = pca.fit_transform(self.X_train)
        X_test_pca = pca.transform(self.X_test)
        
        svm_2d = SVC(kernel=self.best_params['kernel'], 
                     C=self.best_params['C'],
                     gamma=self.best_params.get('gamma', 'scale'),
                     probability=True, random_state=42)
        svm_2d.fit(X_train_pca, self.y_train)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
        y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                            np.linspace(y_min, y_max, 200))
        
        Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlGn', levels=1)
        ax.contour(xx, yy, Z, colors='black', linewidths=0.5, levels=1)
        
        scatter_train = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], 
                                  c=self.y_train, cmap='RdYlGn', 
                                  edgecolors='black', s=50, alpha=0.6, 
                                  label='Training Data')
        
        ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], 
                  c=self.y_test, cmap='RdYlGn', 
                  edgecolors='blue', linewidths=2, s=100, alpha=0.8,
                  marker='s', label='Test Data')
        
        if hasattr(svm_2d, 'support_vectors_'):
            ax.scatter(svm_2d.support_vectors_[:, 0], svm_2d.support_vectors_[:, 1],
                      s=200, linewidth=1.5, facecolors='none', edgecolors='purple',
                      label='Support Vectors')
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', 
                     fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', 
                     fontweight='bold')
        ax.set_title(f'SVM Decision Boundary ({self.best_params["kernel"].upper()} Kernel)\n' +
                    f'2D PCA Projection', fontweight='bold', fontsize=14)
        ax.legend(loc='best')
        ax.grid(alpha=0.3)
        
        cbar = plt.colorbar(scatter_train, ax=ax)
        cbar.set_label('Class (0=No Disease, 1=Disease)', rotation=270, labelpad=20)
        
        viz_dir = Path(__file__).parent.parent / 'visualizations'
        boundary_path = viz_dir / 'svm_decision_boundary.png'
        plt.savefig(boundary_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {boundary_path}")
        plt.close()
        
    def save_model(self):
        print("\nSaving model...")
        
        models_dir = Path(__file__).parent.parent / 'models'
        models_dir.mkdir(exist_ok=True)
        
        with open(models_dir / 'svm.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        print("  Model saved successfully")
        
    def save_report(self, results):
        print("\nGenerating text report...")
        
        reports_dir = Path(__file__).parent.parent / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        report_path = reports_dir / 'svm_results.txt'
        
        total_samples = len(self.X_train) + len(self.X_test)
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("SUPPORT VECTOR MACHINE MODEL - EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("Dataset Information:\n")
            f.write(f"  Total samples: {total_samples}\n")
            f.write(f"  Training samples: {len(self.X_train)}\n")
            f.write(f"  Test samples: {len(self.X_test)}\n")
            f.write(f"  Features: {len(self.feature_names)}\n\n")
            
            f.write("Model Configuration:\n")
            f.write(f"  Best parameters: {self.best_params}\n")
            if hasattr(self.model, 'n_support_'):
                f.write(f"  Support vectors (Class 0): {self.model.n_support_[0]}\n")
                f.write(f"  Support vectors (Class 1): {self.model.n_support_[1]}\n")
                f.write(f"  Total support vectors: {sum(self.model.n_support_)}\n\n")
            
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
            
            f.write("Interpretation:\n")
            f.write("  SVM finds optimal hyperplane that maximizes margin between classes\n")
            f.write("  Support vectors are the critical data points defining the boundary\n")
            f.write("  Kernel trick enables non-linear decision boundaries\n")
        
        print(f"  Report saved: {report_path}")
        
    def run_pipeline(self):
        self.load_engineered_data()
        self.tune_hyperparameters()
        results = self.evaluate_model()
        self.generate_visualizations(results)
        self.save_model()
        self.save_report(results)
        
        print("\n" + "="*70)
        print("SVM PIPELINE COMPLETE")
        print("="*70)


def main():
    pipeline = SVMPipeline()
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
