"""
Logistic Regression Model - Heart Disease Prediction
====================================================

Baseline model using Logistic Regression with custom data structures
for feature engineering and encoding.

Author: NGANA NOAJ Junior Data Scientist
Date: February 2026
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))
from structures import CategoricalEncoder


class LogisticRegressionPipeline:
    
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.feature_names = None
        
    def load_engineered_data(self):
        print("Loading pre-engineered data...")
        
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
        
    def train_model(self):
        print("\nTraining Logistic Regression model...")
        
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver='lbfgs'
        )
        
        self.model.fit(self.X_train, self.y_train)
        print("  Model trained successfully")
        
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
    
    def analyze_coefficients(self):
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE (Coefficients)")
        print("="*70)
        
        coefficients = self.model.coef_[0]
        coef_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': coefficients,
            'Abs_Coefficient': np.abs(coefficients)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(coef_df.head(10).to_string(index=False))
        
        return coef_df
    
    def generate_visualizations(self, results, coef_df):
        print("\nGenerating visualizations...")
        
        viz_dir = Path(__file__).parent.parent / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Logistic Regression - Model Analysis', fontsize=16, fontweight='bold')
        
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        
        top_features = coef_df.head(10)
        colors = ['green' if c > 0 else 'red' for c in top_features['Coefficient']]
        axes[0, 1].barh(range(len(top_features)), top_features['Coefficient'], color=colors)
        axes[0, 1].set_yticks(range(len(top_features)))
        axes[0, 1].set_yticklabels(top_features['Feature'])
        axes[0, 1].set_xlabel('Coefficient Value')
        axes[0, 1].set_title('Top 10 Feature Coefficients')
        axes[0, 1].axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        axes[0, 1].invert_yaxis()
        
        fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
        axes[1, 0].plot(fpr, tpr, color='blue', lw=2, 
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
        colors_bar = ['#2ecc71' if v >= 0.8 else '#f39c12' if v >= 0.7 else '#e74c3c' 
                     for v in values]
        axes[1, 1].bar(metrics, values, color=colors_bar, alpha=0.7, edgecolor='black')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].axhline(y=0.8, color='green', linestyle='--', linewidth=1, alpha=0.5)
        axes[1, 1].axhline(y=0.7, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        for i, v in enumerate(values):
            axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        output_path = viz_dir / 'logistic_regression_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()
        
    def save_model(self):
        print("\nSaving model...")
        
        models_dir = Path(__file__).parent.parent / 'models'
        models_dir.mkdir(exist_ok=True)
        
        with open(models_dir / 'logistic_regression.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        print("  Model saved successfully")
        
    def save_report(self, results, coef_df):
        print("\nGenerating text report...")
        
        reports_dir = Path(__file__).parent.parent / 'reports'
        reports_dir.mkdir(exist_ok=True)
        
        report_path = reports_dir / 'logistic_regression_results.txt'
        
        total_samples = len(self.X_train) + len(self.X_test)
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("LOGISTIC REGRESSION MODEL - EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("Dataset Information:\n")
            f.write(f"  Total samples: {total_samples}\n")
            f.write(f"  Training samples: {len(self.X_train)}\n")
            f.write(f"  Test samples: {len(self.X_test)}\n")
            f.write(f"  Features: {len(self.feature_names)}\n\n")
            
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
            f.write(coef_df.head(10).to_string(index=False))
            f.write("\n\n")
            
            f.write("Interpretation:\n")
            f.write("  Positive coefficients increase disease probability\n")
            f.write("  Negative coefficients decrease disease probability\n")
            f.write("  Larger absolute values indicate stronger influence\n")
        
        print(f"  Report saved: {report_path}")
        
    def run_pipeline(self):
        self.load_engineered_data()
        self.train_model()
        results = self.evaluate_model()
        coef_df = self.analyze_coefficients()
        self.generate_visualizations(results, coef_df)
        self.save_model()
        self.save_report(results, coef_df)
        
        print("\n" + "="*70)
        print("LOGISTIC REGRESSION PIPELINE COMPLETE")
        print("="*70)


def main():
    pipeline = LogisticRegressionPipeline()
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
