"""
UCI Heart Disease Dataset - Exploratory Data Analysis
======================================================

This script performs comprehensive exploratory data analysis and
visualization on the processed heart disease dataset.

Author: Senior Data Scientist
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class HeartDiseaseEDA:
    """
    Comprehensive Exploratory Data Analysis for Heart Disease Dataset.
    """
    
    def __init__(self, base_dir):
        """
        Initialize EDA class.
        
        Parameters:
        -----------
        base_dir : str or Path
            Base directory containing the processed data
        """
        self.base_dir = Path(base_dir)
        self.processed_data_dir = self.base_dir / 'data' / 'processed'
        self.viz_dir = self.base_dir / 'visualizations'
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature categories
        self.categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
        self.continuous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        
    def load_data(self, filename='heart_disease_cleveland.csv'):
        """Load processed dataset."""
        filepath = self.processed_data_dir / filename
        df = pd.read_csv(filepath)
        print(f"✓ Loaded dataset: {filepath}")
        print(f"  Shape: {df.shape}")
        return df
    
    def plot_target_distribution(self, df):
        """Plot distribution of target variable."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        target_counts = df['target'].value_counts()
        axes[0].bar(target_counts.index, target_counts.values, color=['#2ecc71', '#e74c3c'])
        axes[0].set_xlabel('Target (0 = No Disease, 1 = Disease)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[0].set_title('Target Variable Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xticks([0, 1])
        
        # Add value labels on bars
        for i, v in enumerate(target_counts.values):
            axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')
        
        # Pie chart
        colors = ['#2ecc71', '#e74c3c']
        axes[1].pie(target_counts.values, labels=['No Disease', 'Disease'], autopct='%1.1f%%',
                   colors=colors, startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
        axes[1].set_title('Target Variable Proportion', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        filepath = self.viz_dir / '01_target_distribution.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_age_distribution(self, df):
        """Plot age distribution by target."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist([df[df['target']==0]['age'], df[df['target']==1]['age']], 
                     bins=20, label=['No Disease', 'Disease'], color=['#2ecc71', '#e74c3c'], alpha=0.7)
        axes[0].set_xlabel('Age', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title('Age Distribution by Target', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(alpha=0.3)
        
        # Box plot
        df.boxplot(column='age', by='target', ax=axes[1], patch_artist=True,
                   boxprops=dict(facecolor='lightblue', color='navy'),
                   medianprops=dict(color='red', linewidth=2))
        axes[1].set_xlabel('Target (0 = No Disease, 1 = Disease)', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Age', fontsize=12, fontweight='bold')
        axes[1].set_title('Age Distribution by Target', fontsize=14, fontweight='bold')
        plt.suptitle('')
        
        plt.tight_layout()
        filepath = self.viz_dir / '02_age_distribution.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_gender_analysis(self, df):
        """Analyze gender distribution and its relationship with disease."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gender distribution
        gender_counts = df['sex'].value_counts()
        axes[0].bar(['Female', 'Male'], [gender_counts[0], gender_counts[1]], 
                   color=['#ff69b4', '#4169e1'])
        axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[0].set_title('Gender Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate([gender_counts[0], gender_counts[1]]):
            axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')
        
        # Gender vs Target
        gender_target = pd.crosstab(df['sex'], df['target'], normalize='index') * 100
        gender_target.plot(kind='bar', stacked=False, ax=axes[1], 
                          color=['#2ecc71', '#e74c3c'], width=0.7)
        axes[1].set_xticklabels(['Female', 'Male'], rotation=0, fontsize=12)
        axes[1].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Disease Presence by Gender', fontsize=14, fontweight='bold')
        axes[1].legend(['No Disease', 'Disease'], fontsize=10)
        axes[1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = self.viz_dir / '03_gender_analysis.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_chest_pain_analysis(self, df):
        """Analyze chest pain types."""
        cp_labels = {1: 'Typical Angina', 2: 'Atypical Angina', 
                     3: 'Non-Anginal', 4: 'Asymptomatic'}
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # CP distribution
        cp_counts = df['cp'].value_counts().sort_index()
        axes[0].bar(range(len(cp_counts)), cp_counts.values, color='steelblue')
        axes[0].set_xticks(range(len(cp_counts)))
        axes[0].set_xticklabels([cp_labels[i+1] for i in range(len(cp_counts))], 
                                rotation=45, ha='right', fontsize=10)
        axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
        axes[0].set_title('Chest Pain Type Distribution', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3, axis='y')
        
        # CP vs Target
        cp_target = pd.crosstab(df['cp'], df['target'], normalize='index') * 100
        cp_target.plot(kind='bar', stacked=True, ax=axes[1], 
                      color=['#2ecc71', '#e74c3c'])
        axes[1].set_xticklabels([cp_labels[i] for i in cp_target.index], 
                                rotation=45, ha='right', fontsize=10)
        axes[1].set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Disease Presence by Chest Pain Type', fontsize=14, fontweight='bold')
        axes[1].legend(['No Disease', 'Disease'], fontsize=10)
        axes[1].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = self.viz_dir / '04_chest_pain_analysis.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_continuous_features(self, df):
        """Plot distribution of continuous features."""
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.ravel()
        
        feature_info = {
            'age': 'Age (years)',
            'trestbps': 'Resting Blood Pressure (mm Hg)',
            'chol': 'Serum Cholesterol (mg/dl)',
            'thalach': 'Max Heart Rate',
            'oldpeak': 'ST Depression'
        }
        
        for idx, (feature, label) in enumerate(feature_info.items()):
            # Create violin plot
            parts = axes[idx].violinplot(
                [df[df['target']==0][feature].dropna(), 
                 df[df['target']==1][feature].dropna()],
                positions=[0, 1],
                showmeans=True,
                showmedians=True
            )
            
            # Color the violins
            for pc, color in zip(parts['bodies'], ['#2ecc71', '#e74c3c']):
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
            
            axes[idx].set_xticks([0, 1])
            axes[idx].set_xticklabels(['No Disease', 'Disease'])
            axes[idx].set_ylabel(label, fontsize=11, fontweight='bold')
            axes[idx].set_title(f'{label} by Target', fontsize=12, fontweight='bold')
            axes[idx].grid(alpha=0.3, axis='y')
        
        # Remove extra subplot
        fig.delaxes(axes[5])
        
        plt.tight_layout()
        filepath = self.viz_dir / '05_continuous_features.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_correlation_matrix(self, df):
        """Plot correlation heatmap."""
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(14, 12))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        filepath = self.viz_dir / '06_correlation_matrix.png'
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def generate_statistical_summary(self, df):
        """Generate comprehensive statistical summary."""
        reports_dir = self.base_dir / 'reports'
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = reports_dir / 'eda_summary.txt'
        
        with open(filepath, 'w') as f:
            f.write("="*70 + "\n")
            f.write("EXPLORATORY DATA ANALYSIS - STATISTICAL SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            # Basic statistics
            f.write("DATASET OVERVIEW\n")
            f.write("-"*70 + "\n")
            f.write(f"Total Records: {len(df):,}\n")
            f.write(f"Total Features: {df.shape[1]}\n\n")
            
            # Target distribution
            f.write("TARGET DISTRIBUTION\n")
            f.write("-"*70 + "\n")
            target_dist = df['target'].value_counts()
            f.write(f"No Disease (0): {target_dist[0]} ({target_dist[0]/len(df)*100:.1f}%)\n")
            f.write(f"Disease (1): {target_dist[1]} ({target_dist[1]/len(df)*100:.1f}%)\n\n")
            
            # Gender analysis
            f.write("GENDER ANALYSIS\n")
            f.write("-"*70 + "\n")
            gender_dist = df['sex'].value_counts()
            f.write(f"Female (0): {gender_dist[0]} ({gender_dist[0]/len(df)*100:.1f}%)\n")
            f.write(f"Male (1): {gender_dist[1]} ({gender_dist[1]/len(df)*100:.1f}%)\n\n")
            
            # Age statistics by target
            f.write("AGE STATISTICS BY TARGET\n")
            f.write("-"*70 + "\n")
            f.write("No Disease:\n")
            f.write(f"  Mean: {df[df['target']==0]['age'].mean():.1f} years\n")
            f.write(f"  Median: {df[df['target']==0]['age'].median():.1f} years\n")
            f.write(f"  Std Dev: {df[df['target']==0]['age'].std():.1f} years\n\n")
            f.write("Disease:\n")
            f.write(f"  Mean: {df[df['target']==1]['age'].mean():.1f} years\n")
            f.write(f"  Median: {df[df['target']==1]['age'].median():.1f} years\n")
            f.write(f"  Std Dev: {df[df['target']==1]['age'].std():.1f} years\n\n")
            
            # Full descriptive statistics
            f.write("\nDESCRIPTIVE STATISTICS (ALL FEATURES)\n")
            f.write("-"*70 + "\n")
            f.write(df.describe().to_string())
        
        print(f"✓ Saved statistical summary: {filepath}")
    
    def run_eda(self):
        """Execute complete EDA pipeline."""
        print("\n" + "📊 " + "="*58)
        print("EXPLORATORY DATA ANALYSIS PIPELINE")
        print("="*60 + "\n")
        
        # Load data
        df = self.load_data()
        
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        
        # Generate all visualizations
        self.plot_target_distribution(df)
        self.plot_age_distribution(df)
        self.plot_gender_analysis(df)
        self.plot_chest_pain_analysis(df)
        self.plot_continuous_features(df)
        self.plot_correlation_matrix(df)
        
        # Generate statistical summary
        print("\n" + "="*60)
        print("GENERATING STATISTICAL SUMMARY")
        print("="*60)
        self.generate_statistical_summary(df)
        
        print("\n" + "="*60)
        print("✓ EDA COMPLETE!")
        print("="*60 + "\n")
        print(f"📁 Visualizations saved to: {self.viz_dir}")
        print(f"📄 Reports saved to: {self.base_dir / 'heart-disease-analysis' / 'reports'}")


def main():
    """Main execution function."""
    # Set base directory
    base_dir = Path(__file__).parent.parent
    
    # Initialize and run EDA
    eda = HeartDiseaseEDA(base_dir)
    eda.run_eda()


if __name__ == "__main__":
    main()
