"""
UCI Heart Disease Dataset - Data Processing Pipeline
=====================================================

This script performs comprehensive data cleaning and preprocessing on the
UCI Heart Disease dataset from four institutions.

Author: Junior Data Scientist Group ICT University
Date: January 2026
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class HeartDiseaseProcessor:
    """
    A comprehensive data processor for the UCI Heart Disease dataset.
    Handles loading, cleaning, and preprocessing of multi-institutional data.
    """
    
    def __init__(self, base_dir):
        """
        Initialize the processor with base directory paths.
        
        Parameters:
        -----------
        base_dir : str or Path
            Base directory containing the heart disease data
        """
        self.base_dir = Path(base_dir)
        self.raw_data_dir = self.base_dir / 'data' / 'raw'
        self.processed_data_dir = self.base_dir / 'data' / 'processed'
        
        # Define column names based on UCI documentation
        self.column_names = [
            'age',        # Age in years
            'sex',        # 1 = male, 0 = female
            'cp',         # Chest pain type (1-4)
            'trestbps',   # Resting blood pressure (mm Hg)
            'chol',       # Serum cholesterol (mg/dl)
            'fbs',        # Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
            'restecg',    # Resting electrocardiographic results (0-2)
            'thalach',    # Maximum heart rate achieved
            'exang',      # Exercise induced angina (1 = yes, 0 = no)
            'oldpeak',    # ST depression induced by exercise relative to rest
            'slope',      # Slope of the peak exercise ST segment (1-3)
            'ca',         # Number of major vessels colored by fluoroscopy (0-3)
            'thal',       # Thalassemia (3, 6, 7)
            'num'         # Diagnosis (0-4, where 0 = no disease)
        ]
        
        # Dataset metadata - processed data files are in parent directory
        self.datasets = {
            'cleveland': self.base_dir.parent / 'processed.cleveland.data',
            'hungarian': self.base_dir.parent / 'processed.hungarian.data',
            'switzerland': self.base_dir.parent / 'processed.switzerland.data',
            'va': self.base_dir.parent / 'processed.va.data'
        }
        
    def load_single_dataset(self, filepath, dataset_name):
        """
        Load a single heart disease dataset file.
        
        Parameters:
        -----------
        filepath : str or Path
            Path to the data file
        dataset_name : str
            Name identifier for the dataset
            
        Returns:
        --------
        pd.DataFrame
            Loaded dataset with proper column names
        """
        try:
            # Read CSV without headers
            df = pd.read_csv(
                filepath,
                names=self.column_names,
                na_values='?',  # Treat '?' as missing values
                skipinitialspace=True
            )
            
            # Add source column
            df['source'] = dataset_name
            
            print(f"✓ Loaded {dataset_name}: {len(df)} records")
            return df
            
        except FileNotFoundError:
            print(f"✗ Warning: {filepath} not found. Skipping.")
            return None
        except Exception as e:
            print(f"✗ Error loading {dataset_name}: {str(e)}")
            return None
    
    def load_all_datasets(self):
        """
        Load all available heart disease datasets.
        
        Returns:
        --------
        pd.DataFrame
            Combined dataframe from all sources
        """
        print("\n" + "="*60)
        print("LOADING DATASETS")
        print("="*60)
        
        dfs = []
        for name, filepath in self.datasets.items():
            df = self.load_single_dataset(filepath, name)
            if df is not None:
                dfs.append(df)
        
        if not dfs:
            raise ValueError("No datasets could be loaded!")
        
        # Combine all datasets
        combined_df = pd.concat(dfs, ignore_index=True)
        print(f"\n✓ Total records combined: {len(combined_df)}")
        print("="*60)
        
        return combined_df
    
    def create_binary_target(self, df):
        """
        Create binary classification target variable.
        
        Original 'num' values:
        - 0: No heart disease
        - 1-4: Heart disease present (varying severity)
        
        Binary target:
        - 0: No heart disease
        - 1: Heart disease present
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with binary target column added
        """
        df['target'] = (df['num'] > 0).astype(int)
        return df
    
    def generate_data_quality_report(self, df):
        """
        Generate comprehensive data quality report.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        dict
            Dictionary containing quality metrics
        """
        report = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicates': df.duplicated().sum(),
            'data_types': df.dtypes.to_dict()
        }
        
        return report
    
    def print_data_quality_report(self, report):
        """
        Print formatted data quality report.
        
        Parameters:
        -----------
        report : dict
            Data quality report dictionary
        """
        print("\n" + "="*60)
        print("DATA QUALITY REPORT")
        print("="*60)
        print(f"Total Records: {report['total_records']:,}")
        print(f"Total Features: {report['total_features']}")
        print(f"Duplicate Rows: {report['duplicates']}")
        
        print("\n" + "-"*60)
        print("MISSING VALUES SUMMARY")
        print("-"*60)
        
        missing_data = pd.DataFrame({
            'Missing Count': report['missing_values'],
            'Percentage': report['missing_percentage']
        })
        
        # Only show features with missing values
        missing_data = missing_data[missing_data['Missing Count'] > 0]
        
        if len(missing_data) > 0:
            print(missing_data.to_string())
        else:
            print("✓ No missing values detected!")
        
        print("="*60)
    
    def handle_missing_values(self, df, strategy='median'):
        """
        Handle missing values using specified strategy.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        strategy : str
            Strategy for handling missing values ('median', 'mean', 'mode', 'drop')
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with missing values handled
        """
        df_clean = df.copy()
        
        print("\n" + "="*60)
        print(f"HANDLING MISSING VALUES (Strategy: {strategy})")
        print("="*60)
        
        # For 'ca' and 'thal' which have missing values
        numeric_cols_with_missing = df_clean.select_dtypes(include=[np.number]).columns
        numeric_cols_with_missing = [col for col in numeric_cols_with_missing 
                                     if df_clean[col].isnull().any()]
        
        for col in numeric_cols_with_missing:
            missing_count = df_clean[col].isnull().sum()
            
            if strategy == 'median':
                fill_value = df_clean[col].median()
                df_clean[col].fillna(fill_value, inplace=True)
                print(f"✓ {col}: Filled {missing_count} missing values with median ({fill_value:.2f})")
                
            elif strategy == 'mean':
                fill_value = df_clean[col].mean()
                df_clean[col].fillna(fill_value, inplace=True)
                print(f"✓ {col}: Filled {missing_count} missing values with mean ({fill_value:.2f})")
                
            elif strategy == 'mode':
                fill_value = df_clean[col].mode()[0]
                df_clean[col].fillna(fill_value, inplace=True)
                print(f"✓ {col}: Filled {missing_count} missing values with mode ({fill_value})")
        
        if strategy == 'drop':
            initial_rows = len(df_clean)
            df_clean.dropna(inplace=True)
            dropped_rows = initial_rows - len(df_clean)
            print(f"✓ Dropped {dropped_rows} rows with missing values")
        
        print("="*60)
        
        return df_clean
    
    def save_processed_data(self, df, filename):
        """
        Save processed dataframe to CSV.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Processed dataframe
        filename : str
            Output filename
        """
        # Ensure directory exists
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = self.processed_data_dir / filename
        df.to_csv(filepath, index=False)
        print(f"\n✓ Saved: {filepath}")
        print(f"  Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
    
    def process_all(self):
        """
        Execute the complete data processing pipeline.
        
        Returns:
        --------
        tuple
            (combined_df, cleveland_df) - Processed dataframes
        """
        print("\n" + "🔬 " + "="*58)
        print("UCI HEART DISEASE DATASET - PROCESSING PIPELINE")
        print("="*60 + "\n")
        
        # Step 1: Load all datasets
        df_combined = self.load_all_datasets()
        
        # Step 2: Create binary target
        df_combined = self.create_binary_target(df_combined)
        
        # Step 3: Generate and print quality report
        quality_report = self.generate_data_quality_report(df_combined)
        self.print_data_quality_report(quality_report)
        
        # Step 4: Handle missing values
        df_combined_clean = self.handle_missing_values(df_combined, strategy='median')
        
        # Step 5: Create Cleveland-only dataset (most commonly used)
        df_cleveland = df_combined_clean[df_combined_clean['source'] == 'cleveland'].copy()
        
        # Step 6: Save processed datasets
        print("\n" + "="*60)
        print("SAVING PROCESSED DATASETS")
        print("="*60)
        
        self.save_processed_data(df_combined_clean, 'heart_disease_combined.csv')
        self.save_processed_data(df_cleveland, 'heart_disease_cleveland.csv')
        
        # Step 7: Generate summary statistics
        self.save_summary_statistics(df_combined_clean, df_cleveland)
        
        print("\n" + "="*60)
        print("✓ PROCESSING COMPLETE!")
        print("="*60 + "\n")
        
        return df_combined_clean, df_cleveland
    
    def save_summary_statistics(self, df_combined, df_cleveland):
        """
        Save summary statistics to text file.
        
        Parameters:
        -----------
        df_combined : pd.DataFrame
            Combined dataset
        df_cleveland : pd.DataFrame
            Cleveland dataset
        """
        reports_dir = self.base_dir / 'reports'
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = reports_dir / 'data_summary.txt'
        
        with open(filepath, 'w') as f:
            f.write("="*70 + "\n")
            f.write("UCI HEART DISEASE DATASET - SUMMARY STATISTICS\n")
            f.write("="*70 + "\n\n")
            
            # Combined dataset stats
            f.write("COMBINED DATASET (All 4 Institutions)\n")
            f.write("-"*70 + "\n")
            f.write(f"Total Records: {len(df_combined):,}\n")
            f.write(f"Features: {len(df_combined.columns)}\n\n")
            
            f.write("Target Distribution:\n")
            f.write(df_combined['target'].value_counts().to_string() + "\n\n")
            
            f.write("Source Distribution:\n")
            f.write(df_combined['source'].value_counts().to_string() + "\n\n")
            
            f.write("\nDescriptive Statistics:\n")
            f.write("-"*70 + "\n")
            f.write(df_combined.describe().to_string() + "\n\n")
            
            # Cleveland dataset stats
            f.write("\n" + "="*70 + "\n")
            f.write("CLEVELAND DATASET (Most Commonly Used)\n")
            f.write("-"*70 + "\n")
            f.write(f"Total Records: {len(df_cleveland):,}\n")
            f.write(f"Features: {len(df_cleveland.columns)}\n\n")
            
            f.write("Target Distribution:\n")
            f.write(df_cleveland['target'].value_counts().to_string() + "\n\n")
            
            f.write("\nDescriptive Statistics:\n")
            f.write("-"*70 + "\n")
            f.write(df_cleveland.describe().to_string() + "\n")
        
        print(f"✓ Saved summary statistics: {filepath}")


def main():
    """Main execution function."""
    # Set base directory to the heart-disease-analysis folder
    base_dir = Path(__file__).parent.parent
    
    # Initialize processor
    processor = HeartDiseaseProcessor(base_dir)
    
    # Execute processing pipeline
    df_combined, df_cleveland = processor.process_all()
    
    # Display sample data
    print("\nSample of processed data (first 5 rows):")
    print("-"*60)
    print(df_cleveland[['age', 'sex', 'cp', 'trestbps', 'chol', 'target']].head())


if __name__ == "__main__":
    main()
