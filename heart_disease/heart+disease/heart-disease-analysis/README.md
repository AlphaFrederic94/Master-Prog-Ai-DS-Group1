# UCI Heart Disease Dataset Analysis

## 📋 Project Overview

This project contains a comprehensive analysis of the UCI Heart Disease Dataset from 1988. The dataset includes medical records from four institutions for predicting the presence of heart disease.

## 🗂️ Project Structure

```
heart-disease-analysis/
├── data/
│   ├── raw/              # Original unprocessed data files
│   └── processed/        # Cleaned, ML-ready datasets
├── scripts/
│   ├── 01_data_processing.py      # Data cleaning and preprocessing
│   └── 02_exploratory_analysis.py # EDA and visualization
├── notebooks/            # Jupyter notebooks for analysis
├── reports/              # Generated analysis reports
├── visualizations/       # Generated plots and charts
└── README.md            # This file
```

## 📊 Dataset Information

### Source
- **Repository**: UCI Machine Learning Repository
- **Dataset ID**: 45
- **URL**: https://archive.ics.uci.edu/dataset/45/heart+disease
- **Year**: 1988

### Databases Included
1. **Cleveland Clinic Foundation** (303 instances) - Most commonly used
2. **Hungarian Institute of Cardiology** (294 instances)
3. **V.A. Medical Center, Long Beach** (200 instances)
4. **University Hospital, Zurich, Switzerland** (123 instances)

### Features (14 attributes used)
1. **age**: Age in years
2. **sex**: Sex (1 = male; 0 = female)
3. **cp**: Chest pain type (1-4)
4. **trestbps**: Resting blood pressure (mm Hg)
5. **chol**: Serum cholesterol (mg/dl)
6. **fbs**: Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
7. **restecg**: Resting electrocardiographic results (0-2)
8. **thalach**: Maximum heart rate achieved
9. **exang**: Exercise induced angina (1 = yes; 0 = no)
10. **oldpeak**: ST depression induced by exercise relative to rest
11. **slope**: Slope of the peak exercise ST segment (1-3)
12. **ca**: Number of major vessels colored by fluoroscopy (0-3)
13. **thal**: Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)
14. **target**: Diagnosis of heart disease (0 = no disease; 1-4 = disease present)

### Target Variable
- **Original**: 0-4 (0 = no disease, 1-4 = varying degrees of disease)
- **Binary**: 0 = no disease, 1 = disease present

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Running the Analysis
1. **Process the data**:
   ```bash
   python scripts/01_data_processing.py
   ```

2. **Perform exploratory analysis**:
   ```bash
   python scripts/02_exploratory_analysis.py
   ```

## 📈 Key Findings
Results will be documented in the `reports/` directory after running the analysis.

## 📚 Citation

If you use this dataset, please cite:

**Creators**:
- Hungarian Institute of Cardiology, Budapest: Andras Janosi, M.D.
- University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
- University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
- V.A. Medical Center, Long Beach and Cleveland Clinic Foundation: Robert Detrano, M.D., Ph.D.

**Reference Paper**:
Detrano, R., Janosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J., Sandhu, S., Guppy, K., Lee, S., & Froelicher, V. (1989). International application of a new probability algorithm for the diagnosis of coronary artery disease. American Journal of Cardiology, 64, 304-310.

## 📝 License
This dataset is publicly available through the UCI Machine Learning Repository.

---
**Last Updated**: January 2026
