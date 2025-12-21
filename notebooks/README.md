# Notebooks Directory

This directory contains Jupyter notebooks for exploratory data analysis, feature engineering, modeling, and explainability.

## Notebook Structure

### 1. `eda-fraud-data.ipynb`
**Purpose**: Comprehensive exploratory data analysis of e-commerce fraud data.

**Contents**:
- Data loading and initial inspection
- Missing value analysis
- Univariate distributions
- Bivariate analysis (features vs fraud)
- Class imbalance analysis
- Business insights and risk indicators

### 2. `eda-creditcard.ipynb`
**Purpose**: Exploratory analysis of bank credit card transaction data.

**Contents**:
- PCA feature analysis
- Transaction amount distributions
- Time-based patterns
- Class imbalance assessment
- Anomaly detection insights

### 3. `feature-engineering.ipynb`
**Purpose**: Create fraud-relevant features from raw data.

**Key Features**:
- Transaction frequency per user
- Velocity features (time-window based)
- Time-based features (hour, day, time since signup)
- Geolocation-based features
- Device and browser patterns

### 4. `modeling.ipynb`
**Purpose**: Model training, validation, and evaluation.

**Contents**:
- Train-test split
- Class imbalance handling (SMOTE/undersampling)
- Model training (multiple algorithms)
- Cross-validation
- Performance metrics (precision, recall, F1, AUC-ROC)
- Business cost analysis

### 5. `shap-explainability.ipynb`
**Purpose**: Model interpretability using SHAP values.

**Contents**:
- Feature importance analysis
- SHAP summary plots
- Individual prediction explanations
- Business-friendly visualizations

## Usage

1. Start Jupyter: `jupyter lab` or `jupyter notebook`
2. Navigate to this directory
3. Run notebooks in order for complete analysis
4. Ensure data files are in `../data/raw/` directory

