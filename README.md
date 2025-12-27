# Fintech Fraud Detection System

**Adey Innovations Inc.** - Production-Grade Fraud Detection Pipeline

## Table of Contents

- [Project Overview](#project-overview)
- [Business Context](#business-context)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Branch Structure](#branch-structure)
- [Task 1: Data Analysis and Preprocessing](#task-1-data-analysis-and-preprocessing)
- [Task 2: Model Building & Training](#task-2-model-building--training)
- [Development Workflow](#development-workflow)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This repository contains a comprehensive fraud detection system designed for e-commerce and bank credit transaction fraud detection. The project follows industry best practices for data science workflows, emphasizing business-driven decision-making, class imbalance handling, and production-ready implementations.

## Business Context

- **Fraud datasets are highly imbalanced** - requiring sophisticated resampling strategies
- **False positives damage user experience** - impacting customer trust and retention
- **False negatives cause direct financial loss** - requiring cost-sensitive modeling
- **Real-time decision-making** - models must support low-latency inference
- **Feature quality over complexity** - interpretable, robust features drive business value

## Project Structure

```
fraud-detection/
├── .vscode/              # VS Code workspace settings
├── .github/              # CI/CD workflows
│   └── workflows/
│       └── unittests.yml
├── data/                 # Data storage (gitignored)
│   ├── raw/             # Original datasets
│   └── processed/       # Cleaned and transformed data
├── notebooks/            # Jupyter notebooks for analysis
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   └── shap-explainability.ipynb
├── src/                  # Source code modules
├── tests/                # Unit and integration tests
├── models/               # Trained model artifacts (gitignored)
├── scripts/              # Utility scripts
├── requirements.txt      # Python dependencies
└── README.md
```

## Datasets

1. **Fraud_Data.csv** - E-commerce transaction data with user, device, and transaction features
2. **IpAddress_to_Country.csv** - IP address range to country mapping
3. **creditcard.csv** - Bank credit card transaction data (PCA-anonymized features)

## Getting Started

### Prerequisites

- **Python 3.9+** (Python 3.12.0 recommended)
  - Verify installation: `py --version` (Windows) or `python3 --version` (Linux/Mac)
- **Git** for version control
- **Internet connection** for package installation
- **~2GB free disk space** for virtual environment and packages

### Installation

#### Step 1: Create Virtual Environment

**Windows (PowerShell):**
```powershell
# Navigate to project directory
cd Fintech-Fraud-Detect

# Create virtual environment using Python launcher
py -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# If activation fails due to execution policy, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows (Command Prompt):**
```cmd
cd Fintech-Fraud-Detect
py -m venv venv
venv\Scripts\activate.bat
```

**Linux/Mac:**
```bash
cd Fintech-Fraud-Detect
python3 -m venv venv
source venv/bin/activate
```

#### Step 2: Upgrade pip

```bash
# Windows (using venv Python)
.\venv\Scripts\python.exe -m pip install --upgrade pip

# Linux/Mac
pip install --upgrade pip
```

#### Step 3: Install Dependencies

```bash
# Windows (using venv Python)
.\venv\Scripts\python.exe -m pip install -r requirements.txt

# Linux/Mac
pip install -r requirements.txt
```

**Note**: If you encounter network issues (especially with large packages like `xgboost`), you can:
1. Retry the installation
2. Install packages individually: `pip install pandas numpy scikit-learn ...`
3. Use a different package index: `pip install -r requirements.txt -i https://pypi.org/simple`

#### Step 4: Verify Installation

```bash
# Windows
.\venv\Scripts\python.exe -m pip list

# Linux/Mac
pip list
```

You should see all required packages installed. Key packages include:
- pandas, numpy, scipy
- scikit-learn, imbalanced-learn
- xgboost, lightgbm
- matplotlib, seaborn, plotly
- jupyter, jupyterlab
- shap

### Branch Structure

- `main` - Production-ready code
- `task-1-dev` - Task 1: Data Analysis and Preprocessing
- `task-2-dev` - Task 2: Model Building & Training

## Task 1: Data Analysis and Preprocessing

**Objective**: Prepare clean, feature-rich datasets ready for modeling.

**Deliverables**:
- Comprehensive EDA with business insights
- Geolocation integration
- Fraud-relevant feature engineering
- Class imbalance handling
- Reproducible preprocessing pipeline

**Status**: ✅ Complete - Virtual environment created, ready for data processing

### Running the Notebooks

After installing dependencies, you can start Jupyter:

```bash
# Windows
.\venv\Scripts\python.exe -m jupyter lab

# Linux/Mac
jupyter lab
```

Then navigate to the `notebooks/` directory and run:
1. `eda-fraud-data.ipynb` - E-commerce fraud analysis
2. `eda-creditcard.ipynb` - Credit card transaction analysis
3. `feature-engineering.ipynb` - Feature engineering pipeline

## Task 2: Model Building & Training

**Objective**: Build, train, evaluate, and select fraud detection classification models that balance detection performance, stability under class imbalance, and interpretability for business stakeholders.

### Modeling Approach

**Data Preparation**:
- Load processed data from Task 1 (feature-engineered datasets)
- Stratified train-test split (80-20) to preserve class distribution
- No data leakage: Split performed before any resampling or feature engineering
- Justification: 80-20 split balances training data size with reliable test evaluation

**Baseline Model: Logistic Regression**
- **Rationale**: Interpretable, fast, establishes minimum acceptable performance
- **Class Imbalance Handling**: `class_weight='balanced'` automatically adjusts for imbalance
- **Evaluation**: AUC-PR (primary), F1-Score, Confusion Matrix
- **Interpretability**: Coefficients directly show feature importance and direction

**Ensemble Model: XGBoost / LightGBM / Random Forest**
- **Rationale**: Higher performance, captures non-linear relationships and feature interactions
- **Model Selection**: XGBoost (preferred) → LightGBM → Random Forest (fallback)
- **Hyperparameter Tuning**: Basic grid search on key parameters (n_estimators, max_depth, learning_rate)
- **Class Imbalance Handling**: `scale_pos_weight` for XGBoost/LightGBM, `class_weight='balanced'` for Random Forest

**Cross-Validation: Stratified K-Fold (k=5)**
- **Why Stratified**: Preserves class distribution in each fold (critical for imbalanced data)
- **Why k=5**: Balance between computational cost and statistical reliability
- **Metrics**: Mean ± std of AUC-PR and F1-Score across folds
- **Business Justification**: Fraud detection requires reliable performance estimates across different data splits

### Models Trained

1. **Logistic Regression** (Baseline)
   - Class-weighted for imbalance handling
   - Fast training and inference
   - High interpretability (coefficients)

2. **XGBoost / LightGBM / Random Forest** (Ensemble)
   - Hyperparameter-tuned via grid search
   - Handles non-linear patterns and feature interactions
   - Feature importance available

### Evaluation Metrics

**Primary Metric: AUC-PR (Average Precision)**
- Best metric for imbalanced fraud detection
- Focuses on the positive class (fraud)
- Business-relevant: Shows precision-recall trade-offs

**Secondary Metrics**:
- **F1-Score**: Balances precision and recall
- **AUC-ROC**: Overall model discrimination ability
- **Confusion Matrix**: Shows false positives vs false negatives (business trade-offs)

**Visualizations**:
- Precision-Recall curves (model comparison)
- ROC curves (complementary view)
- Normalized confusion matrices (business impact)

### Results Summary

**Model Comparison**:
- Comprehensive comparison table with test set and cross-validation metrics
- Stability analysis (CV std deviation)
- Business impact analysis (false positives vs false negatives)

**Model Selection Criteria**:
1. Primary Metric (AUC-PR): Best performance on imbalanced data
2. Stability: Lower variance in cross-validation scores
3. Interpretability: Ability to explain decisions to business stakeholders
4. Business Risk Trade-offs: Balance between false positives and false negatives

**Selected Model**: Best model chosen based on performance, stability, and interpretability trade-offs

### Key Findings & Insights

**Fraud Detection Capability**:
- Detection rate (recall) indicates percentage of fraud cases caught
- Missed fraud (false negatives) = potential financial loss

**Customer Experience Impact**:
- False positive rate indicates legitimate transactions incorrectly flagged
- Impacts customer trust and may lead to transaction declines

**Precision**:
- When model flags fraud, precision shows percentage that are actually fraud
- Higher precision = fewer false alarms = better customer experience

**Model Stability**:
- Cross-validation standard deviation indicates consistency across data splits
- Critical for production reliability

### Selected Model Justification

The selected model is chosen based on:
- **Performance**: Highest AUC-PR on test set
- **Stability**: Low variance in cross-validation scores
- **Interpretability**: Sufficient for business and risk teams
- **Business Context**: Balances fraud detection with customer experience

**Recommendations**:
- Threshold tuning based on business costs (fraud cost vs false positive cost)
- Model monitoring in production (track performance metrics over time)
- Feature engineering review (top features for business insights)
- Periodic retraining (incorporate new fraud patterns)
- Interpretability tools (SHAP values if ensemble selected)

### Artifacts Saved

All model artifacts are saved to `/models` directory:
- `baseline_logistic_regression.pkl` - Trained baseline model
- `ensemble_{type}.pkl` - Trained ensemble model
- `selected_model.pkl` - Final selected model
- `preprocessing_artifacts.pkl` - Preprocessing pipeline (if available)
- `model_evaluation_results.json` - Complete evaluation results
- `baseline_feature_importance.csv` - Logistic regression coefficients
- `ensemble_{type}_feature_importance.csv` - Ensemble feature importance

**Status**: ✅ Complete - Models trained, evaluated, and artifacts saved

### Running the Modeling Notebook

After completing Task 1, run the modeling notebook:

```bash
# Start Jupyter Lab (if not already running)
.\venv\Scripts\python.exe -m jupyter lab  # Windows
jupyter lab  # Linux/Mac
```

Navigate to `notebooks/modeling.ipynb` and run all cells. The notebook will:
1. Load processed data from Task 1
2. Train baseline and ensemble models
3. Perform cross-validation
4. Compare models and select the best
5. Save all artifacts to `/models` directory

## Development Workflow

1. Create feature branch from `main`
2. Develop and test locally
3. Commit with descriptive messages
4. Push and create pull request
5. Code review and merge to `main`

## Contributing

This is an internal project for Adey Innovations Inc. Please follow:
- Git best practices
- Code documentation standards
- Business-first reasoning in all decisions

## License

Proprietary - Adey Innovations Inc.

