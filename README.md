# Fintech Fraud Detection System

**Adey Innovations Inc.** - Production-Grade Fraud Detection Pipeline

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

