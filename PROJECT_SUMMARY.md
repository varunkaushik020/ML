# Chronic Disease AI Project Summary

This project provides a comprehensive framework for analyzing chronic disease datasets and building predictive models.

## Project Structure

```
chronic_disease_ai/
├── dataset/
│   ├── dataset1/           # Diabetes dataset (769 records)
│   ├── dataset2/           # Kidney disease dataset
│   ├── dataset3/           # Heart disease dataset (921 records)
│   ├── dataset4/           # Chronic disease dataset (3,499 records)
│   ├── dataset5/           # U.S. Chronic Disease Indicators (403,985 records)
│   ├── dataset6/           # COPD dataset (102 records)
│   └── dataset7/           # Chronic disease progression dataset
├── scripts/
│   ├── analyze_datasets.py # Main analysis script
│   ├── combine_datasets.py # Script to combine datasets
│   ├── demo_notebook.py    # Jupyter notebook demo
│   ├── test_datasets.py    # Simple dataset test script
│   └── chronic_disease_analysis.py # Comprehensive ML pipeline
├── api/                    # API deployment files
│   ├── main.py            # FastAPI application
│   ├── requirements.txt   # API dependencies
│   ├── Dockerfile         # Docker configuration
│   └── README.md          # API documentation
├── requirements.txt        # Project dependencies
├── README.md              # Project overview
├── DATASET_GUIDE.md       # Detailed dataset documentation
├── PROJECT_SUMMARY.md     # This file
└── run_analysis.bat       # Windows batch script to run analysis
```

## Key Components

### 1. Dataset Analysis Scripts
- `analyze_datasets.py`: Comprehensive analysis of all datasets with visualizations
- `combine_datasets.py`: Creates a unified dataset from multiple sources
- `test_datasets.py`: Simple script to verify dataset accessibility

### 2. Machine Learning Pipeline
- `chronic_disease_analysis.py`: Complete ML pipeline with all 10 phases:
  1. Data Understanding & Exploration
  2. Data Cleaning
  3. Imputation & Feature Engineering
  4. Data Splitting
  5. Model Training
  6. Hyperparameter Tuning
  7. Model Evaluation
  8. Explainability & Interpretability
  9. Saving & Deploying the Model
  10. Advanced Features

### 3. API Deployment
- `api/main.py`: FastAPI application for model deployment
- Containerized with Docker for easy deployment

### 4. Documentation
- `README.md`: Project overview and usage instructions
- `DATASET_GUIDE.md`: Detailed information about each dataset
- `api/README.md`: API deployment documentation

## Datasets Included

1. **Diabetes Dataset** (`dataset1`): Pima Indians Diabetes Database
2. **Kidney Disease Dataset** (`dataset2`)
3. **Heart Disease Dataset** (`dataset3`): UCI Heart Disease dataset
4. **Chronic Disease Dataset** (`dataset4`): Synthetic data with multiple biomarkers
5. **U.S. Chronic Disease Indicators** (`dataset5`): CDC data with 400K+ records
6. **COPD Dataset** (`dataset6`): Chronic Obstructive Pulmonary Disease data
7. **Chronic Disease Progression** (`dataset7`)

## How to Use

### Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the analysis:
   ```bash
   python scripts/analyze_datasets.py
   ```

3. On Windows, you can also double-click `run_analysis.bat`

### Advanced Usage
1. **Combine datasets**:
   ```bash
   python scripts/combine_datasets.py
   ```

2. **Run Jupyter notebook**:
   ```bash
   jupyter notebook scripts/demo_notebook.py
   ```

3. **Deploy API**:
   ```bash
   cd api
   pip install -r requirements.txt
   python main.py
   ```

## Features Implemented

### Data Science Pipeline
- Data loading and exploration
- Data cleaning and preprocessing
- Feature engineering and selection
- Model training and evaluation
- Hyperparameter tuning
- Model interpretability (SHAP values)
- Cross-validation and stratification

### Machine Learning Models
- Logistic Regression (baseline)
- Random Forest
- XGBoost
- Support for sklearn Pipelines and ColumnTransformer

### Medical ML Best Practices
- Proper train/validation/test splits
- Handling class imbalance
- Preventing data leakage
- Clinical feature engineering
- Model calibration checks
- Privacy considerations

### Deployment Ready
- Model serialization with joblib
- FastAPI REST endpoints
- Docker containerization
- Input/output examples for production

## Output Files

The scripts generate several output files:
- `confusion_matrix.png`: Confusion matrix visualization
- `feature_importance.png`: Feature importance chart
- `diabetes_analysis.png`: Diabetes dataset visualizations
- `heart_disease_analysis.png`: Heart disease dataset visualizations
- `chronic_disease_analysis.png`: Chronic disease dataset visualizations
- `unified_chronic_disease_dataset.csv`: Combined dataset

## Requirements

- Python 3.7+
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- xgboost, shap
- fastapi, uvicorn (for API)
- jupyter (for notebooks)

## Customization

To adapt this framework for your own datasets:
1. Replace the CSV files in the `dataset/` directory
2. Modify column names in the analysis scripts
3. Adjust target variables and preprocessing steps
4. Add domain-specific feature engineering
5. Implement specialized models for your use case

This project provides a solid foundation for chronic disease prediction research that can be extended for specific clinical applications.