# Chronic Disease Dataset Guide

This guide explains the structure and content of the chronic disease datasets included in this repository.

## Dataset Overview

The repository contains 7 different chronic disease datasets stored in the `dataset/` directory:

1. **dataset1** - Diabetes dataset (769 records)
2. **dataset2** - Kidney disease dataset
3. **dataset3** - Heart disease dataset (921 records)
4. **dataset4** - Chronic disease dataset (3499 records)
5. **dataset5** - U.S. Chronic Disease Indicators (403,985 records)
6. **dataset6** - COPD dataset (102 records)
7. **dataset7** - Chronic disease progression dataset

## Detailed Dataset Descriptions

### Dataset 1: Diabetes Dataset
**File**: `dataset/dataset1/diabetes.csv`

This is the Pima Indians Diabetes Database, a classic dataset for diabetes prediction.

**Features**:
- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)^2)
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age in years
- Outcome: Class variable (0 or 1)

### Dataset 3: Heart Disease Dataset
**File**: `dataset/dataset3/heart_disease_uci.csv`

This dataset contains heart disease data from the UCI Machine Learning Repository.

**Features**:
- age: Age in years
- sex: Gender (Male/Female)
- cp: Chest pain type
- trestbps: Resting blood pressure (mm Hg)
- chol: Serum cholesterol (mg/dl)
- fbs: Fasting blood sugar > 120 mg/dl
- restecg: Resting electrocardiographic results
- thalach: Maximum heart rate achieved
- exang: Exercise induced angina
- oldpeak: ST depression induced by exercise
- slope: Slope of the peak exercise ST segment
- ca: Number of major vessels colored by fluoroscopy
- thal: Thalassemia
- num: Heart disease diagnosis (0-4)

### Dataset 4: Chronic Disease Dataset
**File**: `dataset/dataset4/chronic_disease_dataset.csv`

A synthetic dataset with various chronic disease indicators.

**Features**:
- age: Age
- gender: Gender (0/1)
- bmi: Body mass index
- blood_pressure: Blood pressure measurement
- cholesterol_level: Cholesterol level
- glucose_level: Glucose level
- physical_activity: Physical activity level
- smoking_status: Smoking status
- alcohol_intake: Alcohol intake
- family_history: Family history of disease
- biomarker_A, B, C, D: Various biomarkers
- target: Disease target (0-4)

### Dataset 6: COPD Dataset
**File**: `dataset/dataset6/dataset.csv`

Dataset related to Chronic Obstructive Pulmonary Disease (COPD).

**Features**:
- AGE: Patient age
- PackHistory: Smoking pack history
- COPDSEVERITY: COPD severity level
- MWT1, MWT2: 6-minute walk test results
- FEV1, FEV1PRED: Forced expiratory volume
- FVC, FVCPRED: Forced vital capacity
- CAT, HAD, SGRQ: Various assessment scores
- Various comorbidity indicators

## Working with the Datasets

### Loading Data in Python

```python
import pandas as pd

# Load diabetes dataset
diabetes_df = pd.read_csv('dataset/dataset1/diabetes.csv')

# Load heart disease dataset
heart_df = pd.read_csv('dataset/dataset3/heart_disease_uci.csv')

# Load chronic disease dataset
chronic_df = pd.read_csv('dataset/dataset4/chronic_disease_dataset.csv')
```

### Basic Exploration

```python
# Basic information about the dataset
print(df.shape)  # Dimensions
print(df.columns.tolist())  # Column names
print(df.info())  # Data types and missing values
print(df.describe())  # Statistical summary

# Target distribution
print(df['target_column'].value_counts())
```

## Combining Datasets

To work with multiple datasets together:

1. **Identify common features** across datasets
2. **Standardize column names** for consistency
3. **Create a unified schema** that incorporates relevant features
4. **Handle different data types** appropriately

## Best Practices for Chronic Disease Analysis

1. **Data Quality**: Check for missing values, outliers, and inconsistencies
2. **Feature Engineering**: Create meaningful clinical features
3. **Class Imbalance**: Address imbalanced target distributions
4. **Validation**: Use proper train/validation/test splits
5. **Interpretability**: Focus on explainable models for medical applications
6. **Ethics**: Consider privacy and bias issues in medical data

## Model Development Guidelines

1. **Start Simple**: Begin with basic models like logistic regression
2. **Feature Selection**: Identify most relevant clinical indicators
3. **Cross-Validation**: Use stratified k-fold for robust evaluation
4. **Metrics**: Focus on precision, recall, and AUC for medical applications
5. **Interpretation**: Use SHAP values or feature importance for explainability

## Privacy Considerations

These datasets are either:
- Publicly available research datasets
- Synthetic data generated for educational purposes
- Aggregated data without personal identifiers

When working with real medical data, always:
- Follow HIPAA guidelines
- Remove or encrypt personal identifiers
- Obtain proper consent and approval
- Implement secure data handling practices

## Next Steps

1. **Run the analysis scripts**:
   ```bash
   python scripts/analyze_datasets.py
   python scripts/combine_datasets.py
   ```

2. **Explore the Jupyter notebook**:
   ```bash
   jupyter notebook scripts/demo_notebook.py
   ```

3. **Customize for your needs**:
   - Modify column names to match your data
   - Adjust preprocessing steps
   - Add domain-specific features
   - Implement specialized models

This repository provides a foundation for chronic disease analysis that you can build upon for your specific research or clinical applications.