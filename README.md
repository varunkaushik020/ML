# Chronic Disease AI Analysis

This repository contains scripts and tools for analyzing chronic disease datasets and building predictive models.

## Project Structure

```
chronic_disease_ai/
├── dataset/
│   ├── dataset1/           # Diabetes dataset
│   ├── dataset2/           # Kidney disease dataset
│   ├── dataset3/           # Heart disease dataset
│   ├── dataset4/           # Chronic disease dataset
│   ├── dataset5/           # U.S. Chronic Disease Indicators
│   ├── dataset6/           # Additional dataset
│   └── dataset7/           # Chronic disease progression
├── scripts/
│   ├── analyze_datasets.py # Main analysis script
│   ├── combine_datasets.py # Script to combine datasets
│   ├── demo_notebook.py    # Jupyter notebook demo
│   └── ...                 # Other analysis scripts
└── README.md
```

## Datasets

The project includes several chronic disease datasets:

1. **Diabetes Dataset** (`dataset1/diabetes.csv`) - Pima Indians Diabetes Database
2. **Heart Disease Dataset** (`dataset3/heart_disease_uci.csv`) - UCI Heart Disease dataset
3. **Chronic Disease Dataset** (`dataset4/chronic_disease_dataset.csv`) - Synthetic chronic disease data
4. **U.S. Chronic Disease Indicators** (`dataset5/U.S._Chronic_Disease_Indicators.csv`) - CDC data
5. **Additional Datasets** (`dataset6/`, `dataset7/`) - Various chronic disease related data

## Getting Started

### Prerequisites

Install the required packages:

```bash
pip install -r requirements.txt
```

### Running the Analysis

1. **Analyze individual datasets:**
   ```bash
   python scripts/analyze_datasets.py
   ```

2. **Combine multiple datasets:**
   ```bash
   python scripts/combine_datasets.py
   ```

3. **Run Jupyter notebook:**
   ```bash
   jupyter notebook scripts/demo_notebook.py
   ```

## Key Features

- **Data Exploration**: Load and explore multiple chronic disease datasets
- **Data Visualization**: Create charts and plots to understand data distributions
- **Data Preprocessing**: Clean and prepare data for machine learning
- **Model Building**: Train predictive models for disease classification
- **Model Evaluation**: Assess model performance with various metrics
- **Dataset Combination**: Merge multiple datasets for comprehensive analysis

## Scripts Overview

- `analyze_datasets.py`: Main script to analyze all datasets
- `combine_datasets.py`: Script to create a unified dataset from multiple sources
- `demo_notebook.py`: Jupyter notebook with step-by-step analysis
- `chronic_disease_analysis.py`: Comprehensive machine learning pipeline (in scripts/)

## Output

The scripts generate various outputs:

- **Visualizations**: Charts and plots saved as PNG files
- **Analysis Reports**: Console output with statistical summaries
- **Combined Dataset**: Unified dataset saved as CSV
- **Model Performance**: Classification reports and confusion matrices

## Customization

To analyze your own datasets:

1. Add your CSV files to the `dataset/` directory
2. Modify the script paths in `analyze_datasets.py` to include your datasets
3. Adjust column names and target variables as needed

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- jupyter (for notebook)

## License

This project is for educational purposes and does not have any specific license restrictions.