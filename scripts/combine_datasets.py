import pandas as pd
import numpy as np
import os


def load_datasets():
    
    print("=== Loading Datasets ===")

    dataset_paths = [
        'dataset/dataset1/diabetes.csv',
        'dataset/dataset3/heart_disease_uci.csv',
        'dataset/dataset4/chronic_disease_dataset.csv',
        'dataset/dataset6/dataset.csv'
    ]

    datasets = {}

    for path in dataset_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                dataset_name = os.path.basename(os.path.dirname(path))
                datasets[dataset_name] = df
                print(f"Loaded {dataset_name}: {df.shape}")
            except Exception as e:
                print(f"Error loading {path}: {e}")
        else:
            print(f"File not found: {path}")

    return datasets


def analyze_dataset_structure(datasets):
    
    print("\n=== Dataset Structure Analysis ===")

    for name, df in datasets.items():
        print(f"\n{name}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Data types:\n{df.dtypes.value_counts()}")


def identify_common_features(datasets):
   
    print("\n=== Common Features Analysis ===")

 
    all_columns = set()
    dataset_columns = {}

    for name, df in datasets.items():
        columns = set(df.columns)
        all_columns.update(columns)
        dataset_columns[name] = columns

   
    common_columns = set.intersection(*dataset_columns.values())
    print(f"Common columns across all datasets: {common_columns}")

    for col in list(all_columns)[:10]:  
        datasets_with_col = [name for name,
                             cols in dataset_columns.items() if col in cols]
        print(f"  {col}: {datasets_with_col}")


def create_unified_dataset(datasets):
    
    print("\n=== Creating Unified Dataset ===")

    unified_data = []

    if 'dataset1' in datasets:
        df = datasets['dataset1']
        print(f"Processing diabetes dataset ({df.shape[0]} records)")

        for _, row in df.iterrows():
            record = {
                'source': 'diabetes',
                'age': row.get('Age', np.nan),
                'bmi': row.get('BMI', np.nan),
                'glucose': row.get('Glucose', np.nan),
                'blood_pressure': row.get('BloodPressure', np.nan),
                'insulin': row.get('Insulin', np.nan),
                'outcome': row.get('Outcome', np.nan)
            }
            unified_data.append(record)

    if 'dataset3' in datasets:
        df = datasets['dataset3']
        print(f"Processing heart disease dataset ({df.shape[0]} records)")

        for _, row in df.iterrows():
            record = {
                'source': 'heart_disease',
                'age': row.get('age', np.nan),
                'sex': 1 if row.get('sex') == 'Male' else 0,
                'chest_pain': row.get('cp', np.nan),
                'blood_pressure': row.get('trestbps', np.nan),
                'cholesterol': row.get('chol', np.nan),
                'heart_rate': row.get('thalch', np.nan),
                'outcome': 1 if row.get('num', 0) > 0 else 0
            }
            unified_data.append(record)

    if 'dataset4' in datasets:
        df = datasets['dataset4']
        print(f"Processing chronic disease dataset ({df.shape[0]} records)")

        for _, row in df.iterrows():
            record = {
                'source': 'chronic_disease',
                'age': row.get('age', np.nan),
                'gender': row.get('gender', np.nan),
                'bmi': row.get('bmi', np.nan),
                'blood_pressure': row.get('blood_pressure', np.nan),
                'cholesterol': row.get('cholesterol_level', np.nan),
                'glucose': row.get('glucose_level', np.nan),
                'smoking': row.get('smoking_status', np.nan),
                'outcome': row.get('target', np.nan)
            }
            unified_data.append(record)

    unified_df = pd.DataFrame(unified_data)
    print(f"\nUnified dataset created with {len(unified_df)} records")
    print(f"Columns: {list(unified_df.columns)}")

    print("\nUnified dataset info:")
    print(unified_df.info())

    unified_df.to_csv(
        'dataset/unified_chronic_disease_dataset.csv', index=False)
    print("\nUnified dataset saved to 'dataset/unified_chronic_disease_dataset.csv'")

    return unified_df


def analyze_unified_dataset(df):

    print("\n=== Unified Dataset Analysis ===")

    print(f"Dataset shape: {df.shape}")
    print(f"Sources: {df['source'].value_counts().to_dict()}")

    print("\nOutcome distribution by source:")
    outcome_by_source = df.groupby(
        'source')['outcome'].value_counts().unstack(fill_value=0)
    print(outcome_by_source)

    print("\nMissing values per column:")
    print(df.isnull().sum())


def main():
   
    print("Chronic Disease Dataset Combination")
    print("=" * 40)

    datasets = load_datasets()

    if not datasets:
        print("No datasets found!")
        return

    analyze_dataset_structure(datasets)

    identify_common_features(datasets)

    unified_df = create_unified_dataset(datasets)

    analyze_unified_dataset(unified_df)

    print("\nDataset combination process completed!")


if __name__ == "__main__":
    main()
